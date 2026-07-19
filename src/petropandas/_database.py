"""Thin REST client for the PetroDB microprobe database.

Provides easy CRUD operations on projects, samples, spots, areas,
and profiles via the PetroDB HTTP API.

Usage::

    from petropandas import PetroDB

    db = PetroDB("https://petrodb.geoaltay.eu", "user", "password")
    project = db.projects(name="MyProject")
    sample = project.samples(name="DB250")
    df = sample.spots.df(mineral="Grt")
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

__all__ = [
    "APIError",
    "Area",
    "AreaRecords",
    "AuthError",
    "NotFoundError",
    "PetroDB",
    "PetroDBError",
    "Profile",
    "ProfileSpot",
    "ProfileSpotRecords",
    "Project",
    "ReadOnlyError",
    "Sample",
    "Spot",
    "SpotRecords",
]


# ── exceptions ──────────────────────────────────────────────────────────


class PetroDBError(Exception):
    """Base exception for petrodb operations."""


class AuthError(PetroDBError):
    """Authentication failure."""


class NotFoundError(PetroDBError):
    """Resource not found."""


class APIError(PetroDBError):
    """API returned an error response."""


class ReadOnlyError(PetroDBError):
    """A mutating request was blocked because the client is read-only."""


# ── helpers ─────────────────────────────────────────────────────────────


def _zero_negative_nan(x: Any) -> Any:
    """Replace non-positive numeric values with NaN."""
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return x if x > 0 else float("nan")
    return x


def _fetch_concurrently(
    items: Iterable[Any], getter: Callable[[Any], Any], max_workers: int = 8
) -> list[Any]:
    """Call *getter* on each item concurrently, skipping ``APIError`` results.

    Args:
        items: Items to fetch a sub-resource for (e.g. samples in a project).
        getter: Callable applied to each item (e.g. ``lambda s: s.spots``).
        max_workers: Maximum concurrent worker threads.

    Returns:
        Results in the same order as *items*, with items that raised
        :class:`APIError` omitted.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(getter, item) for item in items]
        results = []
        for future in futures:
            try:
                results.append(future.result())
            except APIError:
                continue
    return results


def _read_dotenv(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE .env file.

    Blank lines and lines starting with ``#`` are ignored.

    Args:
        path: Path to the .env file.

    Returns:
        Mapping of variable name to value, or empty if *path* doesn't exist.
    """
    if not path.is_file():
        return {}
    values = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        values[key.strip()] = val.strip().strip("'\"")
    return values


def _resolve_setting(
    value: str | None, env_var: str, dotenv: dict[str, str]
) -> str | None:
    """Fill in a missing setting from an env var, then a parsed .env file."""
    return (
        value if value is not None else os.environ.get(env_var) or dotenv.get(env_var)
    )


def _resolve_connection(
    api_url: str | None, username: str | None, password: str | None
) -> tuple[str | None, str | None, str | None]:
    """Fill in missing connection settings from env vars, then a .env file.

    Checks (in order) the given value, the ``PETRODBAPI``/``PETRODBUSER``/
    ``PETRODBPASSWORD`` environment variables, then a ``.env`` file in the
    current working directory.

    Args:
        api_url: Base URL of the PetroDB server, or None to resolve it.
        username: Auth username, or None to resolve it.
        password: Auth password, or None to resolve it.

    Returns:
        Tuple of (api_url, username, password), each possibly still None if
        unresolved.
    """
    if api_url is not None and username is not None and password is not None:
        return api_url, username, password
    dotenv = _read_dotenv(Path(".env"))
    return (
        _resolve_setting(api_url, "PETRODBAPI", dotenv),
        _resolve_setting(username, "PETRODBUSER", dotenv),
        _resolve_setting(password, "PETRODBPASSWORD", dotenv),
    )


# ── low-level HTTP transport ────────────────────────────────────────────


_MUTATING_METHODS = frozenset({"POST", "PUT", "DELETE"})


class _PetroAPI:
    """Low-level HTTP client for the PetroDB REST API.

    Args:
        api_url: Base URL of the PetroDB server. If None, resolved from the
            ``PETRODBAPI`` environment variable, then a ``.env`` file in the
            current working directory.
        username: Auth username. If None, resolved from the ``PETRODBUSER``
            environment variable, then a ``.env`` file in the current
            working directory.
        password: Auth password. If None, resolved from the
            ``PETRODBPASSWORD`` environment variable, then a ``.env`` file
            in the current working directory.
        timeout: Request timeout in seconds.
        read_only: If True (default), block any POST/PUT/DELETE request with
            :class:`ReadOnlyError` before it reaches the network. Authentication
            (login/re-login) is unaffected.

    Raises:
        AuthError: If api_url/username/password can't be resolved, or if
            authentication fails.
    """

    def __init__(
        self,
        api_url: str | None = None,
        username: str | None = None,
        password: str | None = None,
        timeout: int = 30,
        read_only: bool = True,
    ):
        api_url, username, password = _resolve_connection(api_url, username, password)
        if api_url is None or username is None or password is None:
            raise AuthError(
                "api_url/username/password not given and none found in "
                "PETRODBAPI/PETRODBUSER/PETRODBPASSWORD environment "
                "variables or a .env file"
            )
        self._session = requests.Session()
        self._api_url = api_url.rstrip("/")
        self._username = username
        self._password = password
        self._timeout = timeout
        self._read_only = read_only
        self._token: str | None = None
        self._login()

    def _login(self) -> None:
        response = self._session.post(
            f"{self._api_url}/token",
            data={"username": self._username, "password": self._password},
            timeout=self._timeout,
        )
        if response.ok:
            self._token = response.json().get("access_token")
            logger.info("Authenticated to petrodb at %s", self._api_url)
        else:
            raise AuthError(f"Authentication failed (HTTP {response.status_code})")

    def _reauthenticate(self) -> bool:
        """Attempt to refresh the JWT token."""
        try:
            self._login()
            return True
        except AuthError:
            return False

    def _authorize(self) -> dict[str, str]:
        if self._token is None:
            raise AuthError("Not logged in")
        return {"Authorization": f"Bearer {self._token}"}

    def _request(
        self, method: str, path: str, json_data: dict | list | None = None
    ) -> requests.Response:
        if self._read_only and method.upper() in _MUTATING_METHODS:
            msg = (
                f"Cannot {method.upper()} {path} — client is read-only (read_only=True)"
            )
            raise ReadOnlyError(msg)
        headers = self._authorize()
        url = f"{self._api_url}/api{path}"
        kwargs: dict = {"headers": headers, "timeout": self._timeout}
        if json_data is not None:
            kwargs["json"] = json_data

        http_method = getattr(self._session, method.lower())
        response = http_method(url, **kwargs)

        if response.status_code == 401:
            logger.info("Token expired, re-authenticating...")
            if not self._reauthenticate():
                raise AuthError("Token expired and re-authentication failed")
            headers = self._authorize()
            kwargs["headers"] = headers
            response = http_method(url, **kwargs)

        return response

    def get(self, path: str) -> requests.Response:
        """Send an authenticated GET request."""
        return self._request("GET", path)

    def post(self, path: str, data: dict | list) -> requests.Response:
        """Send an authenticated POST request with JSON body."""
        return self._request("POST", path, json_data=data)

    def put(self, path: str, data: dict) -> requests.Response:
        """Send an authenticated PUT request with JSON body."""
        return self._request("PUT", path, json_data=data)

    def delete(self, path: str) -> requests.Response:
        """Send an authenticated DELETE request."""
        return self._request("DELETE", path)

    def _check(
        self, response: requests.Response, default_msg: str = "Unknown error"
    ) -> Any:
        """Validate a response and return its JSON body.

        Raises:
            APIError: If the response is not successful.
        """
        if response.ok:
            return response.json()
        detail = response.json().get("detail", default_msg)
        logger.error("API error: %s (status %s)", detail, response.status_code)
        raise APIError(detail)


# ── high-level facade ───────────────────────────────────────────────────


class PetroDB:
    """High-level access to an online PetroDB instance.

    Args:
        api_url: Base URL of the PetroDB server. If None, resolved from the
            ``PETRODBAPI`` environment variable, then a ``.env`` file in the
            current working directory.
        username: Auth username. If None, resolved from the ``PETRODBUSER``
            environment variable, then a ``.env`` file in the current
            working directory.
        password: Auth password. If None, resolved from the
            ``PETRODBPASSWORD`` environment variable, then a ``.env`` file
            in the current working directory.
        timeout: Request timeout in seconds.
        read_only: If True (default), any create/update/delete call raises
            :class:`ReadOnlyError` instead of reaching the network. Pass
            ``read_only=False`` to allow database mutations.

    Raises:
        AuthError: If api_url/username/password can't be resolved, or if
            authentication fails.

    Example::

        with PetroDB("https://petrodb.geoaltay.eu", "user", "pass") as db:
            for project in db.projects():
                print(project.name)

        # allow writes
        with PetroDB(url, user, pw, read_only=False) as db:
            db.create_project("New project")

        # resolved from PETRODBAPI/PETRODBUSER/PETRODBPASSWORD env vars or .env
        with PetroDB() as db:
            ...
    """

    def __init__(
        self,
        api_url: str | None = None,
        username: str | None = None,
        password: str | None = None,
        timeout: int = 30,
        read_only: bool = True,
    ):
        self._db = _PetroAPI(api_url, username, password, timeout, read_only=read_only)

    def __enter__(self) -> PetroDB:
        return self

    def __exit__(self, *args: object) -> None:
        self._db._session.close()

    def __repr__(self) -> str:
        return "PetroDB"

    def _check(
        self, response: requests.Response, default_msg: str = "Unknown error"
    ) -> Any:
        return self._db._check(response, default_msg)

    def projects(self, name: str | None = None) -> Project | list[Project]:
        """List or search projects.

        Args:
            name: If given, search for a project by name.

        Returns:
            A single :class:`Project` if *name* is given, otherwise a list.

        Raises:
            APIError: If no matching project was found.
        """
        if name is not None:
            return Project(
                self._db,
                project=self._check(
                    self._db.get(f"/search/project/{name}"), "Project not found"
                ),
            )
        return [
            Project(self._db, project=p)
            for p in self._check(self._db.get("/projects/"), "Projects not found")
        ]

    def create_project(self, name: str, description: str = "") -> Project:
        """Create a new project.

        Args:
            name: Project name.
            description: Optional description.

        Returns:
            The created :class:`Project`.

        Raises:
            APIError: If creation failed.
        """
        return Project(
            self._db,
            project=self._check(
                self._db.post("/project/", {"name": name, "description": description}),
                "Project not created",
            ),
        )


# ── entity classes ──────────────────────────────────────────────────────


class _Entity:
    """Shared id/payload/CRUD boilerplate for PetroDB entity wrappers.

    Subclasses set ``_resource_name`` (used in default error messages) and
    override the ``_url`` property with this entity's API path.
    """

    _db: _PetroAPI
    data: dict[str, Any]
    _resource_name: str = "Resource"

    @property
    def id(self) -> int:
        """Entity ID."""
        return self.data["id"]

    @property
    def _payload(self) -> dict:
        """Entity data excluding the ID field."""
        return {k: v for k, v in self.data.items() if k != "id"}

    @property
    def _url(self) -> str:
        """API path for this entity instance. Overridden by subclasses."""
        raise NotImplementedError

    def _check(self, response: requests.Response, default_msg: str) -> Any:
        return self._db._check(response, default_msg)

    def delete(self) -> dict:
        """Delete this entity."""
        return self._check(
            self._db.delete(self._url), f"{self._resource_name} not deleted"
        )

    def update(self) -> dict:
        """Update this entity in the database."""
        return self._check(
            self._db.put(self._url, self._payload), f"{self._resource_name} not updated"
        )


class Project(_Entity):
    """A PetroDB project.

    Attributes:
        data: Raw project attributes from the API.
    """

    _resource_name = "Project"

    def __init__(self, db: _PetroAPI, project: dict):
        self._db = db
        self.data = dict(project)
        self._project_id: int = self.data["id"]

    def __repr__(self) -> str:
        return f"{self.name}"

    @property
    def _url(self) -> str:
        return f"/project/{self._project_id}"

    @property
    def name(self) -> str:
        """Project name."""
        return self.data["name"]

    @name.setter
    def name(self, name: str) -> None:
        self.data["name"] = name

    @property
    def description(self) -> str | None:
        """Project description."""
        return self.data["description"]

    @description.setter
    def description(self, desc: str | None) -> None:
        self.data["description"] = desc

    def samples(self, name: str | None = None) -> Sample | list[Sample]:
        """List or search samples within this project.

        Args:
            name: If given, search for a sample by name.

        Returns:
            A single :class:`Sample` if *name* is given, otherwise a list.
        """
        if name is not None:
            return Sample(
                self._db,
                self._project_id,
                sample=self._check(
                    self._db.get(f"/search/sample/{self._project_id}/{name}"),
                    "Sample not found",
                ),
            )
        return [
            Sample(self._db, self._project_id, sample=s)
            for s in self._check(
                self._db.get(f"/samples/{self._project_id}"), "Samples not found"
            )
        ]

    def create_sample(self, name: str, description: str = "") -> Sample:
        """Create a new sample in this project."""
        return Sample(
            self._db,
            self._project_id,
            sample=self._check(
                self._db.post(
                    f"/sample/{self._project_id}",
                    {"name": name, "description": description},
                ),
                "Sample not created",
            ),
        )

    @cached_property
    def spots(self) -> SpotRecords:
        """All spots across all samples in this project (fetched concurrently)."""
        records: dict[int, Spot] = {}
        names: list[str] = []
        for s in _fetch_concurrently(self.samples(), lambda sample: sample.spots):
            records = records | s.records
            names += s.sample_names
        return SpotRecords(records, names)

    @cached_property
    def areas(self) -> AreaRecords:
        """All areas across all samples in this project (fetched concurrently)."""
        records: dict[int, Area] = {}
        names: list[str] = []
        for s in _fetch_concurrently(self.samples(), lambda sample: sample.areas):
            records = records | s.records
            names += s.sample_names
        return AreaRecords(records, names)

    def reset(self) -> None:
        """Reset cached properties to access updated data."""
        self.__dict__.pop("spots", None)
        self.__dict__.pop("areas", None)

    def mineral_data(self, mineral: str) -> pd.DataFrame:
        """Return spots and profile spots of a given mineral as a DataFrame."""
        res: list[pd.DataFrame] = []
        for sample in self.samples():
            spots = sample.spots.df(mineral=mineral, sample_name=True)
            if not spots.empty:
                spots["kind"] = "spot"
                res.append(spots)
            profiles = sample.profiles(mineral=mineral)
            for profile in profiles:
                spots = profile.spots.df(sample_name=True)
                spots["kind"] = "profile"
                res.append(spots)
        if res:
            return pd.concat(res)
        raise NotFoundError("No data found.")

    def add_user(self, username: str) -> dict:
        """Add user access to this project."""
        return self._check(
            self._db.put(
                f"/project/{self._project_id}/adduser", {"username": username}
            ),
            "User not added",
        )

    def remove_user(self, username: str) -> dict:
        """Remove user access from this project."""
        return self._check(
            self._db.put(
                f"/project/{self._project_id}/removeuser",
                {"username": username},
            ),
            "User not removed",
        )


class Sample(_Entity):
    """A PetroDB sample within a project.

    Attributes:
        data: Raw sample attributes from the API.
    """

    _resource_name = "Sample"

    def __init__(self, db: _PetroAPI, project_id: int, sample: dict):
        self._db = db
        self._project_id = project_id
        self.data = dict(sample)
        self._sample_id: int = self.data["id"]

    def __repr__(self) -> str:
        return f"{self.name}"

    @property
    def _url(self) -> str:
        return f"/sample/{self._project_id}/{self._sample_id}"

    @property
    def name(self) -> str:
        """Sample name."""
        return self.data["name"]

    @name.setter
    def name(self, name: str) -> None:
        self.data["name"] = name

    @property
    def description(self) -> str | None:
        """Sample description."""
        return self.data["description"]

    @description.setter
    def description(self, desc: str | None) -> None:
        self.data["description"] = desc

    @cached_property
    def spots(self) -> SpotRecords:
        """All spots in this sample."""
        spots = {
            res["id"]: Spot(self._db, self._project_id, self._sample_id, spot=res)
            for res in self._check(
                self._db.get(f"/spots/{self._project_id}/{self._sample_id}"),
                "Spots not found",
            )
        }
        return SpotRecords(spots, [self.name] * len(spots))

    def spot(self, spot_id: int) -> Spot:
        """Get a single spot by ID."""
        return Spot(
            self._db,
            self._project_id,
            self._sample_id,
            spot=self._check(
                self._db.get(f"/spot/{self._project_id}/{self._sample_id}/{spot_id}"),
                "Spot not found",
            ),
        )

    def create_spot(self, label: str, mineral: str, values: dict) -> Spot:
        """Create a single spot in this sample.

        Args:
            label: Spot label.
            mineral: Mineral name (Kretz abbreviations recommended).
            values: Oxide wt% values.
        """
        return Spot(
            self._db,
            self._project_id,
            self._sample_id,
            spot=self._check(
                self._db.post(
                    f"/spot/{self._project_id}/{self._sample_id}",
                    {"label": label, "mineral": mineral, "values": values},
                ),
                "Spot not created",
            ),
        )

    def create_spots(
        self,
        df: pd.DataFrame,
        label_col: str | None = None,
        mineral_col: str | None = None,
    ) -> SpotRecords:
        """Create multiple spots from a DataFrame.

        Args:
            df: DataFrame with oxide values.
            label_col: Column to use as spot label.  If *None*, uses the index.
            mineral_col: Column to use as mineral name.  If *None*, mineral is empty.
        """
        df = df.copy()
        if label_col is None:
            labels = pd.Series(df.index.astype(str), index=df.index)
        else:
            labels = df[label_col].str.strip()
            df = df.drop(columns=[label_col])
        if mineral_col is None:
            minerals = pd.Series("", index=df.index)
        else:
            minerals = df[mineral_col].str.strip()
            df = df.drop(columns=[mineral_col])
        spots = [
            {
                "label": label,
                "mineral": mineral,
                "values": row.apply(_zero_negative_nan).dropna().to_dict(),
            }
            for label, mineral, (_, row) in zip(labels, minerals, df.iterrows())
        ]
        spot_map = {
            res["id"]: Spot(self._db, self._project_id, self._sample_id, spot=res)
            for res in self._check(
                self._db.post(f"/spots/{self._project_id}/{self._sample_id}", spots),
                "Spots not created",
            )
        }
        return SpotRecords(spot_map, [self.name] * len(spot_map))

    @cached_property
    def areas(self) -> AreaRecords:
        """All areas in this sample."""
        areas = {
            res["id"]: Area(self._db, self._project_id, self._sample_id, area=res)
            for res in self._check(
                self._db.get(f"/areas/{self._project_id}/{self._sample_id}"),
                "Areas not found",
            )
        }
        return AreaRecords(areas, [self.name] * len(areas))

    def area(self, area_id: int) -> Area:
        """Get a single area by ID."""
        return Area(
            self._db,
            self._project_id,
            self._sample_id,
            area=self._check(
                self._db.get(f"/area/{self._project_id}/{self._sample_id}/{area_id}"),
                "Area not found",
            ),
        )

    def create_area(self, label: str, values: dict) -> Area:
        """Create a single area in this sample."""
        return Area(
            self._db,
            self._project_id,
            self._sample_id,
            area=self._check(
                self._db.post(
                    f"/area/{self._project_id}/{self._sample_id}",
                    {"label": label, "values": values},
                ),
                "Area not created",
            ),
        )

    def create_areas(
        self, df: pd.DataFrame, label_col: str | None = None
    ) -> AreaRecords:
        """Create multiple areas from a DataFrame.

        Args:
            df: DataFrame with oxide values.
            label_col: Column to use as area label.  If *None*, uses the index.
        """
        df = df.copy()
        if label_col is None:
            labels = pd.Series(df.index.astype(str), index=df.index)
        else:
            labels = df[label_col].str.strip()
            df = df.drop(columns=[label_col])
        areas = [
            {
                "label": label,
                "values": row.apply(_zero_negative_nan).dropna().to_dict(),
            }
            for label, (_, row) in zip(labels, df.iterrows())
        ]
        area_map = {
            res["id"]: Area(self._db, self._project_id, self._sample_id, area=res)
            for res in self._check(
                self._db.post(f"/areas/{self._project_id}/{self._sample_id}", areas),
                "Areas not created",
            )
        }
        return AreaRecords(area_map, [self.name] * len(area_map))

    def profiles(
        self, label: str | None = None, mineral: str | None = None
    ) -> Profile | list[Profile]:
        """List or search profiles within this sample.

        Args:
            label: If given, search for a profile by label.
            mineral: If given, filter profiles by mineral name.
        """
        if label is not None:
            data = self._check(
                self._db.get(
                    f"/search/profile/{self._project_id}/{self._sample_id}/{label}"
                ),
                "Profile not found",
            )
            if mineral is not None and data.get("mineral") != mineral:
                raise NotFoundError(f"Profile with {label=} and {mineral=} not found.")
            return Profile(
                self._db,
                self._project_id,
                self._sample_id,
                self.name,
                profile=data,
            )
        return [
            Profile(
                self._db,
                self._project_id,
                self._sample_id,
                self.name,
                profile=p,
            )
            for p in self._check(
                self._db.get(f"/profiles/{self._project_id}/{self._sample_id}"),
                "Profiles not found",
            )
            if mineral is None or p.get("mineral") == mineral
        ]

    def create_profile(self, label: str, mineral: str | None = None) -> Profile:
        """Create a new profile in this sample."""
        return Profile(
            self._db,
            self._project_id,
            self._sample_id,
            self.name,
            profile=self._check(
                self._db.post(
                    f"/profile/{self._project_id}/{self._sample_id}",
                    {"label": label, "mineral": mineral},
                ),
                "Profile not created",
            ),
        )

    @cached_property
    def profilespots(self) -> ProfileSpotRecords:
        """All profile spots across all profiles in this sample (fetched concurrently)."""
        records: dict[int, ProfileSpot] = {}
        names: list[str] = []
        for s in _fetch_concurrently(self.profiles(), lambda profile: profile.spots):
            records = records | s.records
            names += s.sample_names
        return ProfileSpotRecords(records, names)

    def reset(self) -> None:
        """Reset cached properties to access updated data."""
        self.__dict__.pop("spots", None)
        self.__dict__.pop("areas", None)
        self.__dict__.pop("profilespots", None)

    def mineral_data(self, mineral: str) -> pd.DataFrame:
        """Return spots and profile spots of a given mineral as a DataFrame."""
        res: list[pd.DataFrame] = []
        spots = self.spots.df(mineral=mineral, sample_name=True)
        if not spots.empty:
            spots["kind"] = "spot"
            res.append(spots)
        profiles = self.profiles(mineral=mineral)
        for profile in profiles:
            spots = profile.spots.df(sample_name=True)
            spots["kind"] = "profile"
            res.append(spots)
        if res:
            return pd.concat(res)
        raise NotFoundError("No data found.")


class Spot(_Entity):
    """A PetroDB spot within a sample.

    Attributes:
        data: Raw spot attributes from the API.
    """

    _resource_name = "Spot"

    def __init__(self, db: _PetroAPI, project_id: int, sample_id: int, spot: dict):
        self._db = db
        self._project_id = project_id
        self._sample_id = sample_id
        self.data = dict(spot)
        self._spot_id: int = self.data["id"]

    def __repr__(self) -> str:
        return f"{self.label} ({self.mineral})"

    @property
    def _url(self) -> str:
        return f"/spot/{self._project_id}/{self._sample_id}/{self._spot_id}"

    @property
    def label(self) -> str:
        """Spot label."""
        return self.data["label"]

    @label.setter
    def label(self, lbl: str) -> None:
        self.data["label"] = lbl

    @property
    def mineral(self) -> str | None:
        """Mineral name."""
        return self.data["mineral"]

    @mineral.setter
    def mineral(self, m: str | None) -> None:
        self.data["mineral"] = m


class Area(_Entity):
    """A PetroDB area within a sample.

    Attributes:
        data: Raw area attributes from the API.
    """

    _resource_name = "Area"

    def __init__(self, db: _PetroAPI, project_id: int, sample_id: int, area: dict):
        self._db = db
        self._project_id = project_id
        self._sample_id = sample_id
        self.data = dict(area)
        self._area_id: int = self.data["id"]

    def __repr__(self) -> str:
        return f"{self.label}"

    @property
    def _url(self) -> str:
        return f"/area/{self._project_id}/{self._sample_id}/{self._area_id}"

    @property
    def label(self) -> str:
        """Area label."""
        return self.data["label"]

    @label.setter
    def label(self, lbl: str) -> None:
        self.data["label"] = lbl


class Profile(_Entity):
    """A PetroDB profile within a sample.

    Attributes:
        data: Raw profile attributes from the API.
        samplename: Name of the parent sample.  Passed explicitly because
            profiles are created via the sample, not fetched hierarchically
            from a parent entity object.
    """

    _resource_name = "Profile"

    def __init__(
        self,
        db: _PetroAPI,
        project_id: int,
        sample_id: int,
        samplename: str,
        profile: dict,
    ):
        self._db = db
        self._project_id = project_id
        self._sample_id = sample_id
        self.samplename = samplename
        self.data = dict(profile)
        self._profile_id: int = self.data["id"]

    def __repr__(self) -> str:
        return f"{self.label} ({self.mineral})"

    @property
    def _url(self) -> str:
        return f"/profile/{self._project_id}/{self._sample_id}/{self._profile_id}"

    @property
    def label(self) -> str:
        """Profile label."""
        return self.data["label"]

    @label.setter
    def label(self, lbl: str) -> None:
        self.data["label"] = lbl

    @property
    def mineral(self) -> str | None:
        """Mineral name."""
        return self.data["mineral"]

    @mineral.setter
    def mineral(self, m: str | None) -> None:
        self.data["mineral"] = m

    @cached_property
    def spots(self) -> ProfileSpotRecords:
        """All spots in this profile."""
        spots = {
            res["id"]: ProfileSpot(
                self._db,
                self._project_id,
                self._sample_id,
                self._profile_id,
                spot=res,
            )
            for res in self._check(
                self._db.get(
                    f"/profilespots/{self._project_id}/{self._sample_id}"
                    f"/{self._profile_id}"
                ),
                "Profile spots not found",
            )
        }
        for spot in spots.values():
            spot.data["label"] = self.label
            spot.data["mineral"] = self.mineral
        return ProfileSpotRecords(spots, [self.samplename] * len(spots))

    def spot(self, spot_id: int) -> ProfileSpot:
        """Get a single profile spot by ID."""
        return ProfileSpot(
            self._db,
            self._project_id,
            self._sample_id,
            self._profile_id,
            spot=self._check(
                self._db.get(
                    f"/profilespot/{self._project_id}/{self._sample_id}"
                    f"/{self._profile_id}/{spot_id}"
                ),
                "Profile spot not found",
            ),
        )

    def create_spot(self, index: int, values: dict) -> ProfileSpot:
        """Create a single profile spot.

        Args:
            index: Position order on the profile.
            values: Oxide wt% values.
        """
        return ProfileSpot(
            self._db,
            self._project_id,
            self._sample_id,
            self._profile_id,
            spot=self._check(
                self._db.post(
                    f"/profilespot/{self._project_id}/{self._sample_id}"
                    f"/{self._profile_id}",
                    {"index": index, "values": values},
                ),
                "Profile spot not created",
            ),
        )

    def create_spots(self, df: pd.DataFrame) -> ProfileSpotRecords:
        """Create multiple profile spots from a DataFrame.

        Args:
            df: DataFrame with oxide values.  Index is used as spot order.
        """
        df = df.copy()
        profilespots = [
            {
                "index": index,
                "values": row.apply(_zero_negative_nan).dropna().to_dict(),
            }
            for index, row in df.iterrows()
        ]
        spots = {
            res["id"]: ProfileSpot(
                self._db,
                self._project_id,
                self._sample_id,
                self._profile_id,
                spot=res,
            )
            for res in self._check(
                self._db.post(
                    f"/profilespots/{self._project_id}/{self._sample_id}"
                    f"/{self._profile_id}",
                    profilespots,
                ),
                "Profile spots not created",
            )
        }
        for spot in spots.values():
            spot.data["label"] = self.label
            spot.data["mineral"] = self.mineral
        return ProfileSpotRecords(spots, [self.samplename] * len(spots))

    def reset(self) -> None:
        """Reset cached property spots to access updated data."""
        self.__dict__.pop("spots", None)


class ProfileSpot(_Entity):
    """A PetroDB profile spot within a profile.

    Attributes:
        data: Raw profile spot attributes from the API.
    """

    _resource_name = "Profile spot"

    def __init__(
        self,
        db: _PetroAPI,
        project_id: int,
        sample_id: int,
        profile_id: int,
        spot: dict,
    ):
        self._db = db
        self._project_id = project_id
        self._sample_id = sample_id
        self._profile_id = profile_id
        self.data = dict(spot)
        self._profilespot_id: int = self.data["id"]

    def __repr__(self) -> str:
        return f"Spot {self.index}"

    @property
    def _url(self) -> str:
        return (
            f"/profilespot/{self._project_id}/{self._sample_id}"
            f"/{self._profile_id}/{self._profilespot_id}"
        )

    @property
    def index(self) -> int:
        """Position index on the profile."""
        return self.data["index"]

    @index.setter
    def index(self, idx: int) -> None:
        self.data["index"] = idx


# ── records collections ─────────────────────────────────────────────────


class Records:
    """Base collection of records with DataFrame export."""

    def __init__(self, records: dict[int, Any], sample_names: list[str]):
        self.records = records
        self.sample_names = sample_names
        self.cols: list[str] = []

    def __getitem__(self, key: int) -> Any:
        return self.records[key]

    def df(self, **kwargs: Any) -> pd.DataFrame:
        """Convert records to a DataFrame.

        Keyword arguments ``col=val`` filter rows where *col* equals *val*.
        Only columns listed in :attr:`cols` are available for filtering.

        Returns:
            DataFrame sorted by index.
        """
        res = pd.DataFrame(
            {k: v.data["values"] for k, v in self.records.items()}
        ).T.infer_objects()
        res["sample"] = self.sample_names
        for col in self.cols:
            res[col] = [row.data[col] for row in self.records.values()]
        for col, val in kwargs.items():
            if col in self.cols:
                res = res[res[col] == val]
        return res.sort_index().copy()


class SpotRecords(Records):
    """Collection of spot records."""

    def __init__(self, records: dict[int, Spot], sample_names: list[str]):
        super().__init__(records, sample_names)
        self.cols = ["label", "mineral"]

    def __repr__(self) -> str:
        return f"{len(self.records)} spots"


class AreaRecords(Records):
    """Collection of area records."""

    def __init__(self, records: dict[int, Area], sample_names: list[str]):
        super().__init__(records, sample_names)
        self.cols = ["label"]

    def __repr__(self) -> str:
        return f"{len(self.records)} areas"


class ProfileSpotRecords(Records):
    """Collection of profile spot records."""

    def __init__(self, records: dict[int, ProfileSpot], sample_names: list[str]):
        super().__init__(records, sample_names)
        self.cols = ["label", "mineral"]

    def __repr__(self) -> str:
        return f"{len(self.records)} profile spots"

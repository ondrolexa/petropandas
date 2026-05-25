"""Client for postresql petrodb database API

This module provide easy CRUD operations on petrodb database

Usage examples:

    from petropandas import pd
    from petropandas.database import PetroDB

    db = PetroDB('http://127.0.0.1:8000', 'user', 'password')
    project = db.projects(name="MyProject")
    sample = project.samples(name="DB250")
    df = sample.spots.df(mineral="Grt")

"""

import logging
from functools import cached_property
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)


# ── exceptions ──────────────────────────────────────────────────────────


class PetroDBError(Exception):
    """Base exception for petrodb operations."""


class AuthError(PetroDBError):
    """Authentication failure."""


class NotFoundError(PetroDBError):
    """Resource not found."""


class APIError(PetroDBError):
    """API returned an error response."""


# ── helpers ─────────────────────────────────────────────────────────────


def zero_negative_nan(x: Any) -> Any:
    if isinstance(x, bool):
        return x
    if isinstance(x, complex):
        return x if x.real > 0 else float("nan")
    if isinstance(x, (int, float)):
        return x if x > 0 else float("nan")
    return x


# ── low-level HTTP transport ────────────────────────────────────────────


class PetroAPI:
    """Low-level HTTP client for the petrodb REST API."""

    def __init__(self, api_url: str, username: str, password: str, timeout: int = 30):
        self._session = requests.Session()
        self._api_url = api_url
        self._username = username
        self._password = password
        self._timeout = timeout
        self._token: str | None = None
        self.logged = False
        self._login()

    def _login(self) -> None:
        response = self._session.post(
            f"{self._api_url}/token",
            data={"username": self._username, "password": self._password},
            timeout=self._timeout,
        )
        if response.ok:
            self._token = response.json().get("access_token")
            self.logged = True
            logger.info("Authenticated to petrodb at %s", self._api_url)
        else:
            self.logged = False
            logger.warning("Authentication to petrodb at %s failed", self._api_url)

    def _reauthenticate(self) -> bool:
        """Attempt to refresh the JWT token."""
        self._login()
        return self.logged

    def _authorize(self) -> dict[str, str]:
        if not self.logged or self._token is None:
            raise AuthError("Not logged in")
        return {"Authorization": f"Bearer {self._token}"}

    def _request(
        self, method: str, path: str, json_data: dict | list | None = None
    ) -> requests.Response:
        headers = self._authorize()
        url = f"{self._api_url}/api{path}"
        kwargs: dict = {"headers": headers, "timeout": self._timeout}
        if json_data is not None:
            kwargs["json"] = json_data

        http_method = getattr(self._session, method.lower())
        response = http_method(url, **kwargs)

        if response.status_code == 401:
            logger.info("Token expired, re-authenticating…")
            if self._reauthenticate():
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


# ── high-level facade ───────────────────────────────────────────────────


class PetroDB:
    """Petro database instance

    High-level access to online Petro database

    """

    def __init__(self, api_url: str, username: str, password: str, timeout: int = 30):
        self._db = PetroAPI(api_url, username, password, timeout)

    def __enter__(self):
        return self

    def __exit__(self, *args: object) -> None:
        self._db._session.close()

    def __repr__(self) -> str:
        return f"PetroDB {'OK' if self._db.logged else 'Not logged'}"

    @property
    def logged(self) -> bool:
        """Return True when API credentials are ok."""
        return self._db.logged

    def _check(self, response: requests.Response) -> dict:
        if response.ok:
            return response.json()
        detail = response.json().get("detail", "Unknown error")
        logger.error("API error: %s (status %s)", detail, response.status_code)
        raise APIError(detail)

    def projects(self, name: str | None = None):
        """Get project from database

        Args:
            name (str, optional): search for project by name

        Returns:
            Project instance or list of all projects if the name is not provided.

        Raises:
            APIError: If project(s) was not found.

        """
        if name is not None:
            response = self._db.get(f"/search/project/{name}")
            if response.ok:
                return PetroDBProject(self._db, project=response.json())
            raise APIError(response.json().get("detail", "Project not found"))
        else:
            response = self._db.get("/projects/")
            if response.ok:
                return [PetroDBProject(self._db, project=p) for p in response.json()]
            raise APIError(response.json().get("detail", "Projects not found"))

    def create_project(self, name: str, description: str = ""):
        """Create project in database

        Args:
            name (str): name of the project
            description (str, optional): decription of the project. Default ``.

        Returns:
            Created project instance.

        Raises:
            APIError: If project was not created.

        """
        data = {"name": name, "description": description}
        response = self._db.post("/project/", data)
        if response.ok:
            return PetroDBProject(self._db, project=response.json())
        raise APIError(response.json().get("detail", "Project not created"))


# ── entity classes ──────────────────────────────────────────────────────


class PetroDBProject:
    """Petro DB project instance

    Attributes:
        data (dict): project attributes

    """

    def __init__(self, db: PetroAPI, project: dict):
        self._db = db
        self.data = dict(project)
        self._project_id: int = self.data.pop("id")

    def __repr__(self) -> str:
        return f"{self.name}"

    @property
    def name(self) -> str:
        """str: Name of the project."""
        return self.data["name"]

    @name.setter
    def name(self, name: str) -> None:
        self.data["name"] = name

    @property
    def description(self) -> str:
        """str: Description of the project."""
        return self.data["description"]

    @description.setter
    def description(self, desc: str) -> None:
        self.data["description"] = desc

    def samples(self, name: str | None = None):
        """Get sample from database

        Args:
            name (str, optional): search for sample by name

        Returns:
            Sample instance or list of all samples if the name is not provided.

        Raises:
            APIError: If sample(s) was not found.

        """
        if name is not None:
            response = self._db.get(f"/search/sample/{self._project_id}/{name}")
            if response.ok:
                return PetroDBSample(self._db, self._project_id, sample=response.json())
            raise APIError(response.json().get("detail", "Sample not found"))
        else:
            response = self._db.get(f"/samples/{self._project_id}")
            if response.ok:
                return [
                    PetroDBSample(self._db, self._project_id, sample=s)
                    for s in response.json()
                ]
            raise APIError(response.json().get("detail", "Samples not found"))

    def create_sample(self, name: str, description: str = ""):
        """Create project in database

        Args:
            name (str): name of the sample
            description (str, optional): decription of the sample. Default ``.

        Returns:
            Created sample instance.

        Raises:
            APIError: If sample was not created.

        """
        data = {"name": name, "description": description}
        response = self._db.post(f"/sample/{self._project_id}", data)
        if response.ok:
            return PetroDBSample(self._db, self._project_id, sample=response.json())
        raise APIError(response.json().get("detail", "Sample not created"))

    def delete(self) -> dict:
        """Delete project from database. Use with caution!"""
        response = self._db.delete(f"/project/{self._project_id}")
        if response.ok:
            return response.json()
        raise APIError(response.json().get("detail", "Project not deleted"))

    def update(self) -> dict:
        """Update project in database according to the data attribute."""
        response = self._db.put(f"/project/{self._project_id}", self.data)
        if response.ok:
            return response.json()
        raise APIError(response.json().get("detail", "Project not updated"))

    @property
    def spots(self):
        """Spots accessor"""
        records = {}
        names = []
        for sample in self.samples():
            try:
                s = sample.spots
                records = records | s.records
                names += s.sample
            except APIError:
                pass
        return PetroDBSpotRecords(records, names)

    @property
    def areas(self):
        """Areas accessor"""
        records = {}
        names = []
        for sample in self.samples():
            try:
                s = sample.areas
                records = records | s.records
                names += s.sample
            except APIError:
                pass
        return PetroDBAreaRecords(records, names)

    def mineral_data(self, mineral: str) -> pd.DataFrame:
        """Return spots and profile spots of given mineral dataframe"""
        res = []
        samples = self.samples()
        for sample in samples:
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
        """Add access to the project to user

        Args:
            username (str): username

        """
        response = self._db.put(
            f"/project/{self._project_id}/adduser", {"username": username}
        )
        if response.ok:
            return response.json()
        raise APIError(response.json().get("detail", "User not added"))

    def remove_user(self, username: str) -> dict:
        """Remove access to the project for user

        Args:
            username (str): username

        """
        response = self._db.put(
            f"/project/{self._project_id}/removeuser",
            {"username": username},
        )
        if response.ok:
            return response.json()
        raise APIError(response.json().get("detail", "User not removed"))


class PetroDBSample:
    """Petro DB sample instance

    Attributes:
        data (dict): project attributes

    """

    def __init__(self, db: PetroAPI, project_id: int, sample: dict):
        self._db = db
        self._project_id = project_id
        self.data = dict(sample)
        self._sample_id: int = self.data.pop("id")

    def __repr__(self) -> str:
        return f"{self.name}"

    @property
    def name(self) -> str:
        """str: Name of the sample."""
        return self.data["name"]

    @name.setter
    def name(self, name: str) -> None:
        self.data["name"] = name

    @property
    def description(self) -> str:
        """str: Description of the sample."""
        return self.data["description"]

    @description.setter
    def description(self, desc: str) -> None:
        self.data["description"] = desc

    @cached_property
    def spots(self):
        """Spots accessor"""
        response = self._db.get(f"/spots/{self._project_id}/{self._sample_id}")
        if response.ok:
            spots = {
                res["id"]: PetroDBSpot(
                    self._db, self._project_id, self._sample_id, spot=res
                )
                for res in response.json()
            }
            return PetroDBSpotRecords(spots, len(spots) * [self.name])
        raise APIError(response.json().get("detail", "Spots not found"))

    def spot(self, spot_id: int):
        """Get spot from database

        Args:
            spot_id (int): id of the spot

        Returns:
            Spot instance.

        Raises:
            APIError: If spot was not found.

        """
        response = self._db.get(f"/spot/{self._project_id}/{self._sample_id}/{spot_id}")
        if response.ok:
            return PetroDBSpot(
                self._db,
                self._project_id,
                self._sample_id,
                spot=response.json(),
            )
        raise APIError(response.json().get("detail", "Spot not found"))

    def create_spot(
        self,
        label: str,
        mineral: str,
        values: dict,
    ):
        """Create spot in database

        Args:
            label (str): label of the spot
            mineral (str): Name of mineral. Kretz abbreviations recommended
            values (dict): Data values

        Returns:
            Created spot instance.

        Raises:
            APIError: If spot was not created.

        """
        data = {"label": label, "mineral": mineral, "values": values}
        response = self._db.post(f"/spot/{self._project_id}/{self._sample_id}", data)
        if response.ok:
            return PetroDBSpot(
                self._db,
                self._project_id,
                self._sample_id,
                spot=response.json(),
            )
        raise APIError(response.json().get("detail", "Spot not created"))

    def create_spots(
        self,
        df: pd.DataFrame,
        label_col: str | None = None,
        mineral_col: str | None = None,
    ):
        """Create spots in database from pandas dataframe

        Args:
            df (pandas.DataFrame): data values
            label_col (str, optional): Name of column to be used as label. If not
                provided, dataframe index is used
            mineral_col (str, optional): Name of column to be used as mineral. If not
                provided, mineral will be empty

        Returns:
            Created spots

        Raises:
            APIError: If spots were not created.

        """
        df = df.copy()
        if label_col is None:
            labels = pd.Series(df.index.astype(str), index=df.index)
        else:
            labels = df[label_col].str.strip()
            df.drop(label_col, axis=1, inplace=True)
        if mineral_col is None:
            minerals = pd.Series("", index=df.index)
        else:
            minerals = df[mineral_col].str.strip()
            df.drop(mineral_col, axis=1, inplace=True)
        spots = []
        for label, mineral, (ix, row) in zip(labels, minerals, df.iterrows()):
            spots.append(
                {
                    "label": label,
                    "mineral": mineral,
                    "values": row.apply(zero_negative_nan).dropna().to_dict(),
                }
            )
        response = self._db.post(f"/spots/{self._project_id}/{self._sample_id}", spots)
        if response.ok:
            spots = {
                res["id"]: PetroDBSpot(
                    self._db, self._project_id, self._sample_id, spot=res
                )
                for res in response.json()
            }
            return PetroDBSpotRecords(spots, len(spots) * [self.name])
        raise APIError(response.json().get("detail", "Spots not created"))

    @cached_property
    def areas(self):
        """Areas accessor"""
        response = self._db.get(f"/areas/{self._project_id}/{self._sample_id}")
        if response.ok:
            areas = {
                res["id"]: PetroDBArea(
                    self._db, self._project_id, self._sample_id, area=res
                )
                for res in response.json()
            }
            return PetroDBAreaRecords(areas, len(areas) * [self.name])
        raise APIError(response.json().get("detail", "Areas not found"))

    def area(self, area_id: int):
        """Get area from database

        Args:
            area_id (int): id of the spot

        Returns:
            Area instance.

        Raises:
            APIError: If area was not found.

        """
        response = self._db.get(f"/area/{self._project_id}/{self._sample_id}/{area_id}")
        if response.ok:
            return PetroDBArea(
                self._db,
                self._project_id,
                self._sample_id,
                area=response.json(),
            )
        raise APIError(response.json().get("detail", "Area not found"))

    def create_area(self, label: str, values: dict):
        """Create area in database

        Args:
            label (str): label of the area
            values (dict): Data values

        Returns:
            Created area instance.

        Raises:
            APIError: If area was not created.

        """
        data = {"label": label, "values": values}
        response = self._db.post(f"/area/{self._project_id}/{self._sample_id}", data)
        if response.ok:
            return PetroDBArea(
                self._db,
                self._project_id,
                self._sample_id,
                area=response.json(),
            )
        raise APIError(response.json().get("detail", "Area not created"))

    def create_areas(
        self,
        df: pd.DataFrame,
        label_col: str | None = None,
    ):
        """Create areas in database from pandas dataframe

        Args:
            df (pandas.DataFrame): data values
            label_col (str, optional): Name of column to be used as label. If not
                provided, dataframe index is used

        Returns:
            Created areas

        Raises:
            APIError: If areas were not created.

        """
        df = df.copy()
        if label_col is None:
            labels = pd.Series(df.index.astype(str), index=df.index)
        else:
            labels = df[label_col].str.strip()
            df.drop(label_col, axis=1, inplace=True)
        areas = []
        for label, (ix, row) in zip(labels, df.iterrows()):
            areas.append(
                {
                    "label": label,
                    "values": row.apply(zero_negative_nan).dropna().to_dict(),
                }
            )
        response = self._db.post(f"/areas/{self._project_id}/{self._sample_id}", areas)
        if response.ok:
            return response.json()
        raise APIError(response.json().get("detail", "Areas not created"))

    def profiles(
        self,
        label: str | None = None,
        mineral: str | None = None,
    ):
        """Get profile from database

        Args:
            label (str, optional): search for sample by name

        Returns:
            Profile instance or list of all profiles if the label is not provided.

        Raises:
            APIError: If profile(s) was not found.

        """
        if label is not None:
            response = self._db.get(
                f"/search/profile/{self._project_id}/{self._sample_id}/{label}"
            )
            if response.ok:
                data = response.json()
                if mineral is not None:
                    if data["mineral"] == mineral:
                        return PetroDBProfile(
                            self._db,
                            self._project_id,
                            self._sample_id,
                            self.name,
                            profile=response.json(),
                        )
                    raise NotFoundError(
                        f"Profile with {label=} and {mineral=} not found."
                    )
                return PetroDBProfile(
                    self._db,
                    self._project_id,
                    self._sample_id,
                    self.name,
                    profile=response.json(),
                )
            raise APIError(response.json().get("detail", "Profile not found"))
        else:
            response = self._db.get(f"/profiles/{self._project_id}/{self._sample_id}")
            if response.ok:
                if mineral is not None:
                    return [
                        PetroDBProfile(
                            self._db,
                            self._project_id,
                            self._sample_id,
                            self.name,
                            profile=p,
                        )
                        for p in response.json()
                        if p["mineral"] == mineral
                    ]
                else:
                    return [
                        PetroDBProfile(
                            self._db,
                            self._project_id,
                            self._sample_id,
                            self.name,
                            profile=p,
                        )
                        for p in response.json()
                    ]
            raise APIError(response.json().get("detail", "Profiles not found"))

    def create_profile(self, label: str, mineral: str):
        """Create profile in database

        Args:
            label (str): label of the profile
            mineral (str): Name of mineral. Kretz abbreviations recommended

        Returns:
            Created profile instance.

        Raises:
            APIError: If profile was not created.

        """
        data = {"label": label, "mineral": mineral}
        response = self._db.post(f"/profile/{self._project_id}/{self._sample_id}", data)
        if response.ok:
            return PetroDBProfile(
                self._db,
                self._project_id,
                self._sample_id,
                self.name,
                profile=response.json(),
            )
        raise APIError(response.json().get("detail", "Profile not created"))

    def delete(self) -> dict:
        """Delete sample from database. Use with caution!"""
        response = self._db.delete(f"/sample/{self._project_id}/{self._sample_id}")
        if response.ok:
            return response.json()
        raise APIError(response.json().get("detail", "Sample not deleted"))

    def update(self) -> dict:
        """Update sample in database according to the data attribute."""
        response = self._db.put(
            f"/sample/{self._project_id}/{self._sample_id}", self.data
        )
        if response.ok:
            return response.json()
        raise APIError(response.json().get("detail", "Sample not updated"))

    def reset(self) -> None:
        """Reset cached properties spots and areas to access updated data"""
        self.__dict__.pop("spots", None)
        self.__dict__.pop("areas", None)

    @property
    def profilespots(self):
        records = {}
        names = []
        for profile in self.profiles():
            try:
                s = profile.spots
                records = records | s.records
                names += s.sample
            except APIError:
                pass
        return PetroDBProfileSpotRecords(records, names)

    def mineral_data(self, mineral: str) -> pd.DataFrame:
        """Return spots and profile spots of given mineral dataframe"""
        res = []
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


class PetroDBSpot:
    """Petro DB spot instance

    Attributes:
        data (dict): project attributes

    """

    def __init__(self, db: PetroAPI, project_id: int, sample_id: int, spot: dict):
        self._db = db
        self._project_id = project_id
        self._sample_id = sample_id
        self.data = dict(spot)
        self._spot_id: int = self.data.pop("id")

    def __repr__(self) -> str:
        return f"{self.label} ({self.mineral})"

    @property
    def label(self) -> str:
        """str: Label of the spot."""
        return self.data["label"]

    @label.setter
    def label(self, lbl: str) -> None:
        self.data["label"] = lbl

    @property
    def mineral(self) -> str:
        """str: Mineral of the spot."""
        return self.data["mineral"]

    @mineral.setter
    def mineral(self, m: str) -> None:
        self.data["mineral"] = m

    def delete(self) -> dict:
        """Delete spot from database. Use with caution!"""
        response = self._db.delete(
            f"/spot/{self._project_id}/{self._sample_id}/{self._spot_id}"
        )
        if response.ok:
            return response.json()
        raise APIError(response.json().get("detail", "Spot not deleted"))

    def update(self) -> dict:
        """Update spot in database according to the data attribute."""
        response = self._db.put(
            f"/spot/{self._project_id}/{self._sample_id}/{self._spot_id}",
            self.data,
        )
        if response.ok:
            return response.json()
        raise APIError(response.json().get("detail", "Spot not updated"))


class PetroDBArea:
    """Petro DB area instance

    Attributes:
        data (dict): project attributes

    """

    def __init__(self, db: PetroAPI, project_id: int, sample_id: int, area: dict):
        self._db = db
        self._project_id = project_id
        self._sample_id = sample_id
        self.data = dict(area)
        self._area_id: int = self.data.pop("id")

    def __repr__(self) -> str:
        return f"{self.label}"

    @property
    def label(self) -> str:
        """str: Label of the area."""
        return self.data["label"]

    @label.setter
    def label(self, lbl: str) -> None:
        self.data["label"] = lbl

    def delete(self) -> dict:
        """Delete area from database. Use with caution!"""
        response = self._db.delete(
            f"/area/{self._project_id}/{self._sample_id}/{self._area_id}"
        )
        if response.ok:
            return response.json()
        raise APIError(response.json().get("detail", "Area not deleted"))

    def update(self) -> dict:
        """Update area in database according to the data attribute."""
        response = self._db.put(
            f"/area/{self._project_id}/{self._sample_id}/{self._area_id}",
            self.data,
        )
        if response.ok:
            return response.json()
        raise APIError(response.json().get("detail", "Area not updated"))


class PetroDBProfile:
    """Petro DB sample instance

    Attributes:
        data (dict): project attributes
        samplename (str): name of the sample

    """

    def __init__(
        self,
        db: PetroAPI,
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
        self._profile_id: int = self.data.pop("id")

    def __repr__(self) -> str:
        return f"{self.label} ({self.mineral})"

    @property
    def label(self) -> str:
        """str: Label of the profile."""
        return self.data["label"]

    @label.setter
    def label(self, lbl: str) -> None:
        self.data["label"] = lbl

    @property
    def mineral(self) -> str:
        """str: Mineral of the profile."""
        return self.data["mineral"]

    @mineral.setter
    def mineral(self, m: str) -> None:
        self.data["mineral"] = m

    @cached_property
    def spots(self):
        """Profile spots accessor"""
        response = self._db.get(
            f"/profilespots/{self._project_id}/{self._sample_id}/{self._profile_id}"
        )
        if response.ok:
            spots = {
                res["id"]: PetroDBProfileSpot(
                    self._db,
                    self._project_id,
                    self._sample_id,
                    self._profile_id,
                    spot=res,
                )
                for res in response.json()
            }
            for k in spots:
                spots[k].data["label"] = self.label
                spots[k].data["mineral"] = self.mineral
            return PetroDBProfileSpotRecords(spots, len(spots) * [self.samplename])
        raise APIError(response.json().get("detail", "Profile spots not found"))

    def spot(self, spot_id: int):
        """Get profile spot from database

        Args:
            spot_id (int): id of the profile spot

        Returns:
            Profile spot instance.

        Raises:
            APIError: If profile spot was not found.

        """
        response = self._db.get(
            f"/profilespot/{self._project_id}/{self._sample_id}"
            f"/{self._profile_id}/{spot_id}"
        )
        if response.ok:
            return PetroDBProfileSpot(
                self._db,
                self._project_id,
                self._sample_id,
                self._profile_id,
                spot=response.json(),
            )
        raise APIError(response.json().get("detail", "Profile spot not found"))

    def create_spot(self, index: int, values: dict):
        """Create profile spot in database

        Args:
            index (int): used to define order of spots on profile
            values (dict): Data values

        Returns:
            Created profile spot instance.

        Raises:
            APIError: If profile spot was not created.

        """
        data = {"index": index, "values": values}
        response = self._db.post(
            f"/profilespot/{self._project_id}/{self._sample_id}/{self._profile_id}",
            data,
        )
        if response.ok:
            return PetroDBProfileSpot(
                self._db,
                self._project_id,
                self._sample_id,
                self._profile_id,
                spot=response.json(),
            )
        raise APIError(response.json().get("detail", "Profile spot not created"))

    def create_spots(self, df: pd.DataFrame):
        """Create profile spots in database from pandas dataframe

        Args:
            df (pandas.DataFrame): data values, index must be numeric and is used
                to define order

        Returns:
            Created profile spots

        Raises:
            APIError: If profile spots were not created.

        """
        df = df.copy()
        profilespots = []
        for index, row in df.iterrows():
            profilespots.append(
                {
                    "index": index,
                    "values": row.apply(zero_negative_nan).dropna().to_dict(),
                }
            )
        response = self._db.post(
            f"/profilespots/{self._project_id}/{self._sample_id}/{self._profile_id}",
            profilespots,
        )
        if response.ok:
            spots = {
                res["id"]: PetroDBProfileSpot(
                    self._db,
                    self._project_id,
                    self._sample_id,
                    self._profile_id,
                    spot=res,
                )
                for res in response.json()
            }
            for k in spots:
                spots[k].data["label"] = self.label
                spots[k].data["mineral"] = self.mineral
            return PetroDBProfileSpotRecords(spots, len(spots) * [self.samplename])
        raise APIError(response.json().get("detail", "Profile spots not created"))

    def delete(self) -> dict:
        """Delete profile from database. Use with caution!"""
        response = self._db.delete(
            f"/profile/{self._project_id}/{self._sample_id}/{self._profile_id}"
        )
        if response.ok:
            return response.json()
        raise APIError(response.json().get("detail", "Profile not deleted"))

    def update(self) -> dict:
        """Update profile in database according to the data attribute."""
        response = self._db.put(
            f"/profile/{self._project_id}/{self._sample_id}/{self._profile_id}",
            self.data,
        )
        if response.ok:
            return response.json()
        raise APIError(response.json().get("detail", "Profile not updated"))

    def reset(self) -> None:
        """Reset cached property spots to access updated data"""
        self.__dict__.pop("spots", None)


class PetroDBProfileSpot:
    """Petro DB profile spot instance

    Attributes:
        data (dict): project attributes

    """

    def __init__(
        self,
        db: PetroAPI,
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
        self._profilespot_id: int = self.data.pop("id")

    def __repr__(self) -> str:
        return f"Spot {self.index}"

    @property
    def index(self) -> int:
        """int: Index of the profile spot."""
        return self.data["index"]

    @index.setter
    def index(self, idx: int) -> None:
        self.data["index"] = idx

    def delete(self) -> dict:
        """Delete profile spot from database. Use with caution!"""
        response = self._db.delete(
            f"/profilespot/{self._project_id}/{self._sample_id}"
            f"/{self._profile_id}/{self._profilespot_id}"
        )
        if response.ok:
            return response.json()
        raise APIError(response.json().get("detail", "Profile spot not deleted"))

    def update(self) -> dict:
        """Update profile spot in database according to the data attribute."""
        response = self._db.put(
            f"/profilespot/{self._project_id}/{self._sample_id}"
            f"/{self._profile_id}/{self._profilespot_id}",
            self.data,
        )
        if response.ok:
            return response.json()
        raise APIError(response.json().get("detail", "Profile spot not updated"))


# ── records collections ─────────────────────────────────────────────────


class PetroDBRecords:
    """Petro DB records accessor"""

    def __init__(self, records: dict, sample: list):
        self.records = records
        self.sample = sample
        self.cols: list[str] = []

    def __getitem__(self, id: int):
        if isinstance(id, int):
            return self.records[id]

    def df(self, **kwargs) -> pd.DataFrame:
        """Get records as pandas dataframe

        Note: Keyword arguments `col=val` are used to select records with given
            value. `col` must be on of the available columns in attribute cols.

        Attributes:
            cols (list): list of attributes for selection

        """
        res = pd.DataFrame(
            {k: v.data["values"] for k, v in self.records.items()}
        ).T.infer_objects()
        res["sample"] = self.sample
        for col in self.cols:
            res[col] = [row.data[col] for row in self.records.values()]
        for col, val in kwargs.items():
            if col in self.cols:
                res = res[res[col] == val]
        return res.sort_index().copy()


class PetroDBSpotRecords(PetroDBRecords):
    def __init__(self, records: dict, sample: list):
        super().__init__(records, sample)
        self.cols = ["label", "mineral"]

    def __repr__(self) -> str:
        return f"{len(self.records)} spots"


class PetroDBAreaRecords(PetroDBRecords):
    def __init__(self, records: dict, sample: list):
        super().__init__(records, sample)
        self.cols = ["label"]

    def __repr__(self) -> str:
        return f"{len(self.records)} areas"


class PetroDBProfileSpotRecords(PetroDBRecords):
    def __init__(self, records: dict, sample: list):
        super().__init__(records, sample)
        self.cols = ["label", "mineral"]

    def __repr__(self) -> str:
        return f"{len(self.records)} profile spots"


# backward compatibility alias
PetroDBProfilespotRecords = PetroDBProfileSpotRecords


# ── admin ────────────────────────────────────────────────────────────────


class PetroDBAdmin:
    """Admin client for postresql petrodb database API."""

    def __init__(self, api_url: str, username: str, password: str, timeout: int = 30):
        self._db = PetroAPI(api_url, username, password, timeout)

    def __enter__(self):
        return self

    def __exit__(self, *args: object) -> None:
        self._db._session.close()

    # ---------- USERS

    def users(self, name: str | None = None):
        response = self._db.get("/users/")
        if response.ok:
            return response.json()
        raise APIError(response.json().get("detail", "Users not found"))

    def create_user(self, username: str, password: str, email: str) -> dict:
        """Create user in database

        Args:
            username (str): username
            password (str): password
            email (str): email

        """
        data = {"username": username, "password": password, "email": email}
        response = self._db.post("/user/", data)
        if response.ok:
            return response.json()
        raise APIError(response.json().get("detail", "User not created"))

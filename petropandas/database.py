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

from functools import cached_property

import pandas as pd
import requests


def zero_negative_nan(x):
    if isinstance(x, (int, float, complex)) and not isinstance(x, bool):
        if x > 0:
            return x
        else:
            return float("nan")
    else:
        return x


class PetroAPI:
    def __init__(self, api_url, username, password):
        response = requests.post(
            f"{api_url}/token",
            data={"username": username, "password": password},
        )
        if response.ok:
            self.__api_url = api_url
            self.__username = username
            self.__password = password
            self.logged = True
        else:
            self.logged = False

    def __authorize(self):
        if self.logged:
            response = requests.post(
                f"{self.__api_url}/token",
                data={"username": self.__username, "password": self.__password},
            )
            if response.ok:
                token = response.json()
                return {"Authorization": f"Bearer {token.get('access_token')}"}
            else:
                raise ValueError("Wrong url or credentials")
        else:
            raise ConnectionError("Not logged in")

    def get(self, path):
        headers = self.__authorize()
        return requests.get(f"{self.__api_url}/api{path}", headers=headers)

    def post(self, path, data):
        headers = self.__authorize()
        return requests.post(f"{self.__api_url}/api{path}", json=data, headers=headers)

    def put(self, path, data):
        headers = self.__authorize()
        return requests.put(f"{self.__api_url}/api{path}", json=data, headers=headers)

    def delete(self, path):
        headers = self.__authorize()
        return requests.delete(f"{self.__api_url}/api{path}", headers=headers)


class PetroDB:
    """Petro database instance

    High-level access to online Petro database

    """

    def __init__(self, api_url, username, password):
        self.__db = PetroAPI(api_url, username, password)

    def __repr__(self):
        return f"PetroDB {'OK' if self.__db.logged else 'Not logged'}"

    @property
    def logged(self) -> bool:
        """Return True when API credentials are ok."""
        return self.__db.logged

    def projects(self, name: str | None = None):
        """Get project from database

        Args:
            name (str, optional): search for project by name

        Returns:
            Project instance or list of all projects if the name is not provided.

        Raises:
            ValueError: If project(s) was not found.

        """
        if name is not None:
            response = self.__db.get(f"/search/project/{name}")
            if response.ok:
                return PetroDBProject(self.__db, project=response.json())
            else:
                raise ValueError(response.json()["detail"])
        else:
            response = self.__db.get("/projects/")
            if response.ok:
                return [PetroDBProject(self.__db, project=p) for p in response.json()]
            else:
                raise ValueError(response.json()["detail"])

    def create_project(self, name: str, description: str = ""):
        """Create project in database

        Args:
            name (str): name of the project
            description (str, optional): decription of the project. Default ``.

        Returns:
            Created project instance.

        Raises:
            ValueError: If project was not created.

        """
        data = {"name": name, "description": description}
        response = self.__db.post("/project/", data)
        if response.ok:
            return PetroDBProject(self.__db, project=response.json())
        else:
            raise ValueError(response.json()["detail"])


class PetroDBProject:
    """Petro DB project instance

    Attributes:
        data (dict): project attributes

    """

    def __init__(self, db, project):
        self.__db = db
        self.__project_id = project.pop("id")
        self.data = project

    def __repr__(self):
        return f"{self.name}"

    @property
    def name(self):
        """str: Name of the project."""
        return self.data["name"]

    @name.setter
    def name(self, name: str):
        self.data["name"] = name

    @property
    def description(self):
        """str: Description of the project."""
        return self.data["description"]

    @description.setter
    def description(self, desc: str):
        self.data["description"] = desc

    def samples(self, name: str | None = None):
        """Get sample from database

        Args:
            name (str, optional): search for sample by name

        Returns:
            Sample instance or list of all samples if the name is not provided.

        Raises:
            ValueError: If sample(s) was not found.

        """
        if name is not None:
            response = self.__db.get(f"/search/sample/{self.__project_id}/{name}")
            if response.ok:
                return PetroDBSample(
                    self.__db, self.__project_id, sample=response.json()
                )
            else:
                raise ValueError(response.json()["detail"])
        else:
            response = self.__db.get(f"/samples/{self.__project_id}")
            if response.ok:
                return [
                    PetroDBSample(self.__db, self.__project_id, sample=s)
                    for s in response.json()
                ]
            else:
                raise ValueError(response.json()["detail"])

    def create_sample(self, name: str, description: str = ""):
        """Create project in database

        Args:
            name (str): name of the sample
            description (str, optional): decription of the sample. Default ``.

        Returns:
            Created sample instance.

        Raises:
            ValueError: If sample was not created.

        """
        data = {"name": name, "description": description}
        response = self.__db.post(f"/sample/{self.__project_id}", data)
        if response.ok:
            return PetroDBSample(self.__db, self.__project_id, sample=response.json())
        else:
            raise ValueError(response.json()["detail"])

    def delete(self):
        """Delete project from database. Use with caution!"""
        response = self.__db.delete(f"/project/{self.__project_id}")
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    def update(self):
        """Update project in database according to the data attribute."""
        response = self.__db.put(f"/project/{self.__project_id}", self.data)
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    @property
    def spots(self):
        """Spots accessor"""
        res = []
        names = []
        for sample in self.samples():
            try:
                res += list(sample.spots.records.values())
                names += sample.spots.sample
            except ValueError:
                pass
        return PetroDBSpotRecords(res, sample=names)

    @property
    def areas(self):
        """Areas accessor"""
        res = []
        names = []
        for sample in self.samples():
            try:
                res += list(sample.areas.records.values())
                names += sample.areas.sample
            except ValueError:
                pass
        return PetroDBAreaRecords(res, sample=names)

    def mineral_data(self, mineral: str):
        """Return spots and profile spots of given mineral dataframe"""
        res = []
        samples = self.samples()
        for sample in samples:
            spots = sample.spots.df(mineral=mineral, sample_name=True)
            if not spots.empty:
                res.append(spots)
            profiles = sample.profiles(mineral=mineral)
            for profile in profiles:
                res.append(profile.spots.df(sample_name=True))
        return pd.concat(res)

    def add_user(self, username: str):
        """Add access to the project to user

        Args:
            username (str): username

        """
        response = self.__db.put(
            f"/project/{self.__project_id}/adduser", {"username": username}
        )
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    def remove_user(self, username: str):
        """Remove access to the project for user

        Args:
            username (str): username

        """
        response = self.__db.put(
            f"/project/{self.__project_id}/removeuser", {"username": username}
        )
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])


class PetroDBSample:
    """Petro DB sample instance

    Attributes:
        data (dict): project attributes

    """

    def __init__(self, db, project_id, sample):
        self.__db = db
        self.__project_id = project_id
        self.__sample_id = sample.pop("id")
        self.data = sample

    def __repr__(self):
        return f"{self.name}"

    @property
    def name(self):
        """str: Name of the sample."""
        return self.data["name"]

    @name.setter
    def name(self, name: str):
        self.data["name"] = name

    @property
    def description(self):
        """str: Description of the sample."""
        return self.data["description"]

    @description.setter
    def description(self, desc: str):
        self.data["description"] = desc

    @cached_property
    def spots(self):
        """Spots accessor"""
        response = self.__db.get(f"/spots/{self.__project_id}/{self.__sample_id}")
        if response.ok:
            return PetroDBSpotRecords(
                response.json(), sample=len(response.json()) * [self.name]
            )
        else:
            raise ValueError(response.json()["detail"])

    def spot(self, spot_id: int):
        """Get spot from database

        Args:
            spot_id (int): id of the spot

        Returns:
            Spot instance.

        Raises:
            ValueError: If spot was not found.

        """
        response = self.__db.get(
            f"/spot/{self.__project_id}/{self.__sample_id}/{spot_id}"
        )
        if response.ok:
            return PetroDBSpot(
                self.__db, self.__project_id, self.__sample_id, spot=response.json()
            )
        else:
            raise ValueError(response.json()["detail"])

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
            ValueError: If spot was not created.

        """
        data = {"label": label, "mineral": mineral, "values": values}
        response = self.__db.post(f"/spot/{self.__project_id}/{self.__sample_id}", data)
        if response.ok:
            return PetroDBSpot(
                self.__db, self.__project_id, self.__sample_id, spot=response.json()
            )
        else:
            raise ValueError(response.json()["detail"])

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
            ValueError: If spots were not created.

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
        response = self.__db.post(
            f"/spots/{self.__project_id}/{self.__sample_id}", spots
        )
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    @cached_property
    def areas(self):
        """Areas accessor"""
        response = self.__db.get(f"/areas/{self.__project_id}/{self.__sample_id}")
        if response.ok:
            return PetroDBAreaRecords(
                response.json(), sample=len(response.json()) * [self.name]
            )
        else:
            raise ValueError(response.json()["detail"])

    def area(self, area_id: int):
        """Get area from database

        Args:
            area_id (int): id of the spot

        Returns:
            Area instance.

        Raises:
            ValueError: If area was not found.

        """
        response = self.__db.get(
            f"/area/{self.__project_id}/{self.__sample_id}/{area_id}"
        )
        if response.ok:
            return PetroDBArea(
                self.__db, self.__project_id, self.__sample_id, area=response.json()
            )
        else:
            raise ValueError(response.json()["detail"])

    def create_area(self, label: str, values: dict):
        """Create area in database

        Args:
            label (str): label of the area
            values (dict): Data values

        Returns:
            Created area instance.

        Raises:
            ValueError: If area was not created.

        """
        data = {"label": label, "values": values}
        response = self.__db.post(f"/area/{self.__project_id}/{self.__sample_id}", data)
        if response.ok:
            return PetroDBArea(
                self.__db, self.__project_id, self.__sample_id, area=response.json()
            )
        else:
            raise ValueError(response.json()["detail"])

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
            ValueError: If areas were not created.

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
        response = self.__db.post(
            f"/areas/{self.__project_id}/{self.__sample_id}", areas
        )
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    def profiles(self, label: str | None = None, mineral: str | None = None):
        """Get profile from database

        Args:
            label (str, optional): search for sample by name

        Returns:
            Profile instance or list of all profiles if the label is not provided.

        Raises:
            ValueError: If profile(s) was not found.

        """
        if label is not None:
            response = self.__db.get(
                f"/search/profile/{self.__project_id}/{self.__sample_id}/{label}"
            )
            if response.ok:
                data = response.json()
                if mineral is not None:
                    if data["mineral"] == mineral:
                        return PetroDBProfile(
                            self.__db,
                            self.__project_id,
                            self.__sample_id,
                            self.name,
                            profile=response.json(),
                        )
                    else:
                        raise ValueError(
                            f"Profile with {label=} and {mineral=} not found."
                        )
                return PetroDBProfile(
                    self.__db,
                    self.__project_id,
                    self.__sample_id,
                    self.name,
                    profile=response.json(),
                )
            else:
                raise ValueError(response.json()["detail"])
        else:
            response = self.__db.get(
                f"/profiles/{self.__project_id}/{self.__sample_id}"
            )
            if response.ok:
                if mineral is not None:
                    return [
                        PetroDBProfile(
                            self.__db,
                            self.__project_id,
                            self.__sample_id,
                            self.name,
                            profile=p,
                        )
                        for p in response.json()
                        if p["mineral"] == mineral
                    ]
                else:
                    return [
                        PetroDBProfile(
                            self.__db,
                            self.__project_id,
                            self.__sample_id,
                            self.name,
                            profile=p,
                        )
                        for p in response.json()
                    ]
            else:
                raise ValueError(response.json()["detail"])

    def create_profile(self, label: str, mineral: str):
        """Create profile in database

        Args:
            label (str): label of the profile
            mineral (str): Name of mineral. Kretz abbreviations recommended

        Returns:
            Created profile instance.

        Raises:
            ValueError: If profile was not created.

        """
        data = {"label": label, "mineral": mineral}
        response = self.__db.post(
            f"/profile/{self.__project_id}/{self.__sample_id}", data
        )
        if response.ok:
            return PetroDBProfile(
                self.__db,
                self.__project_id,
                self.__sample_id,
                self.name,
                profile=response.json(),
            )
        else:
            raise ValueError(response.json()["detail"])

    def delete(self):
        """Delete sample from database. Use with caution!"""
        response = self.__db.delete(f"/sample/{self.__project_id}/{self.__sample_id}")
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    def update(self):
        """Update sample in database according to the data attribute."""
        response = self.__db.put(
            f"/sample/{self.__project_id}/{self.__sample_id}", self.data
        )
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    def reset(self):
        """Reset cached properties spots and areas to access updated data"""
        if "spots" in self.__dict__:
            self.__dict__["spots"]
        if "areas" in self.__dict__:
            self.__dict__["areas"]

    @property
    def profilespots(self):
        res = []
        names = []
        for profile in self.profiles():
            try:
                res += list(profile.spots.records.values())
                names += profile.spots.sample
            except ValueError:
                pass
        return PetroDBProfilespotRecords(res, sample=names)


class PetroDBSpot:
    """Petro DB spot instance

    Attributes:
        data (dict): project attributes

    """

    def __init__(self, db, project_id, sample_id, spot):
        self.__db = db
        self.__project_id = project_id
        self.__sample_id = sample_id
        self.__spot_id = spot.pop("id")
        self.data = spot

    def __repr__(self):
        return f"{self.label} ({self.mineral})"

    @property
    def label(self):
        """str: Label of the spot."""
        return self.data["label"]

    @label.setter
    def label(self, lbl: str):
        self.data["label"] = lbl

    @property
    def mineral(self):
        """str: Mineral of the spot."""
        return self.data["mineral"]

    @mineral.setter
    def mineral(self, m: str):
        self.data["mineral"] = m

    def delete(self):
        """Delete spot from database. Use with caution!"""
        response = self.__db.delete(
            f"/spot/{self.__project_id}/{self.__sample_id}/{self.__spot_id}"
        )
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    def update(self):
        """Update spot in database according to the data attribute."""
        response = self.__db.put(
            f"/spot/{self.__project_id}/{self.__sample_id}/{self.__spot_id}",
            self.data,
        )
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])


class PetroDBArea:
    """Petro DB area instance

    Attributes:
        data (dict): project attributes

    """

    def __init__(self, db, project_id, sample_id, area):
        self.__db = db
        self.__project_id = project_id
        self.__sample_id = sample_id
        self.__area_id = area.pop("id")
        self.data = area

    def __repr__(self):
        return f"{self.label}"

    @property
    def label(self):
        """str: Label of the area."""
        return self.data["label"]

    @label.setter
    def label(self, lbl: str):
        self.data["label"] = lbl

    def delete(self):
        """Delete area from database. Use with caution!"""
        response = self.__db.delete(
            f"/area/{self.__project_id}/{self.__sample_id}/{self.__area_id}"
        )
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    def update(self):
        """Update area in database according to the data attribute."""
        response = self.__db.put(
            f"/area/{self.__project_id}/{self.__sample_id}/{self.__area_id}",
            self.data,
        )
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])


class PetroDBProfile:
    """Petro DB sample instance

    Attributes:
        data (dict): project attributes
        samplename (str): name of the sample

    """

    def __init__(self, db, project_id, sample_id, samplename, profile):
        self.__db = db
        self.__project_id = project_id
        self.__sample_id = sample_id
        self.samplename = samplename
        self.__profile_id = profile.pop("id")
        self.data = profile

    def __repr__(self):
        return f"{self.label} ({self.mineral})"

    @property
    def label(self):
        """str: Label of the profile."""
        return self.data["label"]

    @label.setter
    def label(self, lbl: str):
        self.data["label"] = lbl

    @property
    def mineral(self):
        """str: Mineral of the profile."""
        return self.data["mineral"]

    @mineral.setter
    def mineral(self, m: str):
        self.data["mineral"] = m

    @cached_property
    def spots(self):
        """Profile spots accessor"""
        response = self.__db.get(
            f"/profilespots/{self.__project_id}/{self.__sample_id}/{self.__profile_id}"
        )
        if response.ok:
            recs = response.json()
            for rec in recs:
                rec["label"] = self.label
            return PetroDBProfilespotRecords(recs, sample=len(recs) * [self.samplename])
        else:
            raise ValueError(response.json()["detail"])

    def spot(self, spot_id: int):
        """Get profile spot from database

        Args:
            spot_id (int): id of the profile spot

        Returns:
            Profile spot instance.

        Raises:
            ValueError: If profile spot was not found.

        """
        response = self.__db.get(
            f"/profilespot/{self.__project_id}/{self.__sample_id}/{self.__profile_id}/{spot_id}"
        )
        if response.ok:
            return PetroDBProfileSpot(spot=response.json(), **self._kwargs)
        else:
            raise ValueError(response.json()["detail"])

    def create_spot(self, index: int, values: dict):
        """Create profile spot in database

        Args:
            index (int): used to define order of spots on profile
            values (dict): Data values

        Returns:
            Created profile spot instance.

        Raises:
            ValueError: If profile spot was not created.

        """
        data = {"index": index, "values": values}
        response = self.__db.post(
            f"/profilespot/{self.__project_id}/{self.__sample_id}/{self.__profile_id}",
            data,
        )
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    def create_spots(self, df: pd.DataFrame):
        """Create profile spots in database from pandas dataframe

        Args:
            df (pandas.DataFrame): data values, index must be numeric and is used to define order

        Returns:
            Created profile spots

        Raises:
            ValueError: If profile spots were not created.

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
        response = self.__db.post(
            f"/profilespots/{self.__project_id}/{self.__sample_id}/{self.__profile_id}",
            profilespots,
        )
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    def delete(self):
        """Delete profile from database. Use with caution!"""
        response = self.__db.delete(
            f"/profile/{self.__project_id}/{self.__sample_id}/{self.__profile_id}"
        )
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    def update(self):
        """Update profile in database according to the data attribute."""
        response = self.__db.put(
            f"/profile/{self.__project_id}/{self.__sample_id}/{self.__profile_id}",
            self.data,
        )
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    def reset(self):
        """Reset cached property spots to access updated data"""
        if "spots" in self.__dict__:
            self.__dict__["spots"]


class PetroDBProfileSpot:
    """Petro DB profile spot instance

    Attributes:
        data (dict): project attributes

    """

    def __init__(self, db, project_id, sample_id, profile_id, spot):
        self.__db = db
        self.__project_id = project_id
        self.__sample_id = sample_id
        self.__profile_id = profile_id
        self.__profilespot_id = spot.pop("id")
        self.data = spot

    def __repr__(self):
        return f"Spot {self.index}"

    @property
    def index(self):
        """int: Index of the profile spot."""
        return self.data["index"]

    @index.setter
    def index(self, idx: int):
        self.data["index"] = idx

    def delete(self):
        """Delete profile spot from database. Use with caution!"""
        response = self.__db.delete(
            f"/profilespot/{self.__project_id}/{self.__sample_id}/{self.__profile_id}/{self.__profilespot_id}"
        )
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    def update(self):
        """Update profile spot in database according to the data attribute."""
        response = self.__db.put(
            f"/profilespot/{self.__project_id}/{self.__sample_id}/{self.__profile_id}/{self.__profilespot_id}",
            self.data,
        )
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])


class PetroDBRecords:
    """Petro DB records accessor"""

    def __init__(self, recs: list, sample: list):
        self.records = {rec["id"]: rec for rec in recs}
        self.sample = sample
        self.cols = []

    def __repr__(self):
        return f"{len(self.records)} spots"

    def df(self, **kwargs):
        """Get records as pandas dataframe

        Note: Keyword arguments `col=val` are used to select records with given value.
            `col` must be on of the available columns in attribute cols.

        Attributes:
            cols (list): list of attributes for selection

        """
        res = pd.DataFrame(
            [row["values"] for row in self.records.values()], index=self.records.keys()
        )
        res["sample"] = self.sample
        for col in self.cols:
            res[col] = [row[col] for row in self.records.values()]
        for col, val in kwargs.items():
            if col in self.cols:
                res = res[res[col] == val]
        return res.sort_index().copy()


class PetroDBSpotRecords(PetroDBRecords):
    def __init__(self, recs: list, sample: list):
        super().__init__(recs, sample)
        self.cols = ["label", "mineral"]


class PetroDBAreaRecords(PetroDBRecords):
    def __init__(self, recs: list, sample: list):
        super().__init__(recs, sample)
        self.cols = ["label"]


class PetroDBProfilespotRecords(PetroDBRecords):
    def __init__(self, recs: list, sample: list):
        super().__init__(recs, sample)
        self.cols = ["label"]


class PetroDBAdmin:
    """Admin client for postresql petrodb database API."""

    def __init__(self, api_url, username, password):
        self.__db = PetroAPI(api_url, username, password)

    # ---------- USERS

    def users(self, name: str | None = None):
        response = self.__db.get("/users/")
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    def create_user(self, username: str, password: str, email: str):
        """Create user in database

        Args:
            username (str): username
            password (str): password
            email (str): email

        """
        data = {"username": username, "password": password, "email": email}
        response = self.__db.post("/user/", data)
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

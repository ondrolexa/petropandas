"""Client for postresql petrodb database API."""

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
    """Petro DB database instance"""

    def __init__(self):
        pass

    def login(self, api_url, username, password):
        self.__db = PetroAPI(api_url, username, password)

    def __repr__(self):
        return f"PetroDB {'OK' if self.__db.logged else 'Not logged'}"

    def projects(self, **kwargs):
        name = kwargs.get("name", None)
        if name is not None:
            response = self.__db.get(f"/search/project/{name}")
            if response.ok:
                return PetroDBProject(self.__db, project=response.json())
            else:
                if kwargs.get("create", False):
                    return self.create_project(
                        name, description=kwargs.get("description", "")
                    )
                else:
                    raise ValueError(response.json()["detail"])
        else:
            response = self.__db.get("/projects/")
            if response.ok:
                return [PetroDBProject(self.__db, project=p) for p in response.json()]
            else:
                raise ValueError(response.json()["detail"])

    def create_project(self, name: str, description: str = ""):
        data = {"name": name, "description": description}
        response = self.__db.post("/project/", data)
        if response.ok:
            return PetroDBProject(self.__db, project=response.json())
        else:
            raise ValueError(response.json()["detail"])


class PetroDBProject:
    """Petro DB project instance"""

    def __init__(self, db, project):
        self.__db = db
        self.__project_id = project.pop("id")
        self.data = project

    def __repr__(self):
        return f"{self.name}"

    @property
    def name(self):
        return self.data["name"]

    @name.setter
    def name(self, name: str):
        self.data["name"] = name

    @property
    def description(self):
        return self.data["description"]

    @description.setter
    def description(self, desc: str):
        self.data["description"] = desc

    def samples(self, **kwargs):
        name = kwargs.get("name", None)
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
        data = {"name": name, "description": description}
        response = self.__db.post(f"/sample/{self.__project_id}", data)
        if response.ok:
            return PetroDBSample(self.__db, self.__project_id, sample=response.json())
        else:
            raise ValueError(response.json()["detail"])

    def delete(self):
        response = self.__db.delete(f"/project/{self.__project_id}")
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    def update(self):
        response = self.__db.put(f"/project/{self.__project_id}", self.data)
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    def spots(self, **kwargs):
        res = []
        for sample in self.samples():
            try:
                res.append(sample.spots(**kwargs))
            except ValueError:
                pass
        return pd.concat(res, axis=0).sort_index()

    def areas(
        self,
    ):
        res = []
        for sample in self.samples():
            try:
                res.append(sample.areas())
            except ValueError:
                pass
        return pd.concat(res, axis=0).sort_index()


class PetroDBSample:
    """Petro DB sample instance"""

    def __init__(self, db, project_id, sample):
        self.__db = db
        self.__project_id = project_id
        self.__sample_id = sample.pop("id")
        self.data = sample

    def __repr__(self):
        return f"{self.name}"

    @property
    def name(self):
        return self.data["name"]

    @name.setter
    def name(self, name: str):
        self.data["name"] = name

    @property
    def description(self):
        return self.data["description"]

    @description.setter
    def description(self, desc: str):
        self.data["description"] = desc

    def spot(self, *args):
        if args:
            response = self.__db.get(
                f"/spot/{self.__project_id}/{self.__sample_id}/{args[0]}"
            )
            if response.ok:
                return PetroDBSpot(
                    self.__db, self.__project_id, self.__sample_id, spot=response.json()
                )
            else:
                raise ValueError(response.json()["detail"])
        else:
            response = self.__db.get(f"/spots/{self.__project_id}/{self.__sample_id}")
            if response.ok:
                return [
                    PetroDBSpot(self.__db, self.__project_id, self.__sample_id, spot=s)
                    for s in response.json()
                ]
            else:
                raise ValueError(response.json()["detail"])

    def create_spot(
        self,
        label: str,
        mineral: str,
        values: dict,
    ):
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
        """Batch spot insert"""
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

    def area(self, *args):
        if args:
            response = self.__db.get(
                f"/area/{self.__project_id}/{self.__sample_id}/{args[0]}"
            )
            if response.ok:
                return PetroDBArea(
                    self.__db, self.__project_id, self.__sample_id, area=response.json()
                )
            else:
                raise ValueError(response.json()["detail"])
        else:
            response = self.__db.get(f"/areas/{self.__project_id}/{self.__sample_id}")
            if response.ok:
                return [
                    PetroDBArea(self.__db, self.__project_id, self.__sample_id, area=a)
                    for a in response.json()
                ]
            else:
                raise ValueError(response.json()["detail"])

    def create_area(self, label: str, values: dict):
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
        sample: dict,
        df: pd.DataFrame,
        label_col: str | None = None,
    ):
        """Batch area insert"""
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

    def profiles(self, **kwargs):
        label = kwargs.get("label", None)
        if label is not None:
            response = self.__db.get(
                f"/search/profile/{self.__project_id}/{self.__sample_id}/{label}"
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
        else:
            response = self.__db.get(
                f"/profiles/{self.__project_id}/{self.__sample_id}"
            )
            if response.ok:
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
        response = self.__db.delete(f"/sample/{self.__project_id}/{self.__sample_id}")
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    def update(self):
        response = self.__db.put(
            f"/sample/{self.__project_id}/{self.__sample_id}", self.data
        )
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    def spots(self, **kwargs):
        mineral = kwargs.get("mineral", None)
        if mineral is not None:
            response = self.__db.get(
                f"/search/spots/{self.__project_id}/{self.__sample_id}/{mineral}"
            )
        else:
            response = self.__db.get(f"/spots/{self.__project_id}/{self.__sample_id}")
        if response.ok:
            r = response.json()
            res = pd.DataFrame(
                [row["values"] for row in r],
                index=pd.Index([row["id"] for row in r]),
            )
            res["sample"] = self.name
            res["label"] = [row["label"] for row in r]
            res["mineral"] = [row["mineral"] for row in r]
            return res.sort_index()
        else:
            raise ValueError(response.json()["detail"])

    def areas(self):
        response = self.__db.get(f"/areas/{self.__project_id}/{self.__sample_id}")
        if response.ok:
            r = response.json()
            res = pd.DataFrame(
                [row["values"] for row in r], index=pd.Index([row["id"] for row in r])
            )
            res["label"] = [row["label"] for row in r]
            res["sample"] = self.name
            return res.sort_index()
        else:
            raise ValueError(response.json()["detail"])


class PetroDBSpot:
    """Petro DB spot instance"""

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
        return self.data["label"]

    @label.setter
    def label(self, lbl: str):
        self.data["label"] = lbl

    @property
    def mineral(self):
        return self.data["mineral"]

    @mineral.setter
    def mineral(self, m: str):
        self.data["mineral"] = m

    def delete(self):
        response = self.__db.delete(
            f"/spot/{self.__project_id}/{self.__sample_id}/{self.__spot_id}"
        )
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    def update(self):
        response = self.__db.put(
            f"/spot/{self.__project_id}/{self.__sample_id}/{self.__spot_id}",
            self.data,
        )
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])


class PetroDBArea:
    """Petro DB area instance"""

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
        return self.data["label"]

    @label.setter
    def label(self, lbl: str):
        self.data["label"] = lbl

    def delete(self):
        response = self.__db.delete(
            f"/area/{self.__project_id}/{self.__sample_id}/{self.__area_id}"
        )
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    def update(self):
        response = self.__db.put(
            f"/area/{self.__project_id}/{self.__sample_id}/{self.__area_id}",
            self.data,
        )
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])


class PetroDBProfile:
    """Petro DB profile instance"""

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
        return self.data["label"]

    @label.setter
    def label(self, lbl: str):
        self.data["label"] = lbl

    @property
    def mineral(self):
        return self.data["mineral"]

    @mineral.setter
    def mineral(self, m: str):
        self.data["mineral"] = m

    def spot(self, *args):
        if args:
            response = self.__db.get(
                f"/profilespot/{self.__project_id}/{self.__sample_id}/{self.__profile_id}/{args[0]}"
            )
            if response.ok:
                return PetroDBProfileSpot(spot=response.json(), **self._kwargs)
            else:
                raise ValueError(response.json()["detail"])
        else:
            response = self.__db.get(
                f"/profilespot/{self.__project_id}/{self.__sample_id}/{self.__profile_id}"
            )
            if response.ok:
                return [
                    PetroDBProfileSpot(spot=s, **self._kwargs) for s in response.json()
                ]
            else:
                raise ValueError(response.json()["detail"])

    def create_spot(self, index: int, values: dict):
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
        """Batch profilespots insert"""
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
        response = self.__db.delete(
            f"/profile/{self.__project_id}/{self.__sample_id}/{self.__profile_id}"
        )
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    def update(self):
        response = self.__db.put(
            f"/profile/{self.__project_id}/{self.__sample_id}/{self.__profile_id}",
            self.data,
        )
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    def spots(self):
        response = self.__db.get(
            f"/profilespots/{self.__project_id}/{self.__sample_id}/{self.__profile_id}"
        )
        if response.ok:
            r = response.json()
            res = pd.DataFrame(
                [row["values"] for row in r], index=pd.Index([row["id"] for row in r])
            )
            res["index"] = [row["index"] for row in r]
            res["sample"] = self.samplename
            return res.sort_index()
        else:
            raise ValueError(response.json()["detail"])


class PetroDBProfileSpot:
    """Petro DB profile instance"""

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
        return self.data["index"]

    @index.setter
    def label(self, idx: int):
        self.data["index"] = idx

    def delete(self):
        response = self.__db.delete(
            f"/profilespot/{self.__project_id}/{self.__sample_id}/{self.__profile_id}/{self.__profilespot_id}"
        )
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    def update(self):
        response = self.__db.put(
            f"/profilespot/{self.__project_id}/{self.__sample_id}/{self.__profile_id}/{self.__profilespot_id}",
            self.data,
        )
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])


class PetroDBAdmin:
    """Admin client for postresql petrodb database API."""

    def __init__(self, api_url: str, username: str, password: str):
        self.__auth = {"username": username, "password": password}
        self.__api_url = api_url
        # test credentials
        _ = self.__authorize()

    def __authorize(self):
        response = requests.post(f"{self.__api_url}/token", data=self.__auth)
        if response.ok:
            token = response.json()
            return {"Authorization": f"Bearer {token.get('access_token')}"}
        else:
            raise ValueError("Wrong url or credentials")

    def _get(self, path):
        headers = self.__authorize()
        return requests.get(f"{self.__api_url}/api{path}", headers=headers)

    def _post(self, path, data):
        headers = self.__authorize()
        return requests.post(f"{self.__api_url}/api{path}", json=data, headers=headers)

    # ---------- USERS

    def users(self, name: str | None = None):
        response = self.__db.get("/users/")
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    def create_user(self, username: str, password: str, email: str):
        data = {"username": username, "password": password, "email": email}
        response = self.__db.post("/user/", data)
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

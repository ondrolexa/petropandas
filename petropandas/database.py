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


class PetroDB:
    """Petro DB database instance"""

    def __init__(self, *args, **kwargs):
        if args:
            self.__api_url = args[0]
            self.__username = args[1]
            self.__password = args[2]
        else:
            self.__api_url = kwargs["api_url"]
            self.__username = kwargs["username"]
            self.__password = kwargs["password"]
        self._kwargs = {
            "api_url": self.__api_url,
            "username": self.__username,
            "password": self.__password,
        }

    def __repr__(self):
        return f"PetroDB at {self.__api_url}"

    def __authorize(self):
        response = requests.post(
            f"{self.__api_url}/token",
            data={"username": self.__username, "password": self.__password},
        )
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

    def _put(self, path, data):
        headers = self.__authorize()
        return requests.put(f"{self.__api_url}/api{path}", json=data, headers=headers)

    def _delete(self, path):
        headers = self.__authorize()
        return requests.delete(f"{self.__api_url}/api{path}", headers=headers)

    def projects(self, **kwargs):
        name = kwargs.get("name", None)
        if name is not None:
            response = self._get(f"/search/project/{name}")
            if response.ok:
                return PetroDBProject(project=response.json(), **self._kwargs)
            else:
                if kwargs.get("create", False):
                    return self.create_project(
                        name, description=kwargs.get("description", "")
                    )
                else:
                    raise ValueError(response.json()["detail"])
        else:
            response = self._get("/projects/")
            if response.ok:
                return [
                    PetroDBProject(project=p, **self._kwargs) for p in response.json()
                ]
            else:
                raise ValueError(response.json()["detail"])

    def create_project(self, name: str, description: str = ""):
        data = {"name": name, "description": description}
        response = self._post("/project/", data)
        if response.ok:
            return PetroDBProject(project=response.json(), **self._kwargs)
        else:
            raise ValueError(response.json()["detail"])


class PetroDBProject(PetroDB):
    """Petro DB project instance"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._kwargs["project"] = self.project = kwargs["project"]

    def __repr__(self):
        return f"{self.name}"

    @property
    def name(self):
        return self.project["name"]

    @name.setter
    def name(self, name: str):
        self.project["name"] = name

    @property
    def description(self):
        return self.project["description"]

    @description.setter
    def description(self, desc: str):
        self.project["description"] = desc

    def samples(self, **kwargs):
        name = kwargs.get("name", None)
        if name is not None:
            response = self._get(f"/search/sample/{self.project['id']}/{name}")
            if response.ok:
                return PetroDBSample(sample=response.json(), **self._kwargs)
            else:
                raise ValueError(response.json()["detail"])
        else:
            response = self._get(f"/samples/{self.project['id']}")
            if response.ok:
                return [
                    PetroDBSample(sample=s, **self._kwargs) for s in response.json()
                ]
            else:
                raise ValueError(response.json()["detail"])

    def create_sample(self, name: str, description: str = ""):
        data = {"name": name, "description": description}
        response = self._post(f"/sample/{self.project['id']}", data)
        if response.ok:
            return PetroDBSample(sample=response.json(), **self._kwargs)
        else:
            raise ValueError(response.json()["detail"])

    def delete(self):
        response = self._delete(f"/project/{self.project['id']}")
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    def update(self):
        response = self._put(f"/project/{self.project['id']}", self.project)
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


class PetroDBSample(PetroDBProject):
    """Petro DB sample instance"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._kwargs["sample"] = self.sample = kwargs["sample"]

    def __repr__(self):
        return f"{self.name}"

    @property
    def name(self):
        return self.sample["name"]

    @name.setter
    def name(self, name: str):
        self.sample["name"] = name

    @property
    def description(self):
        return self.sample["description"]

    @description.setter
    def description(self, desc: str):
        self.sample["description"] = desc

    def spot(self, *args):
        if args:
            response = self._get(
                f"/spot/{self.project['id']}/{self.sample['id']}/{args[0]}"
            )
            if response.ok:
                return PetroDBSpot(spot=response.json(), **self._kwargs)
            else:
                raise ValueError(response.json()["detail"])
        else:
            response = self._get(f"/spots/{self.project['id']}/{self.sample['id']}")
            if response.ok:
                return [PetroDBSpot(spot=s, **self._kwargs) for s in response.json()]
            else:
                raise ValueError(response.json()["detail"])

    def create_spot(
        self,
        label: str,
        mineral: str,
        values: dict,
    ):
        data = {"label": label, "mineral": mineral, "values": values}
        response = self._post(f"/spot/{self.project['id']}/{self.sample['id']}", data)
        if response.ok:
            return PetroDBSpot(spot=response.json(), **self._kwargs)
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
        response = self._post(f"/spots/{self.project['id']}/{self.sample['id']}", spots)
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    def area(self, *args):
        if args:
            response = self._get(
                f"/area/{self.project['id']}/{self.sample['id']}/{args[0]}"
            )
            if response.ok:
                return PetroDBArea(area=response.json(), **self._kwargs)
            else:
                raise ValueError(response.json()["detail"])
        else:
            response = self._get(f"/areas/{self.project['id']}/{self.sample['id']}")
            if response.ok:
                return [PetroDBArea(area=a, **self._kwargs) for a in response.json()]
            else:
                raise ValueError(response.json()["detail"])

    def create_area(self, label: str, values: dict):
        data = {"label": label, "values": values}
        response = self._post(f"/area/{self.project['id']}/{self.sample['id']}", data)
        if response.ok:
            return PetroDBArea(area=response.json(), **self._kwargs)
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
        response = self._post(f"/areas/{self.project['id']}/{self.sample['id']}", areas)
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    def profiles(self, **kwargs):
        label = kwargs.get("label", None)
        if label is not None:
            response = self._get(
                f"/search/profile/{self.project['id']}/{self.sample['id']}/{label}"
            )
            if response.ok:
                return PetroDBProfile(profile=response.json(), **self._kwargs)
            else:
                raise ValueError(response.json()["detail"])
        else:
            response = self._get(f"/profiles/{self.project['id']}/{self.sample['id']}")
            if response.ok:
                return [
                    PetroDBProfile(profile=p, **self._kwargs) for p in response.json()
                ]
            else:
                raise ValueError(response.json()["detail"])

    def create_profile(self, label: str, mineral: str):
        data = {"label": label, "mineral": mineral}
        response = self._post(
            f"/profile/{self.project['id']}/{self.sample['id']}", data
        )
        if response.ok:
            return PetroDBProfile(profile=response.json(), **self._kwargs)
        else:
            raise ValueError(response.json()["detail"])

    def delete(self):
        response = self._delete(f"/sample/{self.project['id']}/{self.sample['id']}")
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    def update(self):
        response = self._put(
            f"/sample/{self.project['id']}/{self.sample['id']}", self.sample
        )
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    def spots(self, **kwargs):
        mineral = kwargs.get("mineral", None)
        if mineral is not None:
            response = self._get(
                f"/search/spots/{self.project['id']}/{self.sample['id']}/{mineral}"
            )
        else:
            response = self._get(f"/spots/{self.project['id']}/{self.sample['id']}")
        if response.ok:
            r = response.json()
            res = pd.DataFrame(
                [row["values"] for row in r],
                index=pd.Index([row["id"] for row in r]),
            )
            res["sample"] = self.sample["name"]
            res["label"] = [row["label"] for row in r]
            res["mineral"] = [row["mineral"] for row in r]
            return res.sort_index()
        else:
            raise ValueError(response.json()["detail"])

    def areas(self):
        response = self._get(f"/areas/{self.project['id']}/{self.sample['id']}")
        if response.ok:
            r = response.json()
            res = pd.DataFrame(
                [row["values"] for row in r], index=pd.Index([row["id"] for row in r])
            )
            res["label"] = [row["label"] for row in r]
            res["sample"] = self.sample["name"]
            return res.sort_index()
        else:
            raise ValueError(response.json()["detail"])


class PetroDBSpot(PetroDBSample):
    """Petro DB spot instance"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._kwargs["spot"] = self.spot = kwargs["spot"]

    def __repr__(self):
        return f"{self.label} ({self.mineral})"

    @property
    def label(self):
        return self.spot["label"]

    @label.setter
    def label(self, lbl: str):
        self.spot["label"] = lbl

    @property
    def mineral(self):
        return self.spot["mineral"]

    @mineral.setter
    def mineral(self, m: str):
        self.spot["mineral"] = m

    def delete(self):
        response = self._delete(
            f"/spot/{self.project['id']}/{self.sample['id']}/{self.spot['id']}"
        )
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    def update(self):
        response = self._put(
            f"/spot/{self.project['id']}/{self.sample['id']}/{self.spot['id']}",
            self.spot,
        )
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])


class PetroDBArea(PetroDBSample):
    """Petro DB area instance"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._kwargs["area"] = self.area = kwargs["area"]

    def __repr__(self):
        return f"{self.label}"

    @property
    def label(self):
        return self.area["label"]

    @label.setter
    def label(self, lbl: str):
        self.area["label"] = lbl

    def delete(self):
        response = self._delete(
            f"/area/{self.project['id']}/{self.sample['id']}/{self.artea['id']}"
        )
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    def update(self):
        response = self._put(
            f"/area/{self.project['id']}/{self.sample['id']}/{self.artea['id']}",
            self.area,
        )
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])


class PetroDBProfile(PetroDBSample):
    """Petro DB profile instance"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._kwargs["profile"] = self.profile = kwargs["profile"]

    def __repr__(self):
        return f"{self.label} ({self.mineral})"

    @property
    def label(self):
        return self.profile["label"]

    @label.setter
    def label(self, lbl: str):
        self.profile["label"] = lbl

    @property
    def mineral(self):
        return self.profile["mineral"]

    @mineral.setter
    def mineral(self, m: str):
        self.profile["mineral"] = m

    def spot(self, *args):
        if args:
            response = self._get(
                f"/profilespot/{self.project['id']}/{self.sample['id']}/{self.profile['id']}/{args[0]}"
            )
            if response.ok:
                return PetroDBSpot(spot=response.json(), **self._kwargs)
            else:
                raise ValueError(response.json()["detail"])
        else:
            response = self._get(
                f"/profilespot/{self.project['id']}/{self.sample['id']}/{self.profile['id']}"
            )
            if response.ok:
                return [PetroDBSpot(spot=s, **self._kwargs) for s in response.json()]
            else:
                raise ValueError(response.json()["detail"])

    def create_spot(self, index: int, values: dict):
        data = {"index": index, "values": values}
        response = self._post(
            f"/profilespot/{self.project['id']}/{self.sample['id']}/{self.profile['id']}",
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
        response = self._post(
            f"/profilespots/{self.project['id']}/{self.sample['id']}/{self.profile['id']}",
            profilespots,
        )
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    def delete(self):
        response = self._delete(
            f"/profile/{self.project['id']}/{self.sample['id']}/{self.profile['id']}"
        )
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    def update(self):
        response = self._put(
            f"/profile/{self.project['id']}/{self.sample['id']}/{self.profile['id']}",
            self.profile,
        )
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    def spots(self):
        response = self._get(
            f"/profilespots/{self.project['id']}/{self.sample['id']}/{self.profile['id']}"
        )
        if response.ok:
            r = response.json()
            res = pd.DataFrame(
                [row["values"] for row in r], index=pd.Index([row["id"] for row in r])
            )
            res["index"] = [row["index"] for row in r]
            res["sample"] = self.sample["name"]
            return res.sort_index()
        else:
            raise ValueError(response.json()["detail"])


class PetroDBProfileSpot(PetroDBProfile):
    """Petro DB profile instance"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._kwargs["profilespot"] = self.profilespot = kwargs["profilespot"]

    def __repr__(self):
        return f"Spot {self.index}"

    @property
    def index(self):
        return self.profilespot["index"]

    @index.setter
    def label(self, idx: int):
        self.profilespot["index"] = idx

    def delete(self):
        response = self._delete(
            f"/profilespot/{self.project['id']}/{self.sample['id']}/{self.profile['id']}/{self.profilespot['id']}"
        )
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    def update(self):
        response = self._put(
            f"/profilespot/{self.project['id']}/{self.sample['id']}/{self.profile['id']}/{self.profilespot['id']}",
            self.profilespot,
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
        response = self._get("/users/")
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    def create_user(self, username: str, password: str, email: str):
        data = {"username": username, "password": password, "email": email}
        response = self._post("/user/", data)
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

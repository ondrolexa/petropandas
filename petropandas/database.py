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
    """Client for postresql petrodb database API."""

    def __init__(self, api_url: str, username: str, password: str):
        self.auth = {"username": username, "password": password}
        self.api_url = api_url
        # test credentials
        _ = self.authorize()

    def authorize(self):
        response = requests.post(f"{self.api_url}/token", data=self.auth)
        if response.ok:
            token = response.json()
            return {"Authorization": f"Bearer {token.get('access_token')}"}
        else:
            raise ValueError("Wrong url or credentials")

    def get(self, path):
        headers = self.authorize()
        return requests.get(f"{self.api_url}/api{path}", headers=headers)

    def post(self, path, data):
        headers = self.authorize()
        return requests.post(f"{self.api_url}/api{path}", json=data, headers=headers)

    # ---------- PROJECTS

    def projects(self, **kwargs):
        name = kwargs.get("name", None)
        if name is not None:
            response = self.get(f"/search/project/{name}")
            if response.ok:
                return response.json()
            else:
                if kwargs.get("create", False):
                    return self.create_project(
                        name, description=kwargs.get("description", "")
                    )
                else:
                    raise ValueError(response.json()["detail"])
        else:
            response = self.get("/projects/")
            if response.ok:
                return response.json()
            else:
                raise ValueError(response.json()["detail"])

    def create_project(self, name: str, description: str = ""):
        data = {"name": name, "description": description}
        response = self.post("/project/", data)
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    # ---------- SAMPLES

    def samples(self, project: dict, **kwargs):
        name = kwargs.get("name", None)
        if name is not None:
            response = self.get(f"/search/sample/{project['id']}/{name}")
            if response.ok:
                return response.json()
            else:
                if kwargs.get("create", False):
                    return self.create_sample(
                        project, name, description=kwargs.get("description", "")
                    )
                else:
                    raise ValueError(response.json()["detail"])
        else:
            response = self.get(f"/samples/{project['id']}")
            if response.ok:
                return response.json()
            else:
                raise ValueError(response.json()["detail"])

    def create_sample(self, project: dict, name: str, description: str = ""):
        data = {"name": name, "description": description}
        response = self.post(f"/sample/{project['id']}", data)
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    # ---------- SPOTS

    def spots(self, project: dict, sample: dict = {}, **kwargs):
        mineral = kwargs.get("mineral", None)
        if not sample:
            res = []
            for sample in self.samples(project):
                try:
                    res.append(self.spots(project, sample, **kwargs))
                except ValueError:
                    pass
            return pd.concat(res, axis=0)
        if mineral is not None:
            response = self.get(
                f"/search/spots/{project['id']}/{sample['id']}/{mineral}"
            )
        else:
            response = self.get(f"/spots/{project['id']}/{sample['id']}")
        if response.ok:
            r = response.json()
            res = pd.DataFrame(
                [row["values"] for row in r],
                index=pd.Index([row["id"] for row in r]),
            )
            res["sample"] = sample["name"]
            res["label"] = [row["label"] for row in r]
            res["mineral"] = [row["mineral"] for row in r]
            return res
        else:
            raise ValueError(response.json()["detail"])

    def create_spot(
        self,
        project: dict,
        sample: dict,
        label: str,
        mineral: str,
        values: dict,
    ):
        data = {"label": label, "mineral": mineral, "values": values}
        response = self.post(f"/spot/{project['id']}/{sample['id']}", data)
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    def create_spots(
        self,
        project: dict,
        sample: dict,
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
        response = self.post(f"/spots/{project['id']}/{sample['id']}", spots)
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    # ---------- AREAS

    def areas(self, project: dict, sample: dict = {}):
        if not sample:
            res = []
            for sample in self.samples(project):
                try:
                    res.append(self.areas(project, sample))
                except ValueError:
                    pass
            return pd.concat(res, axis=0)
        response = self.get(f"/areas/{project['id']}/{sample['id']}")
        if response.ok:
            r = response.json()
            res = pd.DataFrame(
                [row["values"] for row in r], index=pd.Index([row["id"] for row in r])
            )
            res["label"] = [row["label"] for row in r]
            res["sample"] = sample["name"]
            return res
        else:
            raise ValueError(response.json()["detail"])

    def create_area(self, project: dict, sample: dict, label: str, values: dict):
        data = {"label": label, "values": values}
        response = self.post(f"/area/{project['id']}/{sample['id']}", data)
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    def create_areas(
        self,
        project: dict,
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
        response = self.post(f"/areas/{project['id']}/{sample['id']}", areas)
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    # ---------- PROFILES

    def profiles(self, project: dict, sample: dict, **kwargs):
        label = kwargs.get("label", None)
        if label is not None:
            response = self.get(
                f"/search/profile/{project['id']}/{sample['id']}/{label}"
            )
            if response.ok:
                return response.json()
            else:
                raise ValueError(response.json()["detail"])
        else:
            response = self.get(f"/profiles/{project['id']}/{sample['id']}")
            if response.ok:
                return response.json()
            else:
                raise ValueError(response.json()["detail"])

    def create_profile(self, project: dict, sample: dict, label: str, mineral: str):
        data = {"label": label, "mineral": mineral}
        response = self.post(f"/profile/{project['id']}/{sample['id']}", data)
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    # ---------- PROFILE SPOTS

    def profilespots(self, project: dict, sample: dict, profile: dict):
        response = self.get(
            f"/profilespots/{project['id']}/{sample['id']}/{profile['id']}"
        )
        if response.ok:
            r = response.json()
            res = pd.DataFrame(
                [row["values"] for row in r], index=pd.Index([row["id"] for row in r])
            )
            res["index"] = [row["index"] for row in r]
            res["sample"] = sample["name"]
            return res
        else:
            raise ValueError(response.json()["detail"])

    def create_profilespot(
        self, project: dict, sample: dict, profile: dict, index: int, values: dict
    ):
        data = {"index": index, "values": values}
        response = self.post(
            f"/profilespot/{project['id']}/{sample['id']}/{profile['id']}", data
        )
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    def create_profilespots(
        self, project: dict, sample: dict, profile: dict, df: pd.DataFrame
    ):
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
        response = self.post(
            f"/profilespots/{project['id']}/{sample['id']}/{profile['id']}",
            profilespots,
        )
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])


class PetroDBAdmin:
    """Admin client for postresql petrodb database API."""

    def __init__(self, api_url: str, username: str, password: str):
        self.auth = {"username": username, "password": password}
        self.api_url = api_url
        # test credentials
        _ = self.authorize()

    def authorize(self):
        response = requests.post(f"{self.api_url}/token", data=self.auth)
        if response.ok:
            token = response.json()
            return {"Authorization": f"Bearer {token.get('access_token')}"}
        else:
            raise ValueError("Wrong url or credentials")

    def get(self, path):
        headers = self.authorize()
        return requests.get(f"{self.api_url}/api{path}", headers=headers)

    def post(self, path, data):
        headers = self.authorize()
        return requests.post(f"{self.api_url}/api{path}", json=data, headers=headers)

    # ---------- USERS

    def users(self, name: str | None = None):
        response = self.get("/users/")
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    def create_user(self, username: str, password: str, email: str):
        data = {"username": username, "password": password, "email": email}
        response = self.post("/user/", data)
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

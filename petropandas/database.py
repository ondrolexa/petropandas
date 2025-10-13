import pandas as pd
import requests


class PetroDB:
    """Client for postresql petrodb database API."""

    def __init__(self, api_url: str, username: str, password: str):
        self.auth = {"username": username, "password": password}
        self.api_url = api_url

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

    def projects(self, name: str | None = None):
        if name is not None:
            response = self.get(f"/search/project/{name}")
            if response.ok:
                return response.json()
            else:
                raise ValueError(response.json()["detail"])
        else:
            response = self.get("/projects")
            if response.ok:
                return response.json()
            else:
                raise ValueError(response.json()["detail"])

    def create_project(self, name: str, description: str = ""):
        data = {"name": name, "description": description}
        response = self.post("/project", data)
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    # ---------- SAMPLES

    def samples(self, project: dict, name: str | None = None):
        if name is not None:
            response = self.get(f"/search/sample/{project['id']}/{name}")
            if response.ok:
                return response.json()
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

    def spots(self, project: dict, sample: dict, mineral: str | None = None):
        if mineral is not None:
            response = self.get(
                f"/search/spots/{project['id']}/{sample['id']}/{mineral}"
            )
            if response.ok:
                return response.json()
            else:
                raise ValueError(response.json()["detail"])
        else:
            response = self.get(f"/spots/{project['id']}/{sample['id']}")
            if response.ok:
                r = response.json()
                res = pd.DataFrame(
                    [row["values"] for row in r], index=[row["id"] for row in r]
                )
                res["label"] = [row["label"] for row in r]
                res["mineral"] = [row["mineral"] for row in r]
                return res
            else:
                raise ValueError(response.json()["detail"])

    def create_spot(
        self, project: dict, sample: dict, label: str, mineral: str, values: dict
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
        label_col: str = None,
        mineral_col: str = None,
    ):
        """Batch spot insert"""

        if label_col is None:
            labels = pd.Series(df.index.astype(str), index=df.index)
        else:
            labels = df[label_col].str.strip()
        if mineral_col is None:
            minerals = pd.Series("", index=df.index)
        else:
            minerals = df[mineral_col].str.strip()
        spots = []
        for label, mineral, (ix, row) in zip(
            labels, minerals, df[df.oxides._names].iterrows()
        ):
            spots.append({"label": label, "mineral": mineral, "values": row.to_dict()})
        response = self.post(f"/spots/{project['id']}/{sample['id']}", spots)
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    # ---------- AREAS

    def areas(self, project: dict, sample: dict):
        response = self.get(f"/areas/{project['id']}/{sample['id']}")
        if response.ok:
            r = response.json()
            res = pd.DataFrame(
                [row["values"] for row in r], index=[row["id"] for row in r]
            )
            res["label"] = [row["label"] for row in r]
            res["weight"] = [row["weight"] for row in r]
            return res
        else:
            raise ValueError(response.json()["detail"])

    def create_area(
        self, project: dict, sample: dict, label: str, weight: float, values: dict
    ):
        data = {"label": label, "weight": weight, "values": values}
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
        label_col: str = None,
        weight_col: str = None,
    ):
        """Batch area insert"""

        if label_col is None:
            labels = pd.Series(df.index.astype(str), index=df.index)
        else:
            labels = df[label_col].str.strip()
        if weight_col is None:
            weights = pd.Series(1, index=df.index)
        else:
            weights = pd.to_numeric(df[weight_col])
        areas = []
        for label, weight, (ix, row) in zip(
            labels, weights, df[df.oxides._names].iterrows()
        ):
            areas.append({"label": label, "weight": weight, "values": row.to_dict()})
        response = self.post(f"/areas/{project['id']}/{sample['id']}", areas)
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

    # ---------- PROFILES

    def profiles(self, project: dict, sample: dict):
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
            f"/profilespot/{project['id']}/{sample['id']}/{profile['id']}"
        )
        if response.ok:
            r = response.json()
            res = pd.DataFrame(
                [row["values"] for row in r], index=[row["id"] for row in r]
            )
            res["index"] = [row["index"] for row in r]
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

    def create_profilespots(self, project: dict, sample: dict, df: pd.DataFrame):
        """Batch profilespots insert"""

        profilespots = []
        for index, row in df[df.oxides._names].iterrows():
            profilespots.append({"index": int(index), "values": row.to_dict()})
        response = self.post(
            f"/profilespots/{project['id']}/{sample['id']}", profilespots
        )
        if response.ok:
            return response.json()
        else:
            raise ValueError(response.json()["detail"])

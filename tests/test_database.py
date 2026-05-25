from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from petropandas.database import (
    APIError,
    AuthError,
    NotFoundError,
    PetroAPI,
    PetroDB,
    PetroDBAdmin,
    PetroDBArea,
    PetroDBProfile,
    PetroDBProfileSpot,
    PetroDBProfileSpotRecords,
    PetroDBProject,
    PetroDBRecords,
    PetroDBSample,
    PetroDBSpot,
    PetroDBSpotRecords,
    zero_negative_nan,
)

# ==========================================================
# zero_negative_nan
# ==========================================================


class TestZeroNegativeNan:
    def test_positive_number(self):
        assert zero_negative_nan(5.0) == 5.0

    def test_negative_number(self):
        import math

        assert math.isnan(zero_negative_nan(-1.0))

    def test_zero(self):
        import math

        assert math.isnan(zero_negative_nan(0.0))

    def test_non_number(self):
        assert zero_negative_nan("foo") == "foo"

    def test_bool_true(self):
        assert zero_negative_nan(True) is True

    def test_bool_false(self):
        assert zero_negative_nan(False) is False

    def test_complex_positive(self):
        assert zero_negative_nan(1 + 2j) == 1 + 2j

    def test_complex_negative_real(self):
        import math

        assert math.isnan(zero_negative_nan(-1 + 2j))


# ==========================================================
# PetroAPI
# ==========================================================


class TestPetroAPI:
    @patch("petropandas.database.requests.Session")
    def test_login_success(self, mock_session):
        mock_session.return_value.post.return_value.ok = True
        mock_session.return_value.post.return_value.json.return_value = {
            "access_token": "test-token"
        }

        api = PetroAPI("http://test.url", "user", "pass")
        assert api.logged

    @patch("petropandas.database.requests.Session")
    def test_login_failure(self, mock_session):
        mock_session.return_value.post.return_value.ok = False

        api = PetroAPI("http://test.url", "user", "pass")
        assert not api.logged

    @patch("petropandas.database.requests.Session")
    def test_get_success(self, mock_session):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_session.return_value.post.return_value.ok = True
        mock_session.return_value.post.return_value.json.return_value = {
            "access_token": "test"
        }
        mock_session.return_value.get.return_value = mock_response

        api = PetroAPI("http://test.url", "user", "pass")
        response = api.get("/projects/")
        assert response.ok

    @patch("petropandas.database.requests.Session")
    def test_authorize_when_not_logged(self, mock_session):
        mock_session.return_value.post.return_value.ok = False

        api = PetroAPI("http://test.url", "user", "pass")
        with pytest.raises(AuthError):
            api.get("/projects/")

    @patch("petropandas.database.requests.Session")
    def test_post(self, mock_session):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_session.return_value.post.return_value.ok = True
        mock_session.return_value.post.return_value.json.return_value = {
            "access_token": "test"
        }

        api = PetroAPI("http://test.url", "user", "pass")
        mock_session.return_value.post.return_value = mock_response
        response = api.post("/project/", {"name": "test"})
        assert response.ok

    @patch("petropandas.database.requests.Session")
    def test_put(self, mock_session):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_session.return_value.post.return_value.ok = True
        mock_session.return_value.post.return_value.json.return_value = {
            "access_token": "test"
        }

        api = PetroAPI("http://test.url", "user", "pass")
        mock_session.return_value.put.return_value = mock_response
        response = api.put("/project/1", {"name": "test"})
        assert response.ok

    @patch("petropandas.database.requests.Session")
    def test_delete(self, mock_session):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_session.return_value.post.return_value.ok = True
        mock_session.return_value.post.return_value.json.return_value = {
            "access_token": "test"
        }

        api = PetroAPI("http://test.url", "user", "pass")
        mock_session.return_value.delete.return_value = mock_response
        response = api.delete("/project/1")
        assert response.ok

    @patch("petropandas.database.requests.Session")
    def test_logged_property(self, mock_session):
        mock_session.return_value.post.return_value.ok = True
        mock_session.return_value.post.return_value.json.return_value = {
            "access_token": "test"
        }

        _ = PetroAPI("http://test.url", "user", "pass")
        db = PetroDB("http://test.url", "user", "pass")
        assert "OK" in repr(db)

    @patch("petropandas.database.requests.Session")
    def test_timeout_configurable(self, mock_session):
        mock_session.return_value.post.return_value.ok = True
        mock_session.return_value.post.return_value.json.return_value = {
            "access_token": "test"
        }

        api = PetroAPI("http://test.url", "user", "pass", timeout=60)
        assert api._timeout == 60

    @patch("petropandas.database.requests.Session")
    def test_token_refresh_on_401(self, mock_session):
        """Verify a 401 triggers re-auth and retry."""
        failed = MagicMock()
        failed.ok = False
        failed.status_code = 401

        success = MagicMock()
        success.ok = True
        success.status_code = 200
        success.json.return_value = [{"id": 1, "name": "P", "description": ""}]

        mock_session.return_value.post.return_value.ok = True
        mock_session.return_value.post.return_value.json.return_value = {
            "access_token": "test"
        }

        api = PetroAPI("http://test.url", "user", "pass")
        mock_session.return_value.get.side_effect = [failed, success]

        resp = api.get("/projects/")
        assert resp.ok

    @patch("petropandas.database.requests.Session")
    def test_context_manager_closes_session(self, mock_session):
        mock_session.return_value.post.return_value.ok = True
        mock_session.return_value.post.return_value.json.return_value = {
            "access_token": "test"
        }

        with PetroDB("http://test.url", "user", "pass") as db:
            assert db.logged
        mock_session.return_value.close.assert_called_once()


# ==========================================================
# PetroDBRecords
# ==========================================================


class TestPetroDBRecords:
    @pytest.fixture
    def records(self):
        r = {
            1: MagicMock(data={"values": {"SiO2": 45.0, "MgO": 5.0}}),
            2: MagicMock(data={"values": {"SiO2": 50.0, "MgO": 10.0}}),
        }
        r[1].data["label"] = "A"
        r[1].data["mineral"] = "Grt"
        r[2].data["label"] = "B"
        r[2].data["mineral"] = "Grt"
        return r

    def test_create(self, records):
        pr = PetroDBRecords(records, ["s1", "s1"])
        pr.cols = ["label", "mineral"]
        assert pr[1] is records[1]

    def test_repr(self, records):
        pr = PetroDBRecords(records, ["s1", "s1"])
        pr.cols = ["label", "mineral"]

    def test_df(self, records):
        pr = PetroDBRecords(records, ["s1", "s1"])
        pr.cols = ["label", "mineral"]
        df = pr.df()
        assert len(df) == 2
        assert "SiO2" in df.columns

    def test_df_filter_by_col(self):
        records = {
            1: MagicMock(data={"values": {"SiO2": 45.0}}),
            2: MagicMock(data={"values": {"SiO2": 50.0}}),
        }
        records[1].data["label"] = "A"
        records[1].data["mineral"] = "Grt"
        records[2].data["label"] = "B"
        records[2].data["mineral"] = "Pl"
        pr = PetroDBRecords(records, ["s1", "s1"])
        pr.cols = ["label", "mineral"]
        df = pr.df(mineral="Grt")
        assert len(df) == 1

    def test_getitem_by_id(self, records):
        pr = PetroDBRecords(records, ["s1", "s1"])
        pr.cols = ["label", "mineral"]
        assert pr[1] == records[1]


# ==========================================================
# PetroDBProject
# ==========================================================


class TestPetroDBProject:
    @pytest.fixture
    def project(self):
        return PetroDBProject(
            None, {"id": 1, "name": "MyProject", "description": "desc"}
        )

    def test_repr(self, project):
        assert repr(project) == "MyProject"
        assert project.name == "MyProject"
        assert project.description == "desc"

    def test_setters(self, project):
        project.name = "New"
        project.description = "new"
        assert project.name == "New"
        assert project.description == "new"

    @patch("petropandas.database.requests.Session")
    def test_add_user(self, mock_session):
        mock_session.return_value.post.return_value.ok = True
        mock_session.return_value.post.return_value.json.return_value = {
            "access_token": "test"
        }
        mock_session.return_value.put.return_value.ok = True
        mock_session.return_value.put.return_value.json.return_value = {"message": "ok"}

        db = PetroDB("http://test.url", "user", "pass")
        project = PetroDBProject(db._db, {"id": 1, "name": "P", "description": ""})
        result = project.add_user("alice")
        assert result == {"message": "ok"}

    @patch("petropandas.database.requests.Session")
    def test_remove_user(self, mock_session):
        mock_session.return_value.post.return_value.ok = True
        mock_session.return_value.post.return_value.json.return_value = {
            "access_token": "test"
        }
        mock_session.return_value.put.return_value.ok = True
        mock_session.return_value.put.return_value.json.return_value = {"message": "ok"}

        db = PetroDB("http://test.url", "user", "pass")
        project = PetroDBProject(db._db, {"id": 1, "name": "P", "description": ""})
        result = project.remove_user("alice")
        assert result == {"message": "ok"}


# ==========================================================
# PetroDBSample
# ==========================================================


class TestPetroDBSample:
    @pytest.fixture
    def sample(self):
        return PetroDBSample(None, 1, {"id": 1, "name": "S1", "description": "test"})

    def test_repr(self, sample):
        assert repr(sample) == "S1"
        assert sample.name == "S1"
        assert sample.description == "test"

    def test_setters(self, sample):
        sample.name = "New"
        sample.description = "new"
        assert sample.name == "New"
        assert sample.description == "new"

    def test_reset_clears_cached_spots(self):
        sample = PetroDBSample(None, 1, {"id": 1, "name": "S1", "description": "test"})
        sample.__dict__["spots"] = "cached"
        sample.__dict__["areas"] = "cached"
        sample.reset()
        assert "spots" not in sample.__dict__
        assert "areas" not in sample.__dict__

    def test_reset_partial(self):
        sample = PetroDBSample(None, 1, {"id": 1, "name": "S1", "description": "test"})
        sample.__dict__["spots"] = "cached"
        sample.reset()
        assert "spots" not in sample.__dict__

    @patch("petropandas.database.requests.Session")
    def test_mineral_data_produces_dataframe(self, mock_session):
        """Verify mineral_data assembles a DataFrame with spots + profiles."""
        mock_session.return_value.post.return_value.ok = True
        mock_session.return_value.post.return_value.json.return_value = {
            "access_token": "test"
        }

        # GET /spots/ -> one Grt spot
        mock_session.return_value.get.side_effect = [
            # first call: spots
            MagicMock(
                ok=True,
                json=lambda: [
                    {
                        "id": 1,
                        "label": "S1-A",
                        "mineral": "Grt",
                        "values": {"SiO2": 37.0, "MgO": 5.0},
                    }
                ],
            ),
            # second call: profiles -> empty list
            MagicMock(ok=True, json=lambda: []),
        ]

        db = PetroDB("http://test.url", "user", "pass")
        sample = PetroDBSample(
            db._db, 1, {"id": 1, "name": "S1", "description": "test"}
        )
        df = sample.mineral_data("Grt")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    @patch("petropandas.database.requests.Session")
    def test_create_sposts_builds_payload(self, mock_session):
        """Verify create_spots sends a POST with the right payload."""
        mock_session.return_value.post.return_value.ok = True
        mock_session.return_value.post.return_value.json.return_value = {
            "access_token": "test"
        }

        api = PetroAPI("http://test.url", "user", "pass")
        mock_session.return_value.post.return_value = MagicMock(
            ok=True,
            json=lambda: [
                {"id": 1, "label": "A", "mineral": "Grt", "values": {"SiO2": 45.0}}
            ],
        )

        sample = PetroDBSample(api, 1, {"id": 1, "name": "S1", "description": "test"})
        df = pd.DataFrame({"SiO2": [45.0]}, index=["A"])
        result = sample.create_spots(df, mineral_col=None)
        assert isinstance(result, PetroDBSpotRecords)

    @patch("petropandas.database.requests.Session")
    def test_profilespots_property(self, mock_session):
        """Verify profilespots returns a PetroDBProfileSpotRecords instance."""
        mock_session.return_value.post.return_value.ok = True
        mock_session.return_value.post.return_value.json.return_value = {
            "access_token": "test"
        }
        # First call: profiles() -> one profile
        # Second call: profile.spots -> empty dict
        mock_session.return_value.get.side_effect = [
            MagicMock(
                ok=True,
                json=lambda: [{"id": 1, "label": "P1", "mineral": "Grt"}],
            ),
            MagicMock(ok=True, json=lambda: []),
        ]

        db = PetroDB("http://test.url", "user", "pass")
        sample = PetroDBSample(
            db._db, 1, {"id": 1, "name": "S1", "description": "test"}
        )
        result = sample.profilespots
        assert isinstance(result, PetroDBProfileSpotRecords)
        assert len(result.records) == 0


# ==========================================================
# PetroDBSpot
# ==========================================================


class TestPetroDBSpot:
    @pytest.fixture
    def spot(self):
        return PetroDBSpot(None, 1, 2, {"id": 1, "label": "S1-A", "mineral": "Grt"})

    def test_repr(self, spot):
        assert "S1-A" in repr(spot)
        assert spot.label == "S1-A"
        assert spot.mineral == "Grt"

    def test_setters(self, spot):
        spot.label = "New"
        spot.mineral = "Pl"
        assert spot.label == "New"
        assert spot.mineral == "Pl"


# ==========================================================
# PetroDBArea
# ==========================================================


class TestPetroDBArea:
    @pytest.fixture
    def area(self):
        return PetroDBArea(None, 1, 2, {"id": 1, "label": "Area1"})

    def test_repr(self, area):
        assert "Area1" in repr(area)
        assert area.label == "Area1"

    def test_setter(self, area):
        area.label = "New"
        assert area.label == "New"


# ==========================================================
# PetroDBProfile
# ==========================================================


class TestPetroDBProfile:
    @pytest.fixture
    def profile(self):
        return PetroDBProfile(
            None, 1, 2, "S1", {"id": 1, "label": "P1", "mineral": "Grt"}
        )

    def test_repr(self, profile):
        assert "P1" in repr(profile)
        assert profile.label == "P1"
        assert profile.mineral == "Grt"

    def test_setters(self, profile):
        profile.label = "New"
        profile.mineral = "Pl"
        assert profile.label == "New"
        assert profile.mineral == "Pl"

    def test_reset_clears_cached_spots(self):
        profile = PetroDBProfile(
            None, 1, 2, "S1", {"id": 1, "label": "P1", "mineral": "Grt"}
        )
        profile.__dict__["spots"] = "cached"
        profile.reset()
        assert "spots" not in profile.__dict__

    @patch("petropandas.database.requests.Session")
    def test_create_spot_returns_instance(self, mock_session):
        mock_session.return_value.post.return_value.ok = True
        mock_session.return_value.post.return_value.json.return_value = {
            "access_token": "test"
        }

        api = PetroAPI("http://test.url", "user", "pass")
        mock_session.return_value.post.return_value = MagicMock(
            ok=True, json=lambda: {"id": 10, "index": 1, "values": {"SiO2": 50.0}}
        )

        profile = PetroDBProfile(
            api, 1, 2, "S1", {"id": 1, "label": "P1", "mineral": "Grt"}
        )
        spot = profile.create_spot(1, {"SiO2": 50.0})
        assert isinstance(spot, PetroDBProfileSpot)
        assert spot.index == 1


# ==========================================================
# PetroDBProfileSpot
# ==========================================================


class TestPetroDBProfileSpot:
    @pytest.fixture
    def spot(self):
        return PetroDBProfileSpot(None, 1, 2, 3, {"id": 1, "index": 1})

    def test_repr(self, spot):
        assert "1" in repr(spot)
        assert spot.index == 1

    def test_setter(self, spot):
        spot.index = 5
        assert spot.index == 5


# ==========================================================
# PetroDBProfileSpotRecords
# ==========================================================


class TestPetroDBProfileSpotRecords:
    def test_create(self):
        records = {
            1: MagicMock(data={"values": {"SiO2": 45.0}}),
        }
        records[1].data["label"] = "A"
        records[1].data["mineral"] = "Grt"
        psr = PetroDBProfileSpotRecords(records, ["s1"])
        assert psr.cols == ["label", "mineral"]
        assert "profile spots" in repr(psr)


# ==========================================================
# PetroDBAdmin
# ==========================================================


class TestPetroDBAdmin:
    @patch("petropandas.database.requests.Session")
    def test_login_failure(self, mock_session):
        mock_session.return_value.post.return_value.ok = False
        admin = PetroDBAdmin("http://test.url", "user", "pass")
        assert not admin._db.logged

    @patch("petropandas.database.requests.Session")
    def test_users(self, mock_session):
        mock_session.return_value.post.return_value.ok = True
        mock_session.return_value.post.return_value.json.return_value = {
            "access_token": "test"
        }
        mock_session.return_value.get.return_value.ok = True
        mock_session.return_value.get.return_value.json.return_value = [
            {"username": "alice"}
        ]

        admin = PetroDBAdmin("http://test.url", "user", "pass")
        users = admin.users()
        assert users == [{"username": "alice"}]

    @patch("petropandas.database.requests.Session")
    def test_create_user(self, mock_session):
        mock_session.return_value.post.return_value.ok = True
        mock_session.return_value.post.return_value.json.return_value = {
            "access_token": "test"
        }

        admin = PetroDBAdmin("http://test.url", "user", "pass")
        mock_session.return_value.post.return_value = MagicMock(
            ok=True, json=lambda: {"username": "bob", "email": "bob@test.com"}
        )
        result = admin.create_user("bob", "secret", "bob@test.com")
        assert result["username"] == "bob"


# ==========================================================
# Exceptions
# ==========================================================


class TestExceptions:
    def test_raise_auth_error(self):
        with pytest.raises(AuthError):
            raise AuthError("bad credentials")

    def test_raise_api_error(self):
        with pytest.raises(APIError, match="server error"):
            raise APIError("server error")

    def test_raise_not_found_error(self):
        with pytest.raises(NotFoundError):
            raise NotFoundError("not here")

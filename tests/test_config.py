"""Tests for ppconfig (PPConfig singleton)."""

from petropandas import PPConfig, ppconfig


class TestPPConfigDefaults:
    def test_default_system(self):
        assert ppconfig.default_system == "MnNCKFMASHTO"

    def test_default_oxygen(self):
        assert ppconfig.default_oxygen == 0.01

    def test_default_H2O(self):
        assert ppconfig.default_H2O == -1.0

    def test_default_db(self):
        assert ppconfig.default_db == "mp"

    def test_default_sys_in(self):
        assert ppconfig.default_sys_in == "mol"


class TestPPConfigMutation:
    def test_default_system_set(self):
        ppconfig.default_system = "KFMASH"
        assert ppconfig.default_system == "KFMASH"
        ppconfig.reset()

    def test_default_oxygen_set(self):
        ppconfig.default_oxygen = 0.1
        assert ppconfig.default_oxygen == 0.1
        ppconfig.reset()

    def test_default_H2O_set(self):
        ppconfig.default_H2O = 5.0
        assert ppconfig.default_H2O == 5.0
        ppconfig.reset()

    def test_default_db_set(self):
        ppconfig.default_db = "ig"
        assert ppconfig.default_db == "ig"
        ppconfig.reset()

    def test_default_sys_in_set(self):
        ppconfig.default_sys_in = "wt"
        assert ppconfig.default_sys_in == "wt"
        ppconfig.reset()


class TestPPConfigReset:
    def test_reset_restores_defaults(self):
        ppconfig.default_system = "KFMASH"
        ppconfig.default_oxygen = 0.9
        ppconfig.default_H2O = 99.0
        ppconfig.default_db = "ig"
        ppconfig.default_sys_in = "wt"

        ppconfig.reset()

        assert ppconfig.default_system == "MnNCKFMASHTO"
        assert ppconfig.default_oxygen == 0.01
        assert ppconfig.default_H2O == -1.0
        assert ppconfig.default_db == "mp"
        assert ppconfig.default_sys_in == "mol"


class TestPPConfigExport:
    def test_ppconfig_is_ppconfig_instance(self):
        assert isinstance(ppconfig, PPConfig)

    def test_ppconfig_in_all(self):
        import petropandas

        assert "ppconfig" in petropandas.__all__
        assert "PPConfig" in petropandas.__all__

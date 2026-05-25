import numpy as np

from petropandas import pd


class TestReadActlabs:
    def _write_actlabs(self, tmp_path, filename, data_rows, has_dl=False):
        """Write an ActLabs-style Excel file with proper row layout.

        Expected by read_actlabs(skiprows=2):
          Row 0-1: skipped
          Row 2:   column header (Analyte Symbol, SiO2, ...)
          Row 3:   units
          Row 4:   limits
          Row 5:   method
          Row 6+:  data
        """
        header = ["Analyte Symbol", "SiO2", "Fe2O3(T)", "MgO"]
        units = ["Unit", "%", "%", "%"]
        limits = ["Limit", "0.01", "0.01", "0.01"]
        method = ["Method", "XRF", "XRF", "XRF"]
        rows = [[""] * 4, [""] * 4, header, units, limits, method] + data_rows
        src = tmp_path / filename
        pd.DataFrame(rows).to_excel(src, index=False, header=False)
        return src

    def test_parse_actlabs_skiprows_default(self, tmp_path):
        from petropandas.io import read_actlabs

        data = [
            ["S1", 45.0, 12.5, 5.0],
            ["S2", 50.0, 0.2, 10.0],
        ]
        src = self._write_actlabs(tmp_path, "test.xlsx", data)
        df, units, limits, method = read_actlabs(str(src))
        assert df.index.name == "Sample"
        assert "Fe2O3" in df.columns
        assert df.loc["S1", "SiO2"] == 45.0

    def test_detection_limit_replaced_with_nan(self, tmp_path):
        from petropandas.io import read_actlabs

        data = [
            ["S1", 45.0, 12.5, 5.0],
            ["S2", 50.0, "< 0.01", 10.0],
        ]
        src = self._write_actlabs(tmp_path, "test_dl.xlsx", data, has_dl=True)
        df, units, limits, method = read_actlabs(str(src))
        assert np.isnan(df.loc["S2", "Fe2O3"])

    def test_custom_skiprows(self, tmp_path):
        from petropandas.io import read_actlabs

        data = [["S1", 45.0, 12.5, 5.0]]
        units = ["Unit", "%", "%", "%"]
        limits = ["Limit", "0.01", "0.01", "0.01"]
        method = ["Method", "XRF", "XRF", "XRF"]
        header = ["Analyte Symbol", "SiO2", "Fe2O3(T)", "MgO"]
        rows = [[""] * 1, header, units, limits, method] + data
        src = tmp_path / "custom.xlsx"
        pd.DataFrame(rows).to_excel(src, index=False, header=False)
        df, *_ = read_actlabs(str(src), skiprows=1)
        assert df.index.name == "Sample"


class TestReadBureauVeritas:
    def _write_bv(self, tmp_path, filename, data_rows):
        """Write a Bureau Veritas-style Excel file.

        Expected by read_bureau_veritas(skiprows=9):
          Row 0-8: skipped
          Row 9:   header (anything, becomes DataFrame column names)
          Row 10:  method info
          Row 11:  column names (e.g. '', '', 'SiO2', 'MgO')
          Row 12:  units
          Row 13:  limits
          Row 14+: data
        """
        skip_rows = [[""] * 4] * 9
        header_row = [["hdr", "", "", ""]]
        rows = skip_rows + header_row + data_rows
        src = tmp_path / filename
        pd.DataFrame(rows).to_excel(src, index=False, header=False)
        return src

    def test_parse_bureau_veritas(self, tmp_path):
        from petropandas.io import read_bureau_veritas

        data = [
            ["", "", "XRF", "XRF"],
            ["", "", "SiO2", "MgO"],
            ["", "", "%", "%"],
            ["", "", "0.01", "0.01"],
            ["S1", "Rock Pulp", 45.0, 5.0],
        ]
        src = self._write_bv(tmp_path, "bv.xlsx", data)
        res, units, limits, method = read_bureau_veritas(str(src))
        assert not res.empty
        assert "SiO2" in res.columns
        assert res["SiO2"].iloc[0] == 45.0

    def test_detection_limit_replaced_with_nan(self, tmp_path):
        from petropandas.io import read_bureau_veritas

        data = [
            ["", "", "XRF", "XRF"],
            ["", "", "SiO2", "MgO"],
            ["", "", "%", "%"],
            ["", "", "0.01", "0.01"],
            ["S1", "Rock Pulp", "< 0.01", 5.0],
        ]
        src = self._write_bv(tmp_path, "bv_dl.xlsx", data)
        res, _, _, _ = read_bureau_veritas(str(src))
        assert not res.empty
        assert np.isnan(res["SiO2"].iloc[0])

# fmt: off
REE = ["La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Sc", "Y"]

REE_PLOT = ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]

ISOPLOT = [
    "07/35", "06/38", "38/06", "07/06", "04/38", "04/06", "04/07", "08/32", "32/38", "08/06", "35/07", "08/07",
    "07/35_Err", "06/38_Err", "38/06_Err", "07/06_Err", "04/38_Err", "04/06_Err", "04/07_Err", "08/32_Err", "32/38_Err", "08/06_Err", "35/07_Err", "08/07_Err",
    "RhoXY", "RhoXZ", "RhoYZ"
]

ISOPLOT_FORMATS = {
    1: ["07/35", "07/35_Err", "06/38", "06/38_Err", "RhoXY"],
    2: ["38/06", "38/06_Err", "07/06", "07/06_Err", "RhoXY"],
    3: ["07/35", "07/35_Err", "06/38", "06/38_Err", "07/06", "07/06_Err", "RhoXY", "RhoYZ"],
}

AGECOLS = ["Age75", "Age75_Err", "Age68", "Age68_Err", "Age76", "Age76_Err", "BestAge", "BestAge_Err", "Disc"]
# fmt: on
COLNAMES = {
    "SB": {
        "238U/206Pb": "38/06",
        "238U/206Pb2s": "38/06_Err",
        "207Pb/206Pb": "07/06",
        "207Pb/206Pb2s": "07/06_Err",
        "208Pb/232Th": "08/32",
        "208Pb/232Th2s": "08/32_Err",
        "rho": "RhoXY",
        "U238_ppm_mean": "U",
        "Th232_ppm_mean": "Th",
        "Si28_ppm_mean": "Si",
        "Ca44_ppm_mean": "Ca",
        "Sr88_ppm_mean": "Sr",
        "Y89_ppm_mean": "Y",
        "La139_ppm_mean": "La",
        "Ce140_ppm_mean": "Ce",
        "Pr141_ppm_mean": "Pr",
        "Nd146_ppm_mean": "Nd",
        "Sm147_ppm_mean": "Sm",
        "Eu153_ppm_mean": "Eu",
        "Gd157_ppm_mean": "Gd",
        "Tb159_ppm_mean": "Tb",
        "Dy163_ppm_mean": "Dy",
        "Ho165_ppm_mean": "Ho",
        "Er166_ppm_mean": "Er",
        "Tm169_ppm_mean": "Tm",
        "Yb172_ppm_mean": "Yb",
        "Lu175_ppm_mean": "Lu",
    }
}

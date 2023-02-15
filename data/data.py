CLAYFF_CHARGES_DICT = {
        "st": 4,
        "at": 3,
        "fet": 3,
        "ao": 3,
        "feo": 3,
        "fe2": 2,
        "mgo": 2,
        "lio": 1,
        "cao": 2,
    }

CLAY_DF_INDEX =     exp_index = pd.MultiIndex(
        levels=[
            ["T", "O", "C", "I"],
            [
                "st",
                "at",
                "fet",
                "fe_tot",
                "feo",
                "ao",
                "fe2",
                "mgo",
                "lio",
                "cao",
                "T",
                "O",
                "tot",
                "Ca",
                "Mg",
                "K",
                "Na",
                "Cl"
            ],
        ],
        codes=[
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        ],
        names=["sheet", "element"],
    )

from __future__ import annotations

import sys
from functools import partialmethod, cached_property
import warnings
import logging
import re

import numpy as np
import pandas as pd
from typing import Union, List, Dict

from ClayCode.config.classes import File, Dir, ITPFile

warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(File(__file__).stem)

logger.setLevel(logging.DEBUG)

__all__ = ['ClayComposition']




class UnitCell(ITPFile):
    @property
    def idx(self):
        return self.stem[2:]

    @property
    def clay_type(self):
        return self.parent.name

    @property
    def uc_stem(self):
        return self.stem[:2]

    @property
    def atoms(self):
        return self.data


class UCData(Dir):
    _suffices = [".gro", ".itp"]
    _sheet_grouper = pd.Grouper(level="sheet", sort=False)

    def __init__(self, path: Dir, uc_stem=None, ff=None):
        from ClayCode.config.classes import ForceField
        if uc_stem is None:
            self.uc_stem: str = self.name[-2:]
        else:
            self.uc_stem: str = uc_stem
        self.ff = ForceField(ff)
        self.uc_idxs = list(map(lambda x: str(x[-2:]), self.available))
        self.atomtypes = self.ff["atomtypes"].df
        self.__full_df = None
        self.__df = None
        self.__get_full_df()
        self.__get_df()
        self.__atomic_charges = None

    @property
    def full_df(self) -> pd.DataFrame:
        return self.__full_df

    @property
    def df(self) -> pd.DataFrame:
        return self.__df.loc[:, self.uc_idxs]

    def __get_full_df(self):
        idx = self.atomtypes.iloc[:, 0]
        cols = [*self.uc_idxs, "charge", "sheet"]
        self.__full_df = pd.DataFrame(index=idx, columns=cols)
        self.__full_df["charge"].update(self.atomtypes.set_index("at-type")["charge"])
        self.__get_df_sheet_annotations()
        self.__full_df["sheet"].fillna("X", inplace=True)
        self.__full_df.fillna(0, inplace=True)
        for uc in self.uc_list:
            atoms = uc["atoms"].df
            self.__full_df[f"{uc.idx}"].update(atoms.value_counts("at-type"))
        self.__full_df.set_index("sheet", append=True, inplace=True)
        self.__full_df.sort_index(inplace=True, level=1, sort_remaining=True)
        self.__full_df.index = self.__full_df.index.reorder_levels(["sheet", "at-type"])

    def __get_df_sheet_annotations(self):
        old_index = self.__full_df.index
        regex_dict = {"T": r"[a-z]+t", "O": r"[a-z]*[a-gi-z][o2]", "C": "charge"}
        index_extension_list = []
        for key in regex_dict.keys():
            for element in old_index:
                match = re.fullmatch(regex_dict[key], element)
                if match is not None:
                    index_extension_list.append((key, match.group(0)))
        new_index = pd.MultiIndex.from_tuples(index_extension_list)
        new_index = new_index.to_frame().set_index(1)
        self.__full_df["sheet"].update(new_index[0])

    def __get_df(self):
        self.__df = self.full_df.reset_index("at-type").filter(
            regex=r"^(?![X].*)", axis=0
        )
        self.__df = self.__df.reset_index().set_index(["sheet", "at-type"])

    @cached_property
    def uc_list(self) -> List[UnitCell]:
        uc_list = [UnitCell(itp) for itp in self.itp_filelist]
        return uc_list

    @cached_property
    def occupancies(self) -> Dict[str, int]:
        return self._get_occupancies(self.df)
        # occ = self.df.groupby('sheet').sum().aggregate("unique",
        #                                                axis='columns')
        # idx = self.df.index.get_level_values('sheet').unique()
        # occ_dict = dict(zip(idx, occ))
        # return occ_dict

    @cached_property
    def tot_charge(self) -> pd.Series:
        charge = self.full_df.apply(lambda x: x * self.full_df["charge"], raw=True)
        total_charge = charge.loc[:, self.uc_idxs].sum().round(2).convert_dtypes()
        total_charge.name = "charge"
        return total_charge

    @cached_property
    def uc_composition(self) -> pd.DataFrame:
        return self.full_df.reindex(self.atomtypes, fill_value=0).filter(
            regex=r"^(?![oOhH].*)", axis=0
        )

    @cached_property
    def oxidation_numbers(self) -> Dict[str, int]:
        """Get dictionary of T and O element oxidation numbers for 0 charge sheet"""
        ox_dict = self._get_oxidation_numbers(self.df, self.occupancies, self.tot_charge)[1]
        return dict(zip(ox_dict.keys(), list(map(lambda x: int(x), ox_dict.values()))))
        # import yaml
        # from ClayCode import UCS
        # with open(UCS / 'clay_charges.yaml', 'r') as file:
        #     ox_dict: dict = yaml.safe_load(file)
        # ox_df: pd.DataFrame = self.df.copy()
        # ox_df = ox_df.loc[~(ox_df == 0).all(1), :]
        # ox_df = ox_df.loc[:, self.tot_charge == 0]
        # at_types: pd.DataFrame = ox_df.index.get_level_values('at-type').to_frame()
        # at_types.index = ox_df.index
        # at_types = at_types.applymap(lambda x: ox_dict[x])
        # ox: np.ndarray = ox_df.apply(lambda x: x * at_types['at-type']).groupby('sheet').sum().aggregate('unique',
        #                                                                                                  axis='columns')
        # idx: pd.Index = self.df.index.get_level_values('sheet').unique()
        # ox_dict: dict = dict(zip(idx, ox))
        # ox_dict: dict = dict(map(lambda x: (x, ox_dict[x] // self.occupancies[x]), ox_dict.keys()))
        # return ox_dict

    @cached_property
    def idxs(self) -> np.array:
        return self.full_df.columns

    def check(self) -> None:
        if not self.is_dir():
            raise FileNotFoundError(f"{self.name} is not a directory.")

    @cached_property
    def available(self) -> List[ITPFile]:
        return self.itp_filelist.extract_fstems()

    def __str__(self):
        return f'{self.__class__.__name__}({self.name!r})'

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name!r})'


    @staticmethod
    def _get_occupancies(df) -> Dict[str, int]:
        try:
            occ = df.sort_index(level='sheet', sort_remaining=True).groupby('sheet').sum().aggregate("unique",
                                                      axis='columns')
        except ValueError:
            occ = df.sort_index(level='sheet', sort_remaining=True).groupby('sheet').sum().aggregate("unique")
        idx = df.sort_index(level='sheet', sort_remaining=True).index.get_level_values('sheet').unique()
        occ_dict = dict(zip(idx, occ))
        return occ_dict


    @property
    def atomic_charges(self):
        return self._get_oxidation_numbers(self.df,
                                           self.occupancies,
                                           self.tot_charge,
                                           )[0]


    @staticmethod
    def _get_oxidation_numbers(df, occupancies, tot_charge=None, sum_dict=True) -> Dict[str, int]:
        import yaml
        from ClayCode import UCS
        with open(UCS / 'clay_charges.yaml', 'r') as file:
            ox_dict: dict = yaml.safe_load(file)
        # df = df.loc[['T','O']]
        ox_df: pd.DataFrame = df.copy()
        ox_df.sort_index(level='sheet', sort_remaining=True, inplace=True)
        try:
            ox_df = ox_df.loc[~(ox_df == 0).all(1), :]
        except ValueError:
            ox_df = ox_df.loc[~(ox_df == 0)]
        if tot_charge is not None:
            ox_df = ox_df.loc[:, tot_charge == 0]
        at_types: pd.DataFrame = ox_df.index.get_level_values('at-type').to_frame()
        at_types.index = ox_df.index
        try:
            at_types.drop(('O', 'fe_tot'), inplace=True)
        except KeyError:
            pass
        at_types = at_types.applymap(lambda x: ox_dict[x])
        if type(ox_df) == pd.DataFrame:
            ox: pd.DataFrame = ox_df.apply(lambda x: x * at_types['at-type'])
        else:
            ox: pd.DataFrame = at_types.apply(lambda x: x * ox_df)
        if sum_dict is True:
            ox: np.ndarray = ox.groupby('sheet').sum().aggregate('unique',
                                                                 axis='columns')
            idx: pd.Index = ox_df.sort_index().index.get_level_values('sheet').unique()
            ox_dict: dict = dict(zip(idx, ox))
            ox_dict: dict = dict(map(lambda x: (x, ox_dict[x] / occupancies[x]), occupancies.keys()))
        else:
            ox_dict = ox.groupby('sheet').apply(lambda x: x / occupancies[x.name])
        return at_types, ox_dict

class ClayComposition:

    sheet_grouper = pd.Grouper(level="sheet", sort=False)
    def __init__(self, name, csv_file: Union[str, File], uc_data: UCData):
        self.name: str = name
        self.match_file: File = File(csv_file, check=True)
        self.uc_data: UCData = uc_data
        self.uc_df = self.uc_data.df
        match_df: pd.DataFrame = pd.read_csv(csv_file).fillna(method="ffill")
        match_cols = match_df.columns.values
        match_cols[:2] = self.uc_df.index.names
        match_df.columns = match_cols
        match_df.set_index(self.uc_df.index.names, inplace=True)
        match_idx = (self.uc_df.index.to_flat_index()).union(match_df.index.to_flat_index())
        match_idx = pd.MultiIndex.from_tuples(match_idx, names=self.uc_df.index.names)
        self.match_df = match_df.reindex(match_idx)
        self.match_df = self.match_df.loc[:, self.name].dropna()
        self.correct_occupancies()
        self.match_df = self.match_df.reindex(match_idx)
        self.charge_df = None
        self.sheet_charges
        # self.get_match_charges()
        ...
        # uc_df.sort_index(inplace=True, sort_remaining=True)
        # uc_df['match'] = np.NaN
        # uc_df['match']

    # def get_match_charges(self):
        # ox_numbers = self.oxidation_states
        # tot_charge = self.match_df.loc[['C', 'tot_chg']]
        # uc_charges = UCData._get_oxidation_numbers(self.match_df, self.occupancies)
        # charge_df = self.match_df.loc[['T', 'O'], :]
        # atomic_charges = self.uc_data.atomic_charges
        # atomic_charges['match_charge'] = atomic_charges.apply(lambda x: x * charge_df)
        # charge_df['charge'] = charge_df.groupby('sheet').apply(lambda x: x * self.uc_data.).groupby('sheet').sum().aggregate('unique',
        # ...                                                                                                         axis='columns')
        # ...
    @property
    def sheet_charges(self):
        ox_states = UCData._get_oxidation_numbers(self.match_df.loc[['T', 'O']], self.occupancies)[1]
        charge_delta = dict(map(lambda x: (x, self.uc_data.oxidation_numbers[x] - ox_states[x]), ox_states.keys()))
        sheet_df = self.match_df.copy()

    @property
    def oxidation_states(self):
        return UCData._get_oxidation_numbers(self.match_df.loc[['T', 'O']], self.occupancies, sum_dict=False)[1]

    @property
    def atom_types(self):
        return UCData._get_oxidation_numbers(self.match_df.loc[['T', 'O']], self.occupancies, sum_dict=False)[0]

    @property
    def occupancies(self):
        return UCData._get_occupancies(self.match_df.loc[['T', 'O']].dropna())



    def correct_occupancies(self, idx_sel=["T", "O"]):
        correct_uc_occ: pd.Series = pd.Series(self.uc_data.occupancies)
        input_uc_occ: pd.Series = pd.Series(self.occupancies)
        check_occ: pd.Series = input_uc_occ - correct_uc_occ
        check_occ.dropna(inplace=True)
        for sheet, occ in check_occ.iteritems():
            logger.info(f"Found {sheet!r} sheet occupancies of {input_uc_occ[sheet]:.2f}/{correct_uc_occ[sheet]:.2f} ({occ:+.2f})")
        # exp_occ.apply(lambda x: print(f"{x.name}: {x.sum()}"))
        sheet_df: pd.Series = self.match_df.loc[['T', 'O'], :].copy()
        sheet_df = sheet_df.loc[sheet_df != 0]
        if check_occ.values.any() != 0:
            logger.info("Adjusting values to match expected occupancies:")
            sheet_df = sheet_df.groupby('sheet').apply(lambda x: np.round(x - check_occ.at[x.name] / x.count(), 5))
            accept = None
            old_composition = self.match_df.copy()
            while accept != 'y':
                self.match_df.update(sheet_df)
                new_occ = pd.Series(self.occupancies)
                new_check_df: pd.Series = new_occ - correct_uc_occ
                new_check_df.dropna(inplace=True)
                assert (new_check_df == 0).all(), f'New occupancies are non-integer!'
                print('old occupancies -> new occupancies per unit cell:\n\t'
                      'sheet - atom type : occupancies  (difference)')
                for idx, occ in sheet_df.iteritems():
                    sheet, atom = idx
                    old_val = old_composition[idx]
                    print(
                        f"\t{sheet!r:5} - {atom!r:^10}: {old_val:2.2f} -> {occ:2.2f} ({occ - old_val:+2.2f})"# sheet occupancies of {new_occ[sheet]:.2f}/{correct_uc_occ[sheet]:.2f} ({occ:+.2f})")
                    )
                accept = input('\nAccept new composition? [y/n] (exit with c)\n').lower()
                if accept == 'n':
                    sheet_df: pd.Series = old_composition.copy()
                    sheet_df = sheet_df.loc[['T', 'O'], :]
                    sheet_df = sheet_df.loc[sheet_df != 0]
                    for k, v in check_occ.items():
                        if v != 0:
                            for atom, occ in sheet_df.loc[k, :].iteritems():
                                sheet_df.loc[k, atom] = float(input(f'Enter new value for {k!r} - {atom!r}: ({occ:2.2f}) -> '))
                elif accept == 'c':
                    sys.exit(0)
            # for idx, val in self.match_df.iter
            logger.info('Will use the following clay composition:')
            for idx, occ in sheet_df.iteritems():
                sheet, atom = idx
                logger.info(f"\t{sheet!r} - {atom!r:^10}: {occ:2.2f}")
            # exp_group = exp_occ.groupby(self.sheet_grouper)
            # exp_group.apply(lambda x: print(f"{x.name}: {x.sum()}"))
            # self.df.update(exp_occ)


    correct_t_occupancies = partialmethod(correct_occupancies, idx_sel=["T"])
    correct_o_occupancies = partialmethod(correct_occupancies, idx_sel=["O"])

class ExpComposition:
    # print(ion_charges.head(5))
    charges_mapping = {
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
    exp_index = pd.MultiIndex(
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
            ],
        ],
        codes=[
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        ],
        names=["sheet", "element"],
    )
    sheet_grouper = pd.Grouper(level="sheet", sort=False)

    def __init__(self, data_dir, clay_comp, sysname, ions_ff, clay_ff, uc_type, n_cells):
        ion_charges = ions_ff.atomtypes
        self.target_comp = File(clay_comp).resolve()
        self.data_dir = data_dir
        self.clay_type=uc_type
        self.sysname = sysname
        self.n_cells = n_cells
        self.df = pd.DataFrame(
            index=self.__class__.exp_index, columns=["charge", sysname, "el_chg"], dtype="float"
        )
        self.add_element_charge_info()
        self.add_exp_data()
        self.charge_occ_info = self.get_uc_occ_charge_info()
        self.correct_occupancies()
        self.add_uc_charge_info()
        pd.to_numeric(self.df.loc[:, sysname])
        self.check_exp_charge_info()
        self.round_sheet_df_occ()
        self.get_fe2_feo_values()
        self.add_uc_charge_info()
        self.check_uc_charge()
        self.df.drop(index="fe_tot", level="element", inplace=True)
        self.df.loc[["T", "O"]] = self.df.loc[["T", "O"]].fillna(0)

    @staticmethod
    def get_df_from_csv(fname, index):
        df = pd.read_csv(fname).fillna(method="ffill").set_index(index)
        return df

    def add_element_charge_info(self):
        element_charges = pd.DataFrame.from_dict(
            self.__class__.charges_mapping, orient="index", columns=["charge"]
        )
        self.df.reset_index(level=0, inplace=True)
        self.df.update(element_charges)
        self.df = self.df.reset_index().set_index(["sheet", "element"])

    def add_exp_data(self):
        exp_data = self.get_df_from_csv(self.target_comp, ["sheet", "element"])
        self.df.update(exp_data)

    def get_occupations(self):
        return self.uc_df.iloc[:, :-1].groupby(['sheet']).sum().aggregate("unique",
                                                                          axis='columns')

    def get_charges(self):
        charges = self.uc_df.iloc[:, -1].apply

    def get_uc_occ_charge_info(self):
        charge_occ_info = self.get_df_from_csv(self.data_dir / "charge_occ.csv", ["sheet", "value"])
        charge_occ_info = charge_occ_info.xs(self.clay_type, level=1)
        return charge_occ_info

    def get_tot_charge(self, x):
        return self.charge_occ_info.loc[x.name].product()

    def get_occ_info(self, x):
        return self.charge_occ_info.loc[x.name, "occ"]

    def get_chg_info(self, x):
        # print('get_chg_info')
        chg = x[self.sysname] * x["charge"] - x[self.sysname] / self.get_occ_info(
            x
        ) * self.get_tot_charge(x)
        return chg

    def check_exp_charge_info(self):
        if self.df.at[("C", "tot"), self.sysname] != np.sum(
            self.df.loc[("C", ["T", "O"]), self.sysname]
        ):
            chg_t = self.df["el_chg"].groupby(self.sheet_grouper).sum().loc["T"]
            self.df.loc[("C", "O"), self.sysname] = (
                self.df.at[("C", "tot"), self.sysname] - chg_t
            )
            self.df.loc[("C", "T"), self.sysname] = chg_t
        else:
            print(
                self.df.at[("C", "tot"), self.sysname],
                np.sum(self.df.loc[("C", ["T", "O"]), self.sysname]),
            )

    def add_uc_charge_info(self):
        el_chg = self.df.loc[["T", "O"]]
        el_chg = el_chg.groupby(self.sheet_grouper).apply(
            lambda x: self.get_chg_info(x)
        )
        el_chg.reset_index(level=1, drop=True, inplace=True)
        el_chg.name = "el_chg"
        self.df.update(el_chg)

    def get_fe2_feo_values(self):
        print("fe2_fe3")
        chg_o = self.df.at[("C", "O"), self.sysname]
        print(chg_o)
        chg_not_fe = (
            self.df[~self.df.index.isin(["fe2", "feo"], level=1)]
            .loc["O", "el_chg"]
            .sum()
        )
        self.df.loc[("O", "fe2"), self.sysname] = -chg_o + chg_not_fe
        x_fe_tot = self.df.at[("O", "fe_tot"), self.sysname]
        self.df.loc[("O", "feo"), self.sysname] = (
            x_fe_tot - self.df.loc[("O", "fe2"), self.sysname]
        )

    def check_uc_charge(self):
        chg_df = self.df.loc[["T", "O"], "el_chg"]
        chg_df = chg_df.groupby(self.sheet_grouper).sum()
        chg_df.name = "el_chg"
        chg_df.index.name = "element"
        chg_df.index = pd.MultiIndex.from_product(
            [["C"], chg_df.index], names=self.df.index.names
        )
        self.df.update(chg_df)
        self.df.loc[("C", "tot")] = chg_df.sum()
        check = self.df.xs(("C", "tot")).loc[[self.sysname, "el_chg"]]
        if not check[self.sysname] == check["el_chg"]:
            raise ValueError(
                f"Specified unit cell charge of {self.sysname} "
                f"does not correspond to calculated unit cell "
                "charge."
            )

    def correct_occupancies(self, idx_sel=["T", "O"]):
        exp_occ = self.df.loc[[*idx_sel], self.sysname].dropna()
        print(self.df, exp_occ)
        exp_occ = exp_occ.loc[exp_occ != 0]
        exp_occ = exp_occ.groupby(self.sheet_grouper)
        check_occ = self.charge_occ_info["occ"] - exp_occ.sum()
        print(exp_occ, check_occ)
        print("Sheet occupancies found:")
        exp_occ.apply(lambda x: print(f"{x.name}: {x.sum()}"))
        if check_occ.values.any() != 0:
            print("Adjusting values to match expected values:")
            exp_occ = exp_occ.apply(lambda x: x + check_occ.at[x.name] / x.count())
            exp_group = exp_occ.groupby(self.sheet_grouper)
            exp_group.apply(lambda x: print(f"{x.name}: {x.sum()}"))
            self.df.update(exp_occ)

    correct_t_occupancies = partialmethod(correct_occupancies, idx_sel=["T"])
    correct_o_occupancies = partialmethod(correct_occupancies, idx_sel=["O"])

    def round_sheet_df_occ(self):
        self.df.loc[["T", "O", "C"]] = self.df.loc[["T", "O", "C"]].round(2)

    def write_occ_df(self, outpath):
        if outpath.is_dir():
            self.df.to_csv(outpath / f"{self.sysname}_exp_df.csv")
        else:
            raise NotADirectoryError(f"{outpath} does not exist.")

# e = ExpComposition(CLAY_UNITS_DIR, 'exp_4.csv', 'NAu-1', ions_ff='AmberIons', clay_ff='ClayFF_Fe',
#                    uc_type='smec-dio', n_cells=35)

# print(e.ion_charges, e.clay_charges)
# class ExpComposition:
#     ion_charges = ForceField(
#         selection={"Ion_test": ["ffnonbonded", "ions"]}
#     ).atomic_charges
#     print(ion_charges.head(5))
#     charges_mapping = {
#         "st": 4,
#         "at": 3,
#         "fet": 3,
#         "ao": 3,
#         "feo": 3,
#         "fe2": 2,
#         "mgo": 2,
#         "lio": 1,
#         "cao": 2,
#     }
#     exp_index = pd.MultiIndex(
#         levels=[
#             ["T", "O", "C", "I"],
#             [
#                 "st",
#                 "at",
#                 "fet",
#                 "fe_tot",
#                 "feo",
#                 "ao",
#                 "fe2",
#                 "mgo",
#                 "lio",
#                 "cao",
#                 "T",
#                 "O",
#                 "tot",
#                 "Ca",
#                 "Mg",
#                 "K",
#                 "Na",
#             ],
#         ],
#         codes=[
#             [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
#             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
#         ],
#         names=["sheet", "element"],
#     )
#     sheet_grouper = pd.Grouper(level="sheet", sort=False)
#
#     def __init__(self, sysname, exp_index=exp_index):
#         self._df = pd.DataFrame(
#             index=exp_index, columns=["charge", sysname, "el_chg"], dtype="float"
#         )
#         self.add_element_charge_info()
#         self.add_exp_data()
#         self.charge_occ_info = self.get_uc_occ_charge_info()
#         self.correct_occupancies()
#         self.add_uc_charge_info()
#         pd.to_numeric(self._df.loc[:, sysname])
#         self.check_exp_charge_info()
#         self.round_sheet_df_occ()
#         self.get_fe2_feo_values()
#         self.add_uc_charge_info()
#         self.check_uc_charge()
#         self._df.drop(index="fe_tot", level="element", inplace=True)
#         self._df.loc[["T", "O"]] = self._df.loc[["T", "O"]].fillna(0)
#
#     @staticmethod
#     def get_df_from_csv(fname, index):
#         _df = pd.read_csv(pp.DATA_PATH / fname).fillna(method="ffill").set_index(index)
#         return _df
#
#     def add_element_charge_info(self):
#         element_charges = pd.DataFrame.from_dict(
#             ExpComposition.charges_mapping, orient="index", columns=["charge"]
#         )
#         self._df.reset_index(level=0, inplace=True)
#         self._df.update(element_charges)
#         self._df = self._df.reset_index().set_index(["sheet", "element"])
#
#     def add_exp_data(self):
#         exp_data = self.get_df_from_csv("exp_4.csv", ["sheet", "element"])
#         self._df.update(exp_data)
#
#     def get_uc_occ_charge_info(self):
#         charge_occ_info = self.get_df_from_csv("charge_occ.csv", ["sheet", "value"])
#         charge_occ_info = charge_occ_info.xs(bp.UC_FOLDER, level=1)
#         return charge_occ_info
#
#     def get_tot_charge(self, x):
#         return self.charge_occ_info.loc[x.name].product()
#
#     def get_occ_info(self, x):
#         return self.charge_occ_info.loc[x.name, "occ"]
#
#     def get_chg_info(self, x):
#         # print('get_chg_info')
#         chg = x[self.x_el] * x["charge"] - x[self.x_el] / self.get_occ_info(
#             x
#         ) * self.get_tot_charge(x)
#         return chg
#
#     def check_exp_charge_info(self):
#         if self._df.at[("C", "tot"), self.x_el] != np.sum(
#             self._df.loc[("C", ["T", "O"]), self.x_el]
#         ):
#             chg_t = self._df["el_chg"].groupby(self.sheet_grouper).sum().loc["T"]
#             self._df.loc[("C", "O"), self.x_el] = (
#                 self._df.at[("C", "tot"), self.x_el] - chg_t
#             )
#             self._df.loc[("C", "T"), self.x_el] = chg_t
#         else:
#             print(
#                 self._df.at[("C", "tot"), self.x_el],
#                 np.sum(self._df.loc[("C", ["T", "O"]), self.x_el]),
#             )
#
#     def add_uc_charge_info(self):
#         el_chg = self._df.loc[["T", "O"]]
#         el_chg = el_chg.groupby(self.sheet_grouper).apply(
#             lambda x: self.get_chg_info(x)
#         )
#         el_chg.reset_index(level=1, drop=True, inplace=True)
#         el_chg.name = "el_chg"
#         self._df.update(el_chg)
#
#     def get_fe2_feo_values(self):
#         print("fe2_fe3")
#         chg_o = self._df.at[("C", "O"), self.x_el]
#         print(chg_o)
#         chg_not_fe = (
#             self._df[~self._df.index.isin(["fe2", "feo"], level=1)]
#             .loc["O", "el_chg"]
#             .sum()
#         )
#         self._df.loc[("O", "fe2"), self.x_el] = -chg_o + chg_not_fe
#         x_fe_tot = self._df.at[("O", "fe_tot"), self.x_el]
#         self._df.loc[("O", "feo"), self.x_el] = (
#             x_fe_tot - self._df.loc[("O", "fe2"), self.x_el]
#         )
#
#     def check_uc_charge(self):
#         chg_df = self._df.loc[["T", "O"], "el_chg"]
#         chg_df = chg_df.groupby(self.sheet_grouper).sum()
#         chg_df.name = "el_chg"
#         chg_df.index.name = "element"
#         chg_df.index = pd.MultiIndex.from_product(
#             [["C"], chg_df.index], names=self._df.index.names
#         )
#         self._df.update(chg_df)
#         self._df.loc[("C", "tot")] = chg_df.sum()
#         check = self._df.xs(("C", "tot")).loc[[self.x_el, "el_chg"]]
#         if not check[self.x_el] == check["el_chg"]:
#             raise ValueError(
#                 method"Specified unit cell charge of {self.x_el} "
#                 method"does not correspond to calculated unit cell "
#                 "charge."
#             )
#
#     def correct_occupancies(self, idx_sel=["T", "O"]):
#         exp_occ = self._df.loc[[*idx_sel], self.x_el].dropna()
#         exp_occ = exp_occ.loc[exp_occ != 0]
#         exp_occ = exp_occ.groupby(self.sheet_grouper)
#         check_occ = self.charge_occ_info["occ"] - exp_occ.sum()
#         print("Sheet occupancies found:")
#         exp_occ.apply(lambda x: print(method"{x.name}: {x.sum()}"))
#         if check_occ.values.any() != 0:
#             print("Adjusting values to match expected values:")
#             exp_occ = exp_occ.apply(lambda x: x + check_occ.at[x.name] / x.count())
#             exp_group = exp_occ.groupby(self.sheet_grouper)
#             exp_group.apply(lambda x: print(method"{x.name}: {x.sum()}"))
#             self._df.update(exp_occ)
#
#     correct_t_occupancies = partialmethod(correct_occupancies, idx_sel=["T"])
#     correct_o_occupancies = partialmethod(correct_occupancies, idx_sel=["O"])
#
#     def round_sheet_df_occ(self):
#         self._df.loc[["T", "O", "C"]] = self._df.loc[["T", "O", "C"]].round(2)
#
#     def write_occ_df(self, outpath):
#         if outpath.is_dir():
#             self._df.to_csv(outpath / method"{self.x_el}_exp_df.csv")
#         else:
#             raise NotADirectoryError(method"{outpath} does not exist.")

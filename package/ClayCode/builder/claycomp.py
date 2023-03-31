from __future__ import annotations

import os
import shutil
import sys
import tempfile
from functools import partialmethod, cached_property, lru_cache, singledispatchmethod, cache
import warnings
import logging
import re
from pathlib import PosixPath
import pickle as pkl
from copy import copy
import itertools

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Literal

from ClayCode.config.classes import File, Dir, ITPFile, PathFactory
from ClayCode.core.lib import get_ion_charges

warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(File(__file__).stem)

logger.setLevel(logging.DEBUG)

__all__ = ['TargetClayComposition']

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
        self.ff: ForceField = ForceField(ff)
        self.__uc_idxs: list = list(map(lambda x: str(x[-2:]), self.available))
        self.uc_idxs = self.__uc_idxs.copy()
        self.atomtypes: pd.DataFrame = self.ff["atomtypes"].df
        self.__full_df: pd.DataFrame = None
        self.__df: pd.DataFrame = None
        self.__get_full_df()
        self.__get_df()
        self.__atomic_charges = None
        self.__uc_groups = self.get_uc_groups()

    def get_uc_groups(self):
        box_dims = {}
        uc_groups = {}
        n_groups = 0
        extract_id = lambda file: file.stem[-2:]
        for uc in self.gro_filelist:
            uc_dimensions = uc.universe.dimensions
            dim_str = ''.join(uc_dimensions.round(3).astype(str))
            if dim_str not in box_dims.keys():
                box_dims[dim_str] = n_groups
                uc_groups[n_groups] = [extract_id(uc)]
                n_groups += 1
            else:
                uc_groups[box_dims[dim_str]].append(extract_id(uc))
        return uc_groups

    def group_iter(self) -> Tuple[int, List[str]]:
        for group_id in sorted(self.__uc_groups.keys()):
            yield group_id, self.__uc_groups[group_id]

    @property
    def uc_groups(self):
        return self.__uc_groups

    def select_group(self, group_id: int):
        try:
            self.uc_idxs = self.uc_groups[group_id]
            logger.info(f'Selected unit cells {self.uc_idxs}')
        except KeyError as e:
            e(f'{group_id} is an invalid group id!')

    def select_ucs(self, uc_ids: List[str]):
        assert np.isin(uc_ids, self.uc_idxs).all(), 'Invalid selection'
        self.uc_idxs = uc_ids

    def reset_uc_selection(self):
        self.uc_idxs = self.__uc_idxs.copy()

    @property
    def n_groups(self):
        return len(self.__uc_groups.keys())

    @property
    def full_df(self) -> pd.DataFrame:
        return self.__full_df.sort_index(ascending=False,
                                                         level='sheet',
                                                         sort_remaining=True)

    @property
    def df(self) -> pd.DataFrame:
        return self.__df.loc[:, self.uc_idxs].sort_index(ascending=False,
                                                         level='sheet',
                                                         sort_remaining=True)

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
        ox_dict = self._get_oxidation_numbers(self.occupancies, self.df, self.tot_charge)[1]
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
        return self._get_oxidation_numbers(self.occupancies, self.df, self.tot_charge)[0]

    @singledispatchmethod
    @staticmethod
    def _get_oxidation_numbers(occupancies,
                                df: Union[pd.DataFrame, pd.Series],
                               tot_charge: Optional = None,
                               sum_dict: bool = True) -> Dict[str, int]:
        pass

    @_get_oxidation_numbers.register(dict)
    @staticmethod
    def _(occupancies: Dict[str, int], df: Union[pd.DataFrame, pd.Series],
                               tot_charge: Optional = None, sum_dict: bool = True) -> Dict[str, int]:
        """Get oxidation numbers from unit cell composition and occupancies"""
        ox_dict = UCData._get_ox_dict()
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

    @_get_oxidation_numbers.register(float)
    @_get_oxidation_numbers.register(int)
    @staticmethod
    def _(occupancies: Union[float, int], df: Union[pd.DataFrame, pd.Series],
                               tot_charge: Optional = None, sum_dict: bool = True) -> int:
        """Get oxidation numbers from unit cell composition and occupancies"""
        # df = df.loc[['T','O']]
        ox_dict = UCData._get_ox_dict()
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
            at_types.drop(('fe_tot'), inplace=True)
        except KeyError:
            pass
        at_types = at_types.applymap(lambda x: ox_dict[x])
        if type(ox_df) == pd.DataFrame:
            ox: pd.DataFrame = ox_df.apply(lambda x: x * at_types['at-type'])
        else:
            ox: pd.DataFrame = at_types.apply(lambda x: x * ox_df)
        if sum_dict is True:
            ox: np.ndarray = ox.sum().aggregate('unique')
            # ox_dict: dict = dict(zip(idx, ox))
            ox_val = ox / occupancies
            # ox_dict: dict = dict(map(lambda x: (x, ox_dict[x] / occupancies[x]), occupancies.keys()))
        else:
            ox_val = ox.apply(lambda x: x / occupancies)
        return at_types, ox_val[0]


    @staticmethod
    @cache
    def _get_ox_dict():
        import yaml
        from ClayCode import UCS
        with open(UCS / 'clay_charges.yaml', 'r') as file:
            ox_dict: dict = yaml.safe_load(file)
        return ox_dict

class TargetClayComposition():

    sheet_grouper = pd.Grouper(level="sheet", sort=False)
    def __init__(self, name, csv_file: Union[str, File], uc_data: UCData):
        self.name: str = name
        self.match_file: File = File(csv_file, check=True)
        self.uc_data: UCData = uc_data
        self.uc_df: pd.DataFrame = self.uc_data.df
        match_df: pd.DataFrame = pd.read_csv(csv_file).fillna(method="ffill")
        match_cols = match_df.columns.values
        match_cols[:2] = self.uc_df.index.names
        match_df.columns = match_cols
        match_df.set_index(self.uc_df.index.names, inplace=True)
        ion_idx = tuple(('I', ion_name) for ion_name in get_ion_charges().keys())
        ion_idx = pd.MultiIndex.from_tuples(ion_idx, names=self.uc_df.index.names)
        match_idx = (self.uc_df.index.to_flat_index()).union(match_df.index.to_flat_index())
        match_idx = (match_idx).union(ion_idx)
        match_idx = pd.MultiIndex.from_tuples(match_idx, names=self.uc_df.index.names)
        self.__df = match_df.reindex(match_idx)
        self.__df = self.__df.loc[:, self.name].dropna()
        self.correct_occupancies()
        self.__df = self.__df.reindex(match_idx)
        # self.charge_df = None
        self.split_fe_occupancies
        self.__ion_df: pd.DataFrame = None
        self.get_ion_numbers()

    @property
    def df(self):
        return self.__df.dropna().sort_index(ascending=False,
                                                         level='sheet',
                                                         sort_remaining=True)

    @property
    def clay_df(self):
        return self.__df.loc[['T', 'O']].sort_index(ascending=False,
                                                         level='sheet',
                                                         sort_remaining=True)

    @property
    def ion_df(self) -> pd.DataFrame:
        return self.__ion_df.sort_index()

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
        # atomic_charges = self._uc_data.atomic_charges
        # atomic_charges['match_charge'] = atomic_charges.apply(lambda x: x * charge_df)
        # charge_df['charge'] = charge_df.groupby('sheet').apply(lambda x: x * self._uc_data.).groupby('sheet').sum().aggregate('unique',
        # ...                                                                                                         axis='columns')
        # ...
    @property
    def split_fe_occupancies(self):
        # charges = self.get_charges(self.match_df.xs('C'))
        o_charge = self.get_o_charge()
        try:
            missing_o = self.__df.xs(('O', 'fe_tot'), level=('sheet', 'at-type')).values[0]#.drop(('O', 'fe_tot')))['O']
            not_fe_occ = self.occupancies['O'] - missing_o
            _, ox_state_not_fe = UCData._get_oxidation_numbers(not_fe_occ, self.__df.xs('O', level='sheet'))
            chg_fe = o_charge + (self.uc_data.oxidation_numbers['O'] - ox_state_not_fe) * not_fe_occ
            self.__df.loc[:, 'fe2'] = - (o_charge - chg_fe)
            self.__df.loc[:, 'feo'] = missing_o - self.__df.at[('O', 'fe2')]
            self.__df.drop(('O', 'fe_tot'), inplace=True)
            # charge_delta = dict(map(lambda x: (x, self._uc_data.oxidation_numbers[x] - ox_states[x]), ox_states.keys()))
            assert self.occupancies['O'] == self.uc_data.occupancies['O'], \
                f'Found incorrect occupancies of {self.occupancies["O"]}, expected {self.uc_data.occupancies["O"]}'
        except KeyError:
            pass
        sheet_df = self.__df.copy()
        logger.info('Splitting total iron content by charge.\n')
        self.__print_df_composition(sheet_df.loc[['T', 'O'], :].dropna())
        accept = None
        while accept not  in ['y', 'n']:
            accept = input('Accept clay composition? [y/n]')
        if accept == 'n':
            self.__abort()
        # logger.info(f'{accept}\n')
        return sheet_df

    @staticmethod
    def __abort(self):
        logger.info('Composition not accepted. Aborting model construction.')
        sys.exit(0)

    def _get_charges(self, key: Literal[Union['tot', 'T', 'O']]) -> float:
        return self.get_charges(self.__df.xs('C'))[key]

    get_total_charge = partialmethod(_get_charges, key='tot')
    get_t_charge = partialmethod(_get_charges, key='T')
    get_o_charge = partialmethod(_get_charges, key='O')


    @staticmethod
    def get_charges(charge_df: pd.Series):
        sheet_charges = charge_df.copy()
        charge_dict = sheet_charges.to_dict()
        # for sheet, charge in charge_df.items():
        #     if not charge_df.loc[sheet].isnan():
        #         charge_dict[sheet] = charge
        tot_charge = sheet_charges.pop('tot')
        if charge_df.hasnans:
            # missing charge specifications
            if tot_charge == np.NAN and len(sheet_charges) != len(sheet_charges.dropna()):
                assert not charge_df.drop('tot').hasnans, \
                    'No charges specified'
            # only T/O charges given
            elif tot_charge == np.NAN:
                charge_dict[tot_charge] = sheet_charges.sum()
            elif len(sheet_charges) != len(sheet_charges.dropna()):
                sheet_charges[sheet_charges.isna] = tot_charge - sheet_charges.dropna()
                charge_dict.update(sheet_charges.to_dict())
        else:
            assert sheet_charges.sum() == tot_charge, 'Sheet charges are different from specified total charge'
        return charge_dict


    @property
    def oxidation_states(self):
        return UCData._get_oxidation_numbers(self.occupancies, self.__df.loc[['T', 'O']], sum_dict=False)[1]

    @property
    def atom_types(self):
        return UCData._get_oxidation_numbers(self.occupancies, self.__df.loc[['T', 'O']], sum_dict=False)[0]

    @property
    def occupancies(self):
        return UCData._get_occupancies(self.__df.loc[['T', 'O']].dropna())



    def correct_occupancies(self, idx_sel=["T", "O"]):
        correct_uc_occ: pd.Series = pd.Series(self.uc_data.occupancies)
        input_uc_occ: pd.Series = pd.Series(self.occupancies)
        check_occ: pd.Series = input_uc_occ - correct_uc_occ
        check_occ.dropna(inplace=True)
        for sheet, occ in check_occ.iteritems():
            logger.info(f"Found {sheet!r} sheet occupancies of {input_uc_occ[sheet]:.2f}/{correct_uc_occ[sheet]:.2f} ({occ:+.2f})")
        # exp_occ.apply(lambda x: print(f"{x.name}: {x.sum()}"))
        sheet_df: pd.Series = self.__df.loc[['T', 'O'], :].copy()
        sheet_df = sheet_df.loc[sheet_df != 0]
        if check_occ.values.any() != 0:
            logger.info("Adjusting values to match expected occupancies:")
            sheet_df = sheet_df.groupby('sheet').apply(lambda x: x - check_occ.at[x.name] / x.count())
            accept = None
            old_composition = self.__df.copy()
            while accept != 'y':
                self.__df.update(sheet_df)
                new_occ = pd.Series(self.occupancies)
                new_check_df: pd.Series = new_occ - correct_uc_occ
                new_check_df.dropna(inplace=True)
                assert (new_check_df == 0).all(), f'New occupancies are non-integer!'
                self.__print_df_composition(sheet_df, old_composition=old_composition)
                while accept not in ['y', 'n', 'c']:
                    accept = input('\nAccept new composition? [y/n] (exit with c)\n').lower()
                if accept == 'n':
                    # logger.info(f'{accept}\n')
                    sheet_df: pd.Series = old_composition.copy()
                    sheet_df = sheet_df.loc[['T', 'O'], :]
                    sheet_df = sheet_df.loc[sheet_df != 0]
                    for k, v in check_occ.items():
                        if v != 0:
                            for atom, occ in sheet_df.loc[k, :].iteritems():
                                sheet_df.loc[k, atom] = float(input(f'Enter new value for {k!r} - {atom!r}: ({occ:2.2f}) -> '))
                elif accept == 'c':
                    self.__abort()
            # logger.info(f'{accept}\n')
            # for idx, val in self.match_df.iter
            self.__print_df_composition(sheet_df)
            # logger.info('Will use the following clay composition:')
            # for idx, occ in sheet_df.iteritems():
            #     sheet, atom = idx
            #     logger.info(f"\t{sheet!r} - {atom!r:^10}: {occ:2.2f}")
            # exp_group = exp_occ.groupby(self.sheet_grouper)
            # exp_group.apply(lambda x: print(f"{x.name}: {x.sum()}"))
            # self.df.update(exp_occ)

    @staticmethod
    def __print_df_composition(sheet_df, old_composition=None):
        if old_composition is None:
            logger.info('Will use the following composition:')
        else:
            logger.info('old occupancies -> new occupancies per unit cell:')
            logger.info('sheet - atom type : occupancies  (difference)')
        for idx, occ in sheet_df.iteritems():
            sheet, atom = idx
            try:
                old_val = old_composition[idx]
                logger.info(
                    f"\t{sheet!r:5} - {atom!r:^10}: {old_val:2.2f} -> {occ:2.2f} ({occ - old_val:+2.2f})"
                    # sheet occupancies of {new_occ[sheet]:.2f}/{correct_uc_occ[sheet]:.2f} ({occ:+.2f})")
                )
            except TypeError:
                logger.info(
                    f"\t{sheet!r:5} - {atom!r:^10}: {occ:2.2f}"
                )

    correct_t_occupancies = partialmethod(correct_occupancies, idx_sel=["T"])
    correct_o_occupancies = partialmethod(correct_occupancies, idx_sel=["O"])


    def _write(self, outpath: Union[Dir, File, PosixPath], fmt: Optional[str]=None):
        if type(outpath) == str:
            outpath = PathFactory(outpath)
        if fmt is None:
            if outpath.suffix == '':
                raise ValueError(f'No file format specified')
            else:
                fmt = outpath.suffix
        fmt = f'.{fmt.lstrip(".")}'
        if outpath.suffix == '':
            outpath = outpath / f"{self.name}_exp_df{fmt}"
        tmpfile = tempfile.NamedTemporaryFile(
            suffix=fmt, prefix=outpath.stem
        )
        if fmt == '.csv':
            self.df.to_csv(tmpfile.name)
        elif fmt == '.p':
            with open(tmpfile.name, 'wb') as file:
                pkl.dump(self.df)
        else:
            raise ValueError(f'Invalid format specification {fmt!r}\n'
                             f'Expected {".csv"!r} or {".p"!r}')
        if not outpath.parent.is_dir():
            os.makedirs(outpath.parent)
        logger.info(f'Writing new target clay compoition to {str(outpath)!r}')
        shutil.copy(tmpfile.name, outpath)

    write_csv = partialmethod(_write, fmt='.csv')
    write_pkl = partialmethod(_write, fmt='.csv')


    def get_ion_numbers(self):
        """Read ion types and ratios into pd DataFrame."""
        ion_probs: pd.Series = self.__df.loc['I'].copy()
        ion_df = ion_probs.to_frame(name='probs')
        ion_charge_dict = get_ion_charges()
        ion_df['charges'] = ion_df.index.to_series().apply(lambda x: ion_charge_dict[x])
        assert ion_probs.sum() == 1.00, f'Ion species probabilities need to sum to 1'

        ion_df.dropna(inplace=True, subset='probs')
        ion_df.where(np.sign(ion_df) != np.sign(self.get_total_charge()), level='charge', inplace=True)
        ion_df.where(ion_df != 0, level='probs', inplace=True)
        # charge_avg = ion_df['charges'].mean()
        self.__ion_df = ion_df.copy()

class MatchClayComposition:
    def __init__(self, target_composition, sheet_n_ucs: int):
        self.__target_df: pd.DataFrame = target_composition.clay_df
        self.__uc_data: UCData = target_composition.uc_data
        self.drop_unused_ucs()
        self.sheet_n_ucs = sheet_n_ucs
        self.__unique_uc_match_df

        # self.target_composition = target_composition
        # self.sheet_n_ucs = sheet_n_ucs
        # self.uc_list = self.unique_uc_array()

        # self.clay_df: pd.DataFrame = self.target_composition.df

    def drop_unused_ucs(self):
        all_ucs_df = self.__uc_data.df
        target_df = self.__target_df.dropna().copy() #.reset_index('sheet')
        combined_idx = (all_ucs_df.index.to_flat_index()).union(target_df.index.to_flat_index())
        combined_idx = pd.MultiIndex.from_tuples(combined_idx, names=target_df.index.names)
        all_ucs_df = all_ucs_df.reindex(combined_idx)
        target_df = target_df.reindex(combined_idx)
        unused_target_atype_mask = self.__get_nan_xor_zero_mask(target_df)
        accepted_group = {}
        for group_id, group_uc_ids in self.__uc_data.group_iter():
            uc_group_df = all_ucs_df[group_uc_ids]
            # discard all unit cells with non-zero values where target composition has zeros
            uc_group_df = uc_group_df.loc[:, (uc_group_df[unused_target_atype_mask] == 0).any(axis=0)]
            # check that the group has non-zero values for all atom types in the target composition
            unused_uc_atype_mask = (self.__get_nan_xor_zero_mask(uc_group_df)).all(axis=1)
            missing_uc_at_types = (uc_group_df[unused_uc_atype_mask]).index.difference(target_df[unused_target_atype_mask].index)
            if len(missing_uc_at_types) == 0:
                accepted_group[group_id] = list(uc_group_df.columns)
            # combined_mask = np.logical_and(unused_uc_atype_mask, unused_target_atype_mask)
        accept = None
        if len(accepted_group) == 1:
            selected_ucs_df = all_ucs_df.loc[:, accepted_group[next(iter(accepted_group.keys()))]]
            self.print_groups(accepted_group, all_ucs_df)
            while accept not in ['y', 'n']:
                accept = input(f'Accept unit cell group? [y/n]\n')
        elif len(accepted_group) == 0:
            raise ValueError(f'Not all target compoistion atom types were found in the unit cells!')
        else:
            logger.info(f'Found the following unit cell groups:')
            self.print_groups(accepted_group, all_ucs_df)
            uc_id_str = '/'.join(accepted_group.keys())
            while accept not in [accepted_group.keys(), 'n']:
                accept = input(f'Select unit cell group? [{uc_id_str}n]\n')
            if accept in accepted_group.keys():
                logger.info(f'Selected group {accept}')
                accepted_group = {accept, accepted_group.get(accept)}
                self.print_groups(accepted_group, all_ucs_df)
                accept = 'y'
        if accept == 'n':
            self.__abort()
        else:
            self.__uc_data.select_ucs(accepted_group[next(iter(accepted_group.keys()))])
            assert self.target_df.index.equals(self.uc_df.index), \
                'Target composition and unit cell data must share index'

        #
        #
        # # if all_ucs_df.hasnans:
        # #     missing_uc_atoms = all_ucs_df.is
        # #     accept = None
        # #     while accept not in ['y', 'n']:
        #
        # # all_ucs_df = all_ucs_df.copy().loc[target_at_types]
        #
        # # unused_ucs_mask = (__target_df == 0 || __target_df == np.nan).values
        # clay_m = (all_ucs_df[unused_ucs_mask] == 0)
        # idx = clay_m.where(clay_m == True, np.nan).dropna(axis=1).columns
        # new_all_ucs_df = all_ucs_df.loc[:, idx]
        # return new_all_ucs_df

    @staticmethod
    def __abort(self):
        logger.info('No unit cell group accepted. Aborting model construction.')
        sys.exit(0)

    @staticmethod
    def print_groups(group_dict, uc_df):
        for group_id, group_ucs in group_dict.items():
            uc_group_df = uc_df.loc[:, group_ucs]
            logger.info(f'Group {group_id}:')#{uc_group_df.columns}')
            logger.info(f'{"":10}\tUC occupancies')
            uc_list = '  '.join(list(map(lambda v: f'{v:>3}', uc_group_df.columns)))
            logger.info(f'{"UC index":<10}\t{uc_list}')
            logger.info(f'{"atom type":<10}')
            for idx, values in uc_group_df.sort_index(ascending=False).iterrows():
                sheet, atype = idx
                composition_string = '  '.join(list(map(lambda v: f'{v:>3}', values.astype(int))))
                logger.info(f'{sheet!r:<3} - {atype!r:^4}\t{composition_string}')

    @staticmethod
    def __get_nan_xor_zero_mask(df):
        return np.logical_xor(np.isnan(df), df == 0)

    @property
    def uc_df(self):
        return self.__uc_data.df.reindex(index=self.target_df.index) #.sort_index(ascending=False, level='sheet', sort_remaining=True).dropna()

    @property
    def target_df(self):
        return self.__target_df.sort_index(ascending=False, level='sheet', sort_remaining=True).dropna()

    @property
    def duplicate_ucs(self):
        ...

    @cached_property
    def __unique_uc_match_df(self):
        # rounded_target_composition = self.target_df.round(1)
        uc_df = self.uc_df.copy()
        uc_df.columns = uc_df.columns.astype(int)
        n_ucs_idx = pd.Index([x for x in range(2, len(self.unique_uc_array) + 1)], name='n_ucs')
        match_df = pd.DataFrame(columns=['uc_id', 'uc_list', 'composition', 'dist'],
                                index=n_ucs_idx)
        for n_ucs in n_ucs_idx:
            logger.info(f'Getting combinations for {n_ucs} unique unit cells')
            uc_id_combinations = self.get_uc_combinations(n_ucs)
            occ_combinations = self.get_sheet_uc_weights(n_ucs)
            for uc_ids in uc_id_combinations:
                atype_combinations = uc_df[[*list(uc_ids)]].astype(float).T.values
                combinations_iter = np.nditer([occ_combinations, atype_combinations, None],
                                              flags=['external_loop'],
                                              op_axes=[[1, -1, 0], [0, 1, -1], None])
                for cell, element, weight in combinations_iter:
                    weight[...] = cell * element
                atype_weights = combinations_iter.operands[2] / self.sheet_n_ucs
                atype_weights = np.add.reduce(atype_weights, axis=0)
                diff_array = np.subtract(
                    atype_weights.T, np.squeeze(self.target_df.values))
                diff_array = np.linalg.norm(diff_array, axis=1)
                dist = np.amin(diff_array)
                match = diff_array == np.amin(diff_array)
                match_df.loc[n_ucs] = (uc_ids, np.squeeze(occ_combinations[match]),
                                      np.squeeze(
                                          np.round(atype_weights.T[match], 4)),
                                      np.round(dist, 4))
        return match_df[match_df['dist'] == match_df['dist'].min()].head(1)

    @cached_dproperty
    def match_composition(self):
        return pd.Series(self.__unique_uc_match_df['composition'], index=self.target_df.index)

    @property
    def match_diff(self):
        return self.__unique_uc_match_df['dist']

    def select_duplicate_ucs(self):
        ...

    def get_uc_numbers_df(self):
        ...

    @cached_property
    def unique_uc_array(self):
        unique_uc_ids = self.uc_df.T.drop_duplicates().index.values.astype(int)
        # return self.uc_df.T.drop_duplicates().index.values
        return unique_uc_ids

    # @cached_property
    def get_sheet_uc_weights(self, n_ucs: int):
        """
        Returns a list of lists with the combinations of 2, 3, sheet_n_ucs given the
        columns.
        """
        all_sheet_uc_combinations = {}
        # max_comb_len = len(self.unique_uc_array) + 1
        # for n_ucs in range(2, max_comb_len):
        #     temp_combs = np.asarray(
        #         [x for x in self.get_uc_combination_list(self.sheet_n_ucs, n_ucs)])
        #     all_sheet_uc_combinations[n_ucs] = temp_combs
        # return all_sheet_uc_combinations
        sheet_uc_combinations = np.asarray(
                [x for x in self.get_uc_combination_list(self.sheet_n_ucs, n_ucs)])
        return sheet_uc_combinations

    @staticmethod
    def get_uc_combination_list(N, k):
        for q in itertools.combinations(range(N - 1), k - 1):
            yield [j - i for i, j in zip((-1,) + q, q + (N - 1,))]

    def get_uc_combinations(self, n_ucs):
        return np.asarray(list(itertools.combinations(self.unique_uc_array, n_ucs)))

        # ion_probs = ion_probs.rename(columns={c.X_EL: 'ratios'}
        #                              ).rename_axis(index={'element': 'ion species'}
        #                                            )
        # ion_df = pd.merge(ion_probs.reset_index(),
        #                   ion_charges.reset_index(),
        #                   on='ion species')
        # ion_df.sort_index(inplace=True, kind='mergesort')
        # print(ion_df)
        # return ion_df

    # def get_ion_mols(tot_charge=r.CHARGE):
    #     """Compute number of individual ion species atoms."""
    #     tot_charge = -tot_charge
    #     print(f'{tot_charge} total charge')
    #     n_tot_add = tot_charge % compute_charge_average()
    #     print(n_tot_add)
    #     ion_df = get_ion_df().set_index('ion species')
    #     print(ion_df)
    #     ion_df['n_atoms'] = tot_charge * ion_df['ratios'] // ion_df['charge']
    #     n_tot_add = tot_charge - (ion_df['charge'] * ion_df['n_atoms']).sum()
    #     ion_df.at['Na', 'n_atoms'] += n_tot_add
    #     ion_df = ion_df.where(ion_df['n_atoms'] != 0).dropna().convert_dtypes()
    #     print(f'{ion_df} \n Ion dataframe')
    #     print(ion_df)
    #     print(ion_df['n_atoms'].to_dict())
    #     return ion_df['n_atoms'].to_dict()

# class ExpComposition:
#     # print(ion_charges.head(5))
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
#     def __init__(self, data_dir, _target_comp, sysname, ions_ff, clay_ff, uc_type, n_cells):
#         ion_charges = ions_ff.atomtypes
#         self.target_comp = File(_target_comp).resolve()
#         self.data_dir = data_dir
#         self.clay_type = uc_type
#         self.sysname = sysname
#         self.n_cells = n_cells
#         self.df = pd.DataFrame(
#             index=self.__class__.exp_index, columns=["charge", sysname, "el_chg"], dtype="float"
#         )
#         self.add_element_charge_info()
#         self.add_exp_data()
#         self.charge_occ_info = self.get_uc_occ_charge_info()
#         self.correct_occupancies()
#         self.add_uc_charge_info()
#         pd.to_numeric(self.df.loc[:, sysname])
#         self.check_exp_charge_info()
#         self.round_sheet_df_occ()
#         self.get_fe2_feo_values()
#         self.add_uc_charge_info()
#         self.check_uc_charge()
#         self.df.drop(index="fe_tot", level="element", inplace=True)
#         self.df.loc[["T", "O"]] = self.df.loc[["T", "O"]].fillna(0)
#
#     @staticmethod
#     def get_df_from_csv(fname, index):
#         df = pd.read_csv(fname).fillna(method="ffill").set_index(index)
#         return df
#
#     def add_element_charge_info(self):
#         element_charges = pd.DataFrame.from_dict(
#             self.__class__.charges_mapping, orient="index", columns=["charge"]
#         )
#         self.df.reset_index(level=0, inplace=True)
#         self.df.update(element_charges)
#         self.df = self.df.reset_index().set_index(["sheet", "element"])
#
#     def add_exp_data(self):
#         exp_data = self.get_df_from_csv(self.target_comp, ["sheet", "element"])
#         self.df.update(exp_data)
#
#     def get_occupations(self):
#         return self.uc_df.iloc[:, :-1].groupby(['sheet']).sum().aggregate("unique",
#                                                                           axis='columns')
#
#     def get_charges(self):
#         charges = self.uc_df.iloc[:, -1].apply
#
#     def get_uc_occ_charge_info(self):
#         charge_occ_info = self.get_df_from_csv(self.data_dir / "charge_occ.csv", ["sheet", "value"])
#         charge_occ_info = charge_occ_info.xs(self.clay_type, level=1)
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
#         chg = x[self.sysname] * x["charge"] - x[self.sysname] / self.get_occ_info(
#             x
#         ) * self.get_tot_charge(x)
#         return chg
#
#     def check_exp_charge_info(self):
#         if self.df.at[("C", "tot"), self.sysname] != np.sum(
#             self.df.loc[("C", ["T", "O"]), self.sysname]
#         ):
#             chg_t = self.df["el_chg"].groupby(self.sheet_grouper).sum().loc["T"]
#             self.df.loc[("C", "O"), self.sysname] = (
#                 self.df.at[("C", "tot"), self.sysname] - chg_t
#             )
#             self.df.loc[("C", "T"), self.sysname] = chg_t
#         else:
#             print(
#                 self.df.at[("C", "tot"), self.sysname],
#                 np.sum(self.df.loc[("C", ["T", "O"]), self.sysname]),
#             )
#
#     def add_uc_charge_info(self):
#         el_chg = self.df.loc[["T", "O"]]
#         el_chg = el_chg.groupby(self.sheet_grouper).apply(
#             lambda x: self.get_chg_info(x)
#         )
#         el_chg.reset_index(level=1, drop=True, inplace=True)
#         el_chg.name = "el_chg"
#         self.df.update(el_chg)
#
#     def get_fe2_feo_values(self):
#         print("fe2_fe3")
#         chg_o = self.df.at[("C", "O"), self.sysname]
#         print(chg_o)
#         chg_not_fe = (
#             self.df[~self.df.index.isin(["fe2", "feo"], level=1)]
#             .loc["O", "el_chg"]
#             .sum()
#         )
#         self.df.loc[("O", "fe2"), self.sysname] = -chg_o + chg_not_fe
#         x_fe_tot = self.df.at[("O", "fe_tot"), self.sysname]
#         self.df.loc[("O", "feo"), self.sysname] = (
#             x_fe_tot - self.df.loc[("O", "fe2"), self.sysname]
#         )
#
#     def check_uc_charge(self):
#         chg_df = self.df.loc[["T", "O"], "el_chg"]
#         chg_df = chg_df.groupby(self.sheet_grouper).sum()
#         chg_df.name = "el_chg"
#         chg_df.index.name = "element"
#         chg_df.index = pd.MultiIndex.from_product(
#             [["C"], chg_df.index], names=self.df.index.names
#         )
#         self.df.update(chg_df)
#         self.df.loc[("C", "tot")] = chg_df.sum()
#         check = self.df.xs(("C", "tot")).loc[[self.sysname, "el_chg"]]
#         if not check[self.sysname] == check["el_chg"]:
#             raise ValueError(
#                 f"Specified unit cell charge of {self.sysname} "
#                 f"does not correspond to calculated unit cell "
#                 "charge."
#             )
#
#     def correct_occupancies(self, idx_sel=["T", "O"]):
#         exp_occ = self.df.loc[[*idx_sel], self.sysname].dropna()
#         print(self.df, exp_occ)
#         exp_occ = exp_occ.loc[exp_occ != 0]
#         exp_occ = exp_occ.groupby(self.sheet_grouper)
#         check_occ = self.charge_occ_info["occ"] - exp_occ.sum()
#         print(exp_occ, check_occ)
#         print("Sheet occupancies found:")
#         exp_occ.apply(lambda x: print(f"{x.name}: {x.sum()}"))
#         if check_occ.values.any() != 0:
#             print("Adjusting values to match expected values:")
#             exp_occ = exp_occ.apply(lambda x: x + check_occ.at[x.name] / x.count())
#             exp_group = exp_occ.groupby(self.sheet_grouper)
#             exp_group.apply(lambda x: print(f"{x.name}: {x.sum()}"))
#             self.df.update(exp_occ)
#
#     correct_t_occupancies = partialmethod(correct_occupancies, idx_sel=["T"])
#     correct_o_occupancies = partialmethod(correct_occupancies, idx_sel=["O"])
#
#     def round_sheet_df_occ(self):
#         self.df.loc[["T", "O", "C"]] = self.df.loc[["T", "O", "C"]].round(2)
#
#     def write_occ_df(self, outpath):
#         if outpath.is_dir():
#             self.df.to_csv(outpath / f"{self.sysname}_exp_df.csv")
#         else:
#             raise NotADirectoryError(f"{outpath} does not exist.")

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

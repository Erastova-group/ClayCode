from pathlib import Path
from typing import Union, Dict, List

import attr
from attr import validators, define, converters

from config import buildprms, defaults

from config.lib import (
    get_params,
    get_paths,
    init_outpath,
    check_outpath,
    get_outdir,
    check_build,
    assign_params,
    find_files,
    file_exists,
    # validate_path,
    set_il_solv,
    check_bool_false,
    assign_paths, process_ff, process_ucs
)
from config.types import validate_path


__name__ = 'params'

__all__ = ['Params', 'BuildParams', 'SiminpParams']


# _BUILDPRMS = {
#     str.lower(key): value
#     for key, value in buildprms.__dict__.items()
#     if not key.startswith("__")
# }
# _PATHPRMS = {str.lower(key): value
#              for key, value in pathprms.__dict__.items()
#              if not key.startswith("__")
#              }
#


PRMS = get_params(buildprms) | get_paths(defaults) # | get_paths(paths)
print(PRMS)




@define(kw_only=True, slots=True)
class Params:
    sysname = attr.ib(type=str, validator=validators.instance_of(str), init=False)

    build = attr.ib(
        type=Union[bool, str, Dict[str, Union[str, Path]]],
        init=False,
        validator=[
            # validators.instance_of(Union[bool, str, dict]),
            check_bool_false,
            check_build,
        ],
    )
    siminp = attr.ib(
        type=Union[bool, str, List[str]],
        init=False,
        # validator=[validators.instance_of(Union[bool, str, list]), check_bool_false],
    )
    clay_type = attr.ib(
        type=str, init=False, validator=validators.matches_re("D21|D11|T21|T11")
    )
    outpath = attr.ib(
        type=Union[str, Path],
        # validator=validate_path,
        converter=Path,
        default=Path.cwd(),
        init=True,
    )
    prm_dict = attr.ib(type=dict)
    ff = attr.ib(
        default={
            "clay": {
                "selection": {
                    "ClayFF_Fe": [
                        "atomtypes",
                        "ffbonded",
                        "ffnonbonded",
                        "forcefield",
                        "spc",
                        "interlayer_spc",
                    ]
                }
            },
            "ions": {"selection": {"AmberIons": ["AMBER-IOD", "ions"]}},
        }
    )
    clay_ff = attr.ib(init=False)
    ions_ff = attr.ib(init=False)
    clay_units = attr.ib(init=False)
    uc_charges = attr.ib(init=False)
    _wdir = attr.ib(default=Path.cwd(), type=Path)
    _srcdir = attr.ib(default=Path(__file__).parents[1], type=Path)
    data_dir = attr.ib(init=False, type=Path, converter=Path, validator=validate_path)
    ff_dir = attr.ib(
        init=False, type=Path, converter=Path, validator=validate_path
    )  # (default=_data_path.rglob('FF'), validator=validators.deep_iterable(member_validator=Path, iterable_validator=list))
    clay_units_dir = attr.ib(init=False, type=Path, converter=Path, validator=validate_path)
    _mdp_dir = attr.ib(init=False, type=Path, converter=Path, validator=validate_path)
    mdp_dict = attr.ib(
        init=False,
        type=Dict[str, Path],
        validator=validators.deep_mapping(
            key_validator=validators.instance_of(str),
            value_validator=validate_path,
            mapping_validator=validators.instance_of(dict),
        ),
    )

    def __attrs_post_init__(self):
        print(self.ff)
        assign_params(self)
        print(self.ff)
        assign_paths(self)
        print(self.ff)
        check_outpath(self)
        # print(self._ff_path)
        print(self.__slots__)
        print(self.ff)
        process_ff(self)
        process_ucs(self)

    def extract_vars(self):
        return {key: getattr(self, key) for key in self.__slots__ if not key.startswith('_')}


@define(kw_only=True, slots=True, init=False)
class BuildParams(Params):
    build = attr.ib(
        type=Union[str, Path, Dict[str, List[Union[str, Path]]]],
        validator=[check_build],  # validators.instance_of(Union[str, Path, dict]),
    )
    clay_comp = attr.ib(
        type=Union[str, Path],
        validator=[
            validators.optional(validators.matches_re(".*[.]csv")),
            # validators.optional(validators.instance_of(Union[str, Path])),
        ],
    )
    x_cells = attr.ib(type=int, validator=validators.instance_of(int))
    y_cells = attr.ib(validator=validators.instance_of(int), type=int)
    n_sheets = attr.ib(default=1, type=int, validator=validators.instance_of(int))
    box_height = attr.ib(
        default=0,
        type=Union[float, int],
        # validator=[
        #     validators.instance_of(Union[float, int]),
        #     validators.le(20.0),
        # ],
        converter=converters.optional(float),
    )
    il_solv = attr.ib(
        type=Union[bool, Dict[str, Union[int, Dict[str, int]]]]
    )  # , validator=validators.instance_of(bool))
    _il_waters = attr.ib(
        type=Dict[str, Union[Dict[str, int], int]]
        # validator=[
        # validators.deep_mapping(
        # key_validator=validators.in_(["ion", "uc", "spacing"]),
        # value_validator=validators.instance_of(Any)
        # value_validator=validators.instance_of(Union[dict, list, int, float]),
        # )
        # ],
    )
    _uc_waters = attr.ib(
        type=int,
        validator=[validators.instance_of(int), validators.ge(0), validators.le(40)],
    )
    _ion_waters = attr.ib(
        type=int,
        validator=[validators.instance_of(int), validators.ge(0), validators.le(20)],
    )
    _spacing_waters = attr.ib(
        type=float,
        validator=[
            validators.instance_of(Union[int, float]),
            validators.ge(0),
            validators.le(4),
        ],
        converter=float,
    )
    #root = attr.ib(type=Path, default=Path(__file__).parents[1])

    def __attrs_post_init__(self):
        assign_params(self)
        assign_paths(self)
        setattr(self, "uc_dir", self.uc_dir / self.clay_type)
        check_outpath(self)
        get_outdir(self, "build")
        init_outpath(self)
        set_il_solv(self)


@define(kw_only=True, slots=True, init=False)
class SiminpParams(Params):
    siminp = attr.ib(
        type=Union[str, List[str]],
        # validator=validators.instance_of(Union[str, list]),
        converter=list,
    )
    gro = attr.ib(
        type=Union[str, Path],
        validator=[validators.matches_re(".*[.]gro"), file_exists],
        converter=str,
    )
    top = attr.ib(
        type=Union[str, Path],
        validator=[validators.matches_re(".*[.]top"), file_exists],
        converter=str,
    )
    mdruns_remote = attr.ib(type=bool, validator=validators.instance_of(bool))

    d_space = attr.ib(
        type=Union[int, float],
        validator=validators.le(25.0),
        converter=lambda space: space / 10,
    )
    remove_steps = attr.ib(
        type=int,
        validator=[
            validators.instance_of(int),
            validators.le(1000000),
            validators.ge(1000),
        ],
    )
    sheet_wat = attr.ib(
        type=int, validator=[validators.instance_of(int), validators.gt(0)]
    )
    uc_wat = attr.ib(
        type=Union[int, float],
        validator=[validators.instance_of(Union[int, float]), validators.gt(0.0)],
        converter=float,
    )
    percent_wat = attr.ib(
        type=Union[int, float],
        validator=[validators.instance_of(Union[int, float]), validators.gt(0.0)],
        converter=float,
    )
    siminp_options = attr.ib(
        type=Union[list, str],
        # validator=validators.instance_of(Union[str, list]),
        converter=list,
    )
    ff_path_sim = attr.ib(
        type=Union[str, Path], validator=validate_path, converter=Path
    )
    ff_path_sim = attr.ib(
        type=Union[str, Path], validator=validate_path, converter=Path
    )
    gmx_path_sim = attr.ib(
        type=Union[str, Path], validator=validate_path, converter=Path
    )
    python_path_remote = attr.ib(
        type=Union[str, Path], validator=validate_path, converter=Path
    )
    conda_envs_dirs = attr.ib(
        type=Union[str, Path], validator=validate_path, converter=Path
    )
    conda_pkgs_dirs = attr.ib(
        type=Union[str, Path], validator=validate_path, converter=Path
    )
    conda_env_name = attr.ib(
        type=Union[str, Path], validator=validate_path, converter=Path
    )
    reinsert_ions = attr.ib(
        type=str,
        validator=validators.instance_of(str),
    )
    substitute_ions = attr.ib(type=str, validator=validators.instance_of(str))
    reinsert_dist = attr.ib(
        type=Union[int, float],
        validator=validators.instance_of(Union[int, float]),
        converter=float,
    )

    def __attrs_post_init__(self):
        assign_params(self)
        check_outpath(self)
        get_outdir(self, "siminp")
        init_outpath(self)
        find_files(self, "gro")
        find_files(self, "top")


# @define(kw_only=True, slots=True)
# class SiminpParams(Params):
#     siminp = attr.ib(
#         type=Union[str, List[str]],
#         default=Factory(list),
#         validator=validators.instance_of(Union[str, list]),
#     )


# def assign_consts(instance, prm_dict):
#     for prm_key in prm_dict:
#         try:
#             instance.__settattr__(str.lower(prm_key), prm_dict[prm_key])
#         except AttributeError:
#             continue
#     return instance


PRMS = Params(prm_dict=PRMS)
# print(CONSTS.build, CONSTS.siminp)
# # PRMS = assign_consts(PRMS, prm_dict=PRMS)
# print(PRMS.build, PRMS.siminp)
# for prm in PRMS:
#     try:
#         PRMS.__setattr__(str.lower(prm), PRMS[prm])
#         print(f'assigned {prm}')
#     except:
#         print(prm)
#
# # print(PRMS.build, PRMS.siminp)
# if CONSTS.build != False:
#     BUILD_CONSTS = BuildParams(prm_dict=PRMS)
#     print(
#         BUILD_CONSTS.il_solv,
#         BUILD_CONSTS.FF["clay"] | BUILD_CONSTS.FF["ions"],
#         BUILD_CONSTS._ff_dir,
#     )
# #     assign_consts(b, prm_dict=PRMS)
# #
# # nested_dict = BUILD_CONSTS.FF
# # reformed_dict = {}
# # for outerKey, innerDict in nested_dict.items():
# #     for innerKey, values in innerDict.items():
# #         reformed_dict[(outerKey, innerKey)] = values
# # print(reformed_dict)
# # # Display multiindex dataframe
# # multiIndex_df = pd.DataFrame.from_dict(reformed_dict.items(), orient='columns')
# # print(multiIndex_df)
#
# if CONSTS.siminp != False:
#     print("Setting siminp CONSTS!")
#     SIMINP_CONSTS = SiminpParams(prm_dict=PRMS)

#     assign_consts(s, prm_dict=PRMS)
# # CONSTS = ArgumentParser('Get run parameters for ClayCode')
# # CONSTS.add_argument('-name', type=str, help='Name of clay system.', dest='sysname')
# # CONSTS.parse_args(['-name', 'NAu-1-new'])
# #
# # class Cons
#
#
# CONFIG = Params()
# try:
#     CONFIG.outpath = bp.OUTPATH
# except AttributeError:
#     CONFIG.outpath = Path.cwd()

# print(CONFIG.outpath)
#
# class Params(BaseModel):
#     sysname: str
#     outpath: Union[str, Path] = Path.cwd()
#
#     @validator('sysname')
#     def name_length(cls, value):
#         maxlen = 15
#         if len(value) >= maxlen:
#             raise ValueError(f'Selected system name {value!r} exceeded maximum number of {maxlen} characters.')
#
#     @validator('outpath', pre=True)
#     def check_outpath(cls, value):
#         if not Path(value).is_dir():
#             raise NotADirectoryError(f'{value} is not a directory.')
#         self.__setattr__('outpath', self.outpath / self.sysname)
#
#     def __new__(self):
#         self.__setattr__('outpath', self.outpath / self.sysname)
#         print(self.outpath)
#
# a = 'a'
# print(f'{a!r}')
#
# x = Params(sysname='a')

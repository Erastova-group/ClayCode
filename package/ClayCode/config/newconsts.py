import os
from collections import UserList
from typing import Optional, Literal, Union, List, Dict, TypeVar

from pydantic import BaseModel, create_model, Field, validator
from pydantic.utils import GetterDict
from typing_extensions import TypeAlias

SimTypes: TypeAlias = Union[Literal['EM'], Literal['EQ'], Literal['P'], Literal['D_SPACE'], Literal['REINSERT']]

def assert_file_exists(fname):
    file = File(fname)
    assert file.is_file(), f'{file!r} does not exist'
    return file


class LiteralListUnion:
    def __new__(self, *args):
        return Union[tuple(Literal[arg] for arg in args)]


from config.classes import File, Dir, PosixFile, PosixDir, _PosixPath, _Path

# def get_literals(*args):
#     for arg in args:
#         yield type[arg]

FileType: TypeAlias = Union[File, PosixFile, os.PathLike, _PosixPath, _Path]

class Consts(BaseModel):
    sysname: str = Field(default=...,
                         alias='sysname',
                         description='Name for output folder')
    build: Optional[Union[LiteralListUnion('new', 'load'),
                          Dict[Literal['load'], Union[str, File, List[Union[str, File]]]]]] = Field(
                                                           alias='buildspecs',
                                                           description='"new" for building new clay model\n'
                                                                       '"load" for loading existent model')

    siminp: Optional[LiteralListUnion('EM', 'EQ', 'P', 'D_SPACE', 'REINSERT')] = Field(
        alias='siminpspecs',
        description='"EM": Energy minimisation\n'
                    '"EQ": Equilibration\n'
                    '"P": Production\n'
                    '"D_SPACE": d-spacing equilibration\n'
                    '"INSERT": molecule (re-)insertion'
    )
    clay_comp: Optional[Union[str, Union[str, File]]]

    @validator('build')
    def process_buildspecs(cls, value):
        if type(value) == Dict['load', List[Union[str, File]]]:
            for vi, v in enumerate(value['load'].values()):
                v = assert_file_exists(v)
                value['load'][vi] = v
        elif type(value) == Dict['load', Union[str, File]]:
            v = assert_file_exists(value['load'])
        return value


    @validator('clay_comp', pre=True)
    def file_exists(cls, value):
        v = assert_file_exists(value)
        return v


a = Consts(sysname='b', buildspecs='load',
           clay_comp='/storage/PycharmProjects/clay/data/exp_4.csv')

print(a)


# BuildParams = create_model('Params',
#

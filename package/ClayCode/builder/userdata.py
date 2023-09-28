#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import os
import shutil
import sys
from typing import List, Literal, Optional, Union

import numpy as np
from ClayCode import PathOrStr
from ClayCode.builder.utils import select_input_option
from ClayCode.core.classes import (
    Dir,
    DirFactory,
    File,
    FileList,
    PathListFactory,
    YAMLFile,
)
from ClayCode.core.consts import DATA, FF, UCS

logger = logging.getLogger(__name__)


class UserData:
    """Class for managing user data files.
    Example:
    >>> u = UserData()
    >>> u.add("/path/to/new_ff.ff", dtype="FF", new_name="NewFF.ff", exists_ok=True, filenames=["ffnonbonded", "spc"])
    """

    _dest = {"FF": FF, "UCS": UCS}
    _added_files_yaml = YAMLFile(DATA / "user_files.yaml")
    _odir = DATA

    def add(
        self,
        path: PathOrStr,
        dtype: Union[Literal["FF"], Literal["UCS"]],
        new_name: Optional[str] = None,
        filenames: Optional[List[PathOrStr]] = None,
        exists_ok: bool = False,
    ) -> None:
        """Add a new FF or UCS file to the ClayCode data directory.
        :param path: Path to the new directory.
        :type path: PathOrStr
        :param dtype: Type of data to add. Either "FF" or "UCS".
        :type dtype: Literal["FF", "UCS"]
        :param new_name: Name of the new data directory. If None, the name of the new directory will be the same as the name of the new directory.
        :type new_name: Optional[str]
        :param filenames: List of filenames to copy. If None, all files will be copied.
        :type filenames: Optional[List[PathOrStr]]
        :param exists_ok: If True, the new data directory will be written into an existing directory with the same name. If False, the program will exit if the new data directory already exists.
        :type exists_ok: bool
        :raises ValueError: If the number of unit cell ITP and GRO files do not match.
        :return: None
        """

        datapath = DirFactory(path)
        if new_name is None:
            new_name = datapath.name
        try:
            dest = Dir(self.__class__._dest[dtype])
        except KeyError:
            logger.error(f"Invalid data category {dtype!r}")
            sys.exit(3)
        filelists = {}
        filelists["itp"] = datapath.itp_filelist
        if dtype == "UCS":
            filelists["gro"] = datapath.gro_filelits
        for list_id, filelist in filelists.items():
            if filenames is not None:
                filelists[list_id] = filelist.filter(filenames)
        if dtype == "UCS":
            if (
                len(
                    np.unique(
                        [len(filelist) for filelist in filelists.values()]
                    )
                )
                != 1
            ):
                raise ValueError(
                    "Numbers of unit cell ITP and GRO files do not match!"
                )
            copy_filelist = filelist["gro"] + filelists["itp"]
        else:
            copy_filelist = filelists["itp"]
        if len(copy_filelist) != 0:
            odir = dest / new_name
            try:
                os.mkdir(odir)
            except FileExistsError:
                if odir == datapath:
                    logger.error(
                        f"New data directory is identical to destination path!\nAborting data addition."
                    )
                    sys.exit(2)

                exists_ok = select_input_option(
                    instance_or_manual_setup=False,
                    query="Write into existing directory?\n [y]es/[n]o (default no)",
                    options=["y", "n", True, False],
                    result=exists_ok,
                    result_map={
                        "y": True,
                        "n": False,
                        True: True,
                        False: False,
                    },
                )
                if not exists_ok:
                    logger.error(
                        f"{odir.name!r} already exists!\n"
                        "Not writing into existing directory"
                    )
                    sys.exit(2)
                else:
                    logger.finfo(
                        f"Will write into existing directory {odir.name!r}."
                    )
                    existing_files = odir.filelist
                    duplicates = set(existing_files.names) & set(
                        copy_filelist.copy().extract_fnames(
                            pre_reset=False, post_reset=False
                        )
                    )
                    if duplicates:
                        if new_name in self.added_files[dtype]:
                            overwrite = select_input_option(
                                instance_or_manual_setup=False,
                                query=", ".join(duplicates)
                                + " already exists! Overwrite? [y]es/[n]o\n",
                                options=["y", "n"],
                                result_map={"y": True, "n": False},
                            )
                            if not overwrite:
                                logger.finfo(
                                    f"Will not replace "
                                    + ", ".join(duplicates)
                                )
                                copy_filelist = copy_filelist - duplicates
                    for file in copy_filelist:
                        shutil.copy(file, odir / file.name)
                        logger.finfo(
                            f"Copied {file.name!r} into {odir.name!r}"
                        )
            else:
                self.added_files = {dtype: odir.name}
        else:
            logger.error(f"No datafiles found in {datapath.name!r}")
            sys.exit(2)

    @property
    def added_files(self) -> dict:
        """Dictionary of added files.
        :return: Dictionary of added files."""
        try:
            added_files = self._added_files_yaml.data
        except FileNotFoundError:
            added_files = {}
        finally:
            return added_files

    @added_files.setter
    def added_files(self, added_files_dict):
        """Set the dictionary of added files."""
        self._added_files_yaml.data.update(added_files_dict)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.added_files.__repr__()})"

    def __str__(self):
        return f"{self.__class__.__name__}({self.added_files.__str__()})"

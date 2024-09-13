# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# r""":mod:`ClayCode.siminp.rmwat` --- Remove water molecules from a clay model
# ==============================================================================
# """
# import logging
# import os
# import pathlib as pl
# import pickle as pkl
# import random
# import re
# import shutil
# import sys
# import warnings
# from argparse import ArgumentParser
# from functools import cached_property
# from typing import Optional
#
# import MDAnalysis as mda
# import numpy as np
#
# logger = logging.getLogger(__name__)
#
# warnings.filterwarnings("ignore")
#
# parser = ArgumentParser(
#     description="Remove water molecules from a the clay model interlayer."
# )
#
# parser.add_argument(
#     "-i", type=int, help="The index of the spacing equilibration run."
# )
# parser.add_argument(
#     "-n",
#     type=str,
#     help="The name of the spacing equilibration run.",
#     required=False,
#     default=None,
# )
# parser.add_argument(
#     "-w", type=int, help="The number of water molecules to remove."
# )
# parser.add_argument("-o", type=str, help="The name of the output directory.")
# parser.add_argument(
#     "-c",
#     type=str,
#     help="The name of the input .gro file.",
#     required=False,
#     default=None,
# )
# parser.add_argument(
#     "-t",
#     type=str,
#     help="The name of the input .top file.",
#     required=False,
#     default=None,
# )
# parser.add_argument(
#     "-n_ucs",
#     type=int,
#     help="Number of unit cells per sheet.",
#     required=False,
#     default=None,
# )
# remove_wat_args = parser.add_mutually_exclusive_group(required=False)
# remove_wat_args.add_argument(
#     "uc_wat",
#     type=float,
#     help="Number of water molecules per unit cell to remove.",
# )
# remove_wat_args.add_argument(
#     "sheet_wat",
#     type=float,
#     help="Fraction of water molecules per clay sheet to remove.",
# )
# remove_wat_args.add_argument(
#     "percent_wat", type=float, help="Percent of water molecules to remove."
# )
#
#
# WDIR = pl.Path(__file__).parent
# logger.info(f"Current working directory: {WDIR.resolve()}\n")
# SPACINGS_FILENAME = WDIR / "spacings_array.p"
#
# logger.info(f'Running "{sys.argv[0]} {" ".join(sys.argv[1:])}"\n')
#
#
# def raise_sys_exit(*optional_printargs):
#     if optional_printargs:
#         optional_print = "\n".join(optional_printargs)
#         logger.info(optional_print)
#     raise SystemExit(
#         f"Usage: {sys.argv[0]} <run_number> " "<run_name> <waters_to_remove>"
#     )
#
#
# def get_sim_fname(simnum, simname, ext, path=WDIR):
#     return path / f"{simname}_{simnum:02d}/{simname}_{simnum:02d}.{ext}"
#
#
# class RMWat:
#     def __init__(
#         self,
#         remove_waters: int,
#         run_id: Optional[int] = None,
#         run_name: Optional[str] = None,
#         igro: Optional[str] = None,
#         itop: Optional[str] = None,
#         itraj: Optional[str] = None,
#     ):
#         try:
#             self.remove_waters = int(remove_waters)
#         except ValueError:
#             raise TypeError(
#                 f"remove_waters must be an integer, not {remove_waters.__class__.__name__}"
#             )
#         self.run_name = run_name
#         self.filesdict = {"igro": igro, "itop": itop, "itraj": itraj}
#         for filetype in ["gro", "top", "trr"]:
#             for file_id, inout in enumerate(["i", "o"]):
#                 if (
#                     self.filesdict[f"{inout}{filetype}"] is None
#                     and self.run_name is not None
#                     and self.run_id is not None
#                 ):
#                     self.filesdict[f"{inout}{filetype}"] = get_sim_fname(
#                         self.run_id + file_id, self.run_name, filetype
#                     )
#                 elif self.filesdict[f"{inout}{filetype}"] is not None:
#                     if file_id == 0:
#                         fname = self.filesdict[f"{inout}{filetype}"].stem
#                         fname = re.match(
#                             r"(.*?)(_[0-9]*)?", fname, flags=re.DOTALL
#                         ).group(1)
#                         if self.run_name is None:
#                             self.run_name = fname
#                         if self.run_id is None:
#                             self.run_id = int(fname.group(2))
#                         self.filesdict[f"o{filetype}"] = get_sim_fname(
#                             self.run_id + 1, self.run_name, filetype
#                         )
#                 else:
#                     raise ValueError(
#                         f"Either file names or run_name must be specified."
#                     )
#         self.outdir = WDIR / f"{self.run_name}_{self.run_id:02d}"
#         if not self.outdir.is_dir():
#             os.mkdir(self.outdir)
#         self.init_array()
#         self.check_files()
#
#     def check_files(self):
#         for filetype, filename in self.filesdict.items():
#             try:
#                 filename = pl.Path(filename)
#             except TypeError:
#                 raise_sys_exit(f'"{filename}" is not a valid path argument.\n')
#             else:
#                 if not filename.parents[1] == WDIR:
#                     raise_sys_exit(
#                         f'"{filename.name}" must be in a folder in '
#                         f'"{WDIR.absolute()}".\n'
#                     )
#                 if filetype[0] == "i":
#                     if not filename.exists():
#                         raise FileNotFoundError(
#                             f'"{filename.name}" does not exist.'
#                         )
#                     else:
#                         self.filesdict[filetype] = filename
#                 else:
#                     if not filename.parent.is_dir():
#                         logger.info(
#                             f"Creating output directory {filename.parent}\n"
#                         )
#                         os.mkdir(filename.parent)
#
#     @property
#     def topstr(self):
#         with open(filesdict["itop"], "r") as topfile:
#             topstr = topfile.read()
#         return topstr
#
#     @cached_property
#     def top_iSL_regex_pattern(self):
#         re.compile(r"(?<=iSL)(\s+)(\d+)", flags=re.MULTILINE)
#
#     @property
#     def n_iSL(self):
#         return int(self.top_iSL_regex_pattern.search(self.topstr).group(2))
#
#     @property
#     def n_IL(self):
#         return len(self.top_iSL_regex_pattern.findall(self.topstr))
#
#     def init_array(self):
#         try:
#             with open(SPACINGS_FILENAME, "rb") as array_pickle:
#                 self.spacings_array = pkl.load(array_pickle)
#             logger.info(
#                 f'Opening d-spacing datafile "{SPACINGS_FILENAME.name}".'
#             )
#         except FileNotFoundError:
#             self.spacings_array_exists = False
#         else:
#             self.spacings_array_exists = True
#
#     def run(self):
#         logger.info(f"Topology contains {self.n_iSL} iSL solvent residues.\n")
#
#         topstr = self.top_iSL_regex_pattern.sub(
#             rf"\1 {n_iSL-n_wat_to_remove}", topstr
#         )
#         logger.info(
#             f'Removing {self.remove_waters} iSL residues from "{self.filesdict["itop"].name}"\n.'
#         )
#         shutil.copy2(self.filesdict["itop"], self.filesdict["otop"])
#         logger.info(
#             f'Writing new topology to "{self.filesdict["otop"].name}".\n'
#         )
#         u = mda.Universe(str(filesdict["igro"]), str(filesdict["itraj"]))
#         resnames = u.residues.resnames
#         clay = []
#         next_clay = False
#         prev_clay = False
#         n_ucs = 0
#         n_iSL = 0
#         n_iL = 0
#         for resid, resname in enumerate(resnames):
#             if resname in ["SOL", "iSL"]:
#                 next_clay = True
#                 if prev_clay:
#                     self.n_ucs = n_ucs
#                     n_ucs = 0
#                     if resname == "iSL":
#                         n_iL += 1
#                         n_iSL = 1
#                     else:
#                         next_clay = False
#                     prev_clay = False
#                 elif resname == "iSL":
#                     n_iSL += 1
#
#             elif re.match(r"[A-Z][a-z]?", resname):
#                 if prev_clay:
#                     self.n_ucs = n_ucs
#                     break
#             elif next_clay:
#                 n_ucs += 1
#                 clay.append(resname)
#                 prev_clay = True
#                 self.n_iSL_counted = n_iSL
#             else:
#                 pass
#
#         clay = u.select_atoms("not resname iSL SOL Cl Na Ca Mg K Cs")
#         sheet_list = []
#         remove_list = []
#         isl_residues = u.select_atoms("resname iSL").residues
#         clay_start = 0
#         for IL in range(n_IL):
#             start = IL * n_iSL
#             IL_resid_list = list(isl_residues[start : start + n_iSL].resids)
#             remove_list.extend(
#                 random.sample(IL_resid_list, int(n_wat_to_remove))
#             )
#             sheet_list.append(
#                 clay.select_atoms(f"resid {clay_start}:{IL_resid_list[0]}")
#             )
#             clay_start = IL_resid_list[0]
#         sheet_list.append(
#             clay.select_atoms(
#                 f"resid {IL_resid_list[-1]}:{clay.residues[-1].resid}"
#             )
#         )
#         sheet_list = list(map(lambda x: x.center_of_geometry()[2], sheet_list))
#         sheet_spacing = np.ediff1d(np.array(sheet_list))
#         sheet_spacing = np.mean(sheet_spacing)
#
#         logger.info(
#             f"Number of iSL waters per IL: {n_iSL}\n"
#             f"Average d-spacing: {np.round(sheet_spacing, 3)} A\n"
#         )
#
#         spacings_array_line = np.array([n_iSL, sheet_spacing])
#
#         if spacings_array_exists is False:
#             spacings_array = spacings_array_line
#         else:
#             spacings_array = np.vstack((spacings_array, spacings_array_line))
#
#         with open(SPACINGS_FILENAME, "wb") as array_pickle:
#             logger.info(f"Writing {spacings_array} to {SPACINGS_FILENAME}")
#             pkl.dump(spacings_array, array_pickle)
#
#         removestr = " ".join(list(map(lambda x: str(x), remove_list)))
#         new_u = u.select_atoms(f"not resid {removestr}")
#         new_u.write(filesdict["ogro"])
#         new_u.write(filesdict["otraj"], frames=u.trajectory[-1:])
#         logger.info(
#             f'Removing iSL residues {removestr} from "{filesdict["igro"].name}" and '
#             f'writing to "{filesdict["ogro"].name} and {filesdict["otraj"].name}".\n'
#         )
#
#         sys.exit(int(n_iSL))
#
#
# try:
#     n_run = int(sys.argv[1])
#     run_name = sys.argv[2]
#     n_wat_to_remove = int(sys.argv[3])
# except IndexError:
#     raise_sys_exit()
# except TypeError:
#     raise_sys_exit(
#         f"{sys.argv[1]} must be an integer specifying the simulation "
#         "run number\n"
#         f"{sys.argv[2]} must be an string specifying the simulation "
#         "run name\n"
#         f"{sys.argv[3]} must be an integer specifying "
#         "the number of water molecules to remove.i\n"
#     )
#
#
# filesdict = {
#     "igro": get_sim_fname(n_run - 1, run_name, "gro"),
#     "ogro": get_sim_fname(n_run, run_name, "gro"),
#     "itop": get_sim_fname(n_run - 1, run_name, "top"),
#     "otop": get_sim_fname(n_run, run_name, "top"),
#     "itraj": get_sim_fname(n_run - 1, run_name, "trr"),
#     "otraj": get_sim_fname(n_run, run_name, "trr"),
# }
#
# outdir = WDIR / f"{run_name}_{n_run:02d}"
# if not outdir.is_dir():
#     os.mkdir(outdir)
#
# try:
#     with open(SPACINGS_FILENAME, "rb") as array_pickle:
#         spacings_array = pkl.load(array_pickle)
#     logger.info(f'Opening d-spacing datafile "{SPACINGS_FILENAME.name}".')
# except FileNotFoundError:
#     spacings_array_exists = False
# else:
#     spacings_array_exists = True
#
# try:
#     logger.info("Checking file path arguments:")
#     for file in filesdict:
#         filesdict[file] = pl.Path(filesdict[file])
#         logger.info(f"{pl.Path(filesdict[file]).name}")
# except IndexError:
#     raise_sys_exit(f'"{filesdict[file]}" is not a valid path argument.\n')
#
# try:
#     for file in filesdict:
#         if not filesdict[file].parents[1] == WDIR:
#             raise_sys_exit(
#                 f'"{filesdict[file].name}" must be in a folder in '
#                 f'"{WDIR.absolute()}".\n'
#             )
#         if not filesdict[file].parent.is_dir():
#             logger.info(
#                 f"Creating output directory {filesdict[file].parent}\n"
#             )
#             os.mkdir(filesdict[file].parent)
# except IndexError:
#     raise_sys_exit(
#         f'"{filesdict[file].name}" must be the path of a filename '
#         f'within in a folder in "{WDIR.absolute()}".\n'
#     )
#
# with open(filesdict["itop"], "r") as topfile:
#     topstr = topfile.read()
#
# re_top_iSL = re.compile(r"(?<=iSL)(\s+)(\d+)")
# n_iSL = int(re_top_iSL.search(topstr).group(2))
# n_IL = len(re_top_iSL.findall(topstr))
#
# logger.info(f"Topology contains {n_iSL} iSL solvent residues.\n")
#
#
# topstr = re_top_iSL.sub(rf"\1 {n_iSL-n_wat_to_remove}", topstr)
# logger.info(
#     f'Removing {n_wat_to_remove} iSL residues from "{filesdict["itop"].name}"\n.'
# )
# with open(filesdict["otop"], "w") as topfile:
#     logger.info(f'Writing new topology to "{filesdict["otop"].name}".\n')
#     topfile.write(topstr)
# u = mda.Universe(str(filesdict["igro"]), str(filesdict["itraj"]))
#
# clay = u.select_atoms("not resname iSL SOL Cl Na Ca Mg K Cs")
# sheet_list = []
# remove_list = []
# isl_residues = u.select_atoms("resname iSL").residues
# clay_start = 0
# for IL in range(n_IL):
#     start = IL * n_iSL
#     IL_resid_list = list(isl_residues[start : start + n_iSL].resids)
#     remove_list.extend(random.sample(IL_resid_list, int(n_wat_to_remove)))
#     sheet_list.append(
#         clay.select_atoms(f"resid {clay_start}:{IL_resid_list[0]}")
#     )
#     clay_start = IL_resid_list[0]
# sheet_list.append(
#     clay.select_atoms(f"resid {IL_resid_list[-1]}:{clay.residues[-1].resid}")
# )
# sheet_list = list(map(lambda x: x.center_of_geometry()[2], sheet_list))
# sheet_spacing = np.ediff1d(np.array(sheet_list))
# sheet_spacing = np.mean(sheet_spacing)
#
# logger.info(
#     f"Number of iSL waters per IL: {n_iSL}\n"
#     f"Average d-spacing: {np.round(sheet_spacing, 3)} A\n"
# )
#
# spacings_array_line = np.array([n_iSL, sheet_spacing])
#
# if spacings_array_exists is False:
#     spacings_array = spacings_array_line
# else:
#     spacings_array = np.vstack((spacings_array, spacings_array_line))
#
# with open(SPACINGS_FILENAME, "wb") as array_pickle:
#     logger.info(f"Writing {spacings_array} to {SPACINGS_FILENAME}")
#     pkl.dump(spacings_array, array_pickle)
#
#
# removestr = " ".join(list(map(lambda x: str(x), remove_list)))
# new_u = u.select_atoms(f"not resid {removestr}")
# new_u.write(filesdict["ogro"])
# new_u.write(filesdict["otraj"], frames=u.trajectory[-1:])
# logger.info(
#     f'Removing iSL residues {removestr} from "{filesdict["igro"].name}" and '
#     f'writing to "{filesdict["ogro"].name} and {filesdict["otraj"].name}".\n'
# )
#
# sys.exit(int(n_iSL))
#
# if __name__ == "__main__":
#     args = parser.parse_args(sys.argv[1:])
#     rmwat = RMWat(args.i, args.w, args.n, args.c, args.t)
#     rmwat.run()

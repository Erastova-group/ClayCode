#!/usr/bin/env python3
import logging
import os
import pathlib as pl
import re
import shutil
import tempfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Literal, Optional, Tuple, Union

import ClayCode.addmols.ph as ph
import ClayCode.core.gmx as gmx
import ClayCode.core.lib as lib
import ClayCode.core.utils as ut
import MDAnalysis as mda
import numpy as np
from ClayCode.builder import UCData
from ClayCode.builder.topology import TopologyConstructor
from ClayCode.core.parsing import AddMolsArgs
from ClayCode.data.consts import AA, DATA, FF, MDP

__all__ = [
    "check_insert_numbers",
    "get_insert_charge",
    "init_outpaths",
    "run_em",
    "add_aa",
]

from ClayCode.core.classes import SimDir
from ClayCode.core.lib import (
    add_resnum,
    center_clay_universe,
    rename_il_solvent,
)

logger = logging.getLogger(Path(__file__).stem)


def check_insert_numbers(
    add_repl: Literal["Added", "Replaced"], searchstr: str
) -> int:
    """Get number of inserted/replaced molecules.

    :param add_repl: specify if molecules search for added or replaced molecules
    :type add_repl: 'Added' or 'Replaced'
    :param searchstr: string to be searched
    :type searchstr: str
    :return: number of added or replaced molecules
    :rtype: int or None if no match found

    """
    return int(
        re.search(
            rf".*{add_repl} (\d+)", searchstr, flags=re.MULTILINE | re.DOTALL
        ).group(1)
    )


def get_insert_charge(
    itp_path: Union[str, pl.Path], include_dir: Union[str, pl.Path]
) -> Union[int, None]:
    """Find charge of amino acid structure in itp_path.
    Uses charge information from force field in include_dir.

    :param itp_path: amino acid structure path
    :type itp_path: str or Path
    :param include_dir: force field directory
    :type include_dir: str or path
    :return: amino acid structure charge
    :rtype: int or ValueError for non integer charge
    """
    itp_path = str(itp_path)
    include_dir = str(include_dir)
    insert = mda.Universe(
        itp_path, topology_format="ITP", include_dir=include_dir
    )
    charge = np.sum(insert.atoms.charges)
    if np.round(charge, 3) != np.rint(charge):
        raise ValueError("Amino acid residue has non-integer charge!")
    else:
        charge = np.rint(charge)
    return int(charge)


def init_outpaths(
    outpath: Path,
    crdout: str,
    crdin: Union[str, Path],
    topin: Union[str, Path],
) -> Tuple[Path, Path]:
    """
    Get output .gro and .top file names and create directory.
    :param outpath: Output directory name
    :type outpath: Union[Path, str]
    :param crdout: Output coordinate filename
    :type crdout: Union[Path, str]
    :param crdin: Input coordinate filename
    :type crdin: Union[Path, str]
    :param topin: input topology filename
    :type topin: Union[Path, str]
    :return: .gro, .top filenames
    :rtype: Path, Path
    """
    if not outpath.is_dir():
        os.makedirs(outpath)
    else:
        ut.remove_files(outpath, "#*")
    crdout = outpath / Path(crdout).with_suffix(".gro").name
    topout = crdout.with_suffix(".top")
    shutil.copyfile(crdin, crdout)
    shutil.copyfile(topin, topout)
    assert crdout.is_file() and topout.is_file()
    return crdout, topout


def run_em(
    mdp: str,
    crdin: Union[str, Path],
    topin: Union[str, Path],
    tpr: Union[str, Path],
    odir: Path,
    outname: str = "em",
) -> Union[str, None]:
    """
    Run an energy minimisation using gmx and
    return convergence information if successful.
    :param mdp: mdp parameter file path
    :type mdp: Union[Path, str]
    :param crdin: Input coordinate file name
    :type crdin: Union[Path, str]
    :param topin: In put topology file name
    :type topin: Union[Path, str]
    :param tpr: Output tpr file name
    :type tpr: Union[Path, str]
    :param odir: Output directory path
    :type odir: Path
    :param outname: Default stem for output files
    :type outname: str
    :return: Convergence information message
    :rtype: Union[str, None]
    """
    if not pl.Path(mdp).is_file():
        mdp = MDP / mdp
        assert mdp.is_file()
    logger.info("# MINIMISING ENERGY")
    outname = (Path(odir) / outname).resolve()
    otop = ut.change_suffix(outname, "top")
    if topin.resolve() == otop.resolve():
        otop = otop.parent / f"{otop.stem}_temp.top"
        otop_copy = True
    else:
        otop_copy = False
    _, em = gmx.run_gmx_grompp(
        f=mdp,
        c=crdin,
        p=topin,
        o=tpr,
        pp=otop,
        v="",
        po=ut.change_suffix(tpr, "mdp"),
    )
    err = re.search(r"error", em)
    logger.info(em)
    if err is None:
        _, em = gmx.run_gmx_mdrun(s=tpr, deffnm=outname)
        conv = re.search(
            r"converged to Fmax < (\d+) in (\d+) steps",
            em,
            flags=re.MULTILINE | re.DOTALL,
        )
        if conv is None:
            logger.info(em)
        assert conv is not None, "Energy minimisation run not converged!"
        fmax, n_steps = conv.groups()
        logger.info(f"Fmax: {fmax}, reached in {n_steps} steps")
        logger.info(f"Output written to {outname!r}")
        conv = (
            f"Fmax: {fmax}, reached in {n_steps} steps."
            f"Output written to {outname!r}"
        )
        if otop_copy is True:
            shutil.move(otop, topin)
    else:
        logger.info(f"\n{em}")
        raise RuntimeError("gmx grompp raised error!")
    return conv


def add_aa(
    data: AddMolsArgs,
    aa: Union[str, List[str]],
    clay_type: str,
    conc: Union[int, float],
    pion: str,
    nion: str,
    odir: Union[str, pl.Path],
    crdin: Union[str, pl.Path],
    pH: Union[int, float, List[Union[int, float]]],
    ff_path: Union[str, pl.Path] = FF,
    topin: Optional[Union[str, pl.Path]] = None,
    posfile: Union[str, pl.Path] = "pos.dat",
    rm_tempfiles: bool = True,
    overwrite_aa_numbers: bool = True,
    new_outpath: Union[bool, str, Path] = True,
    overwrite: bool = False,
    outname_suffix: Optional[str] = None,
):
    # initialise output directory
    odir = pl.Path(odir).resolve()
    while odir.stem == "setup":
        odir = odir.parent
    odir = odir / "setup"
    # get list of amino acids to add
    if isinstance(aa, str):
        aa = [aa]
    if aa is not None:
        aa = list(aa)
    aa_str = "_".join(aa)
    if not odir.is_dir():
        odir = odir.parent
        os.makedirs(odir, exist_ok=True)
    if len(list(odir.glob("*.gro"))) != 0 and overwrite is False:
        logger.info(
            f"Skipping {aa_str.upper()}, directory {odir} already exists."
        )
        out = SimDir(odir, check=False)
        crdout, topout = out.gro, out.top
    else:
        # get temporary directory
        # tmpdir = odir / ".tmp"
        tmpdir = TemporaryDirectory(
            suffix=None, prefix=".tmp", dir=odir.parent
        )
        tmppath = Path(tmpdir.name)
        tmpfile = tempfile.NamedTemporaryFile(
            prefix="tmpfile", dir=tmppath, delete=False
        )
        logger.info(f"Setting up temporary directory {tmppath.name!r}")
        # get pH value
        pH = ut.get_first_item_as_int(pH)
        # set default output name
        outname = f"{aa_str}_{pH}"
        if outname_suffix is not None:
            outname += f'_{outname_suffix.strip("_")}'
        outname = Path(outname)
        # add logfile
        logger.set_file_name(new_filepath=tmppath.parent, new_filename=aa[0])
        # fhandler = logging.FileHandler(
        #     filename=ut.get_logfname(
        #         logname=logger.name, run_name=aa[0], logpath=tmppath.parent
        #     ),
        #     mode="w",
        # )
        # # add file logging
        # logger.addHandler(fhandler)
        logger.finfo(f"# RUNNING {Path(__file__).stem!r}")
        # define input and data paths
        crdin = pl.Path(crdin)
        if topin is None:
            topin = crdin.with_suffix(".top")
        ff_path = pl.Path(ff_path)
        logger.info(f"gro = {str(crdin.resolve())!r}")
        logger.info(f"top = {str(topin.resolve())!r}")
        logger.info(f"FF = {str(ff_path.resolve())!r}")
        # else:
        # center atoms in coordinate file
        centered_crd = tmppath / "center.gro"
        u = center_clay_universe(
            crdname=str(crdin), crdout=centered_crd, uc_name=clay_type
        )
        assert centered_crd.is_file()
        crdout, topout = init_outpaths(
            outpath=tmppath,
            crdin=centered_crd,
            crdout=outname.name,
            topin=topin,
        )
        # print(np.unique(u.residues.resnames))
        conc /= 1000
        n_mols = lib.get_n_mols(conc=conc, u=u)
        # determine aa species to be added at pH
        conc_dict = ph.get_aa_numbers(
            aa=aa,
            pH=pH,
            totmols=n_mols,
            o=AA / "aa_numbers.pkl",
            new=overwrite_aa_numbers,
        )
        # get total number of AA to add
        n_mols = np.sum(np.array([*conc_dict.values()], dtype=int))
        logger.info(f"pH = {pH}, conc = {conc} mol L-1, n_mols = {n_mols}")
        # add AA if n_mols != 0
        if n_mols != 0:
            logger.info(f"# INSERTING MOLECULES:")
            logger.info(f"Target concentration: {conc} mol L-1")
            logger.info(f"pH: {pH}")
            logger.info(f"Inserting {n_mols} amino acid residues")
            # add all AA species
            for aa_species in conc_dict:
                logger.info(f"AA species: {aa_species.upper()}")
                inserted = []
                replaced_sol = 0
                logger.info(f"Inserting {aa_str.upper()}:")
                aa_gros = sorted(
                    AA.glob(rf"pK[1-4]/{aa_species.upper()}[1-4].gro")
                )
                aa_itps = sorted(
                    AA.glob(rf"pK[1-4]/{aa_species.upper()}[1-4].itp")
                )
                if not len(aa_gros) == len(conc_dict[aa_species]):
                    logger.debug(f"{aa_gros} != {conc_dict[aa_species]}!")
                    raise ValueError(
                        f"Number of found files ({len(aa_gros)}, {len(aa_itps)}) does not match "
                        f"specified number of amino acid conformations {len(conc_dict[aa_species])}."
                    )
                insert_charges = 0
                aa_charge_dict = lib.get_aa_charges(aa_name=aa_species)
                gmx_commands = gmx.GMXCommands()
                # add all AA species
                for prot_id, prot_species in enumerate(conc_dict[aa_species]):
                    prot_species = int(prot_species)
                    if prot_species != 0:
                        logger.info(
                            f"Inserting {prot_species} {aa_species.upper()} molecules from pK{prot_id + 1}"
                        )
                        # determine positions for adding AA
                        posfile = tmppath / posfile
                        lib.write_insert_dat(n_mols=prot_species, save=posfile)
                        assert posfile.is_file()
                        (
                            insert_out,
                            insert_err,
                        ) = gmx_commands.run_gmx_insert_mols(
                            f=crdout,
                            ci=aa_gros[prot_id],
                            ip=posfile,
                            nmol=prot_species,
                            o=crdout,
                            replace="SOL",
                            dr="10 10 1",
                        )
                        u = mda.Universe(str(crdout))
                        assert crdout.is_file()
                        replace_check = check_insert_numbers(
                            add_repl="Added", searchstr=insert_out
                        )
                        if replace_check != prot_species:
                            raise ValueError(
                                f"Number of inserted molecules ({replace_check}) does not match target number "
                                f"({prot_species})!"
                            )
                        else:
                            inserted.append(
                                f"{aa_species.upper()}{prot_id + 1}  {replace_check}"
                            )
                            insert_charge = aa_charge_dict[
                                f"{aa_species.upper()}{prot_id + 1}"
                            ]
                            # add to total aa charge
                            insert_charges += insert_charge * replace_check
                            replaced_sol += check_insert_numbers(
                                add_repl="Replaced", searchstr=insert_out
                            )
                            logger.info(
                                f"pk{prot_id + 1}: {prot_species} {aa_species.upper()}{prot_id + 1} (q = {insert_charge * replace_check})"
                            )
                    else:
                        logger.info(
                            f"Not inserting {aa_species} mols from pK{prot_id + 1}"
                        )
            uc_data = UCData(
                path=DATA, uc_stem=clay_type, ff=data.ff, write=False
            )
            topology = TopologyConstructor(uc_data, data.ff, gmx_commands)
            lib.add_mol_list_to_top(
                topin=topout,
                topout=topout,
                insert_list=inserted,
                ff_path=ff_path,
            )
            # remove replaced solvent molecules from topology file
            lib.remove_replaced_SOL(
                topin=topout, topout=topout, n_mols=replaced_sol
            )
            logger.info(f"Replaced {replaced_sol} SOL molecules")
            u = mda.Universe(
                crdout
            )  # if necessary, rename interlayer SOL to iSL  # lib.fix_gro_residues(crdin=crdout, crdout=crdout)  # # neutralise system charge  # replaced = lib.neutralise_system(  #     odir=tmppath,  #     crdin=crdout,  #     topin=topout,  #     topout=topout,  #     pion=pion,  #     nion=nion,  # )  # if replaced is not None:  #     logger.info(f"Replaced {replaced} SOL molecules.")  # u = mda.Universe(str(crdout))  # # print(np.unique(u.residues.resnames))  # # run energy minimisation  # _ = run_em(  #     mdp="em.mdp",  #     crdin=crdout,  #     topin=topout,  #     tpr=tmppath / "em.tpr",  #     odir=tmppath,  #     outname=outname,  # )  # u = mda.Universe(str(crdout))  # print(np.unique(u.residues.resnames))  # lib.fix_gro_residues(crdin=crdout, crdout=crdout)  # # write final coordinates and topology to output directory  # crdout, topout = ut.copy_final_setup(  #     outpath=odir, tmpdir=tmppath, rm_tempfiles=False  # )
        else:
            # setup for system without aa
            logger.info("Not adding any molecules.")
            lib.add_mol_list_to_top(
                topin=topout, topout=topout, insert_list=[], ff_path=ff_path
            )
        # if necessary, rename interlayer SOL to iSL
        add_resnum(crdin=crdout, crdout=crdout)
        rename_il_solvent(crdin=crdout, crdout=crdout)
        # lib.fix_gro_residues(crdin=crdout, crdout=crdout)
        # neutralise system charge
        replaced = lib.add_ions_neutral(
            odir=tmppath,
            crdin=crdout,
            topin=topout,
            pion=pion,
            nion=nion,
            gmx_commands=gmx_commands,
        )
        if replaced is not None:
            logger.info(f"Replaced {replaced} SOL molecules.")
        u = mda.Universe(str(crdout))
        # print(np.unique(u.residues.resnames))
        # run energy minimisation
        _ = run_em(
            mdp="em.mdp",
            crdin=crdout,
            topin=topout,
            tpr=tmppath / "em.tpr",
            odir=tmppath,
            outname=outname,
        )
        # lib.fix_gro_residues(crdin=crdout, crdout=crdout)
        add_resnum(crdin=crdout, crdout=crdout)
        rename_il_solvent(crdin=crdout, crdout=crdout)
        # write final coordinates and topology to output directory
        crdout, topout = ut.copy_final_setup(
            outpath=odir, tmpdir=tmppath, rm_tempfiles=False
        )
    if (type(crdout) == pl.PosixPath and crdout.is_file()) and (
        type(topout) == pl.PosixPath and topout.is_file()
    ):
        return (crdout, topout)
    else:
        return (None, None)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        prog="AAadd",
        description="Add amino acid to coordinate and topology files.",
        add_help=True,
        allow_abbrev=False,
    )

    parser.add_argument(
        "-sysdir", type=pl.Path, help="System directory", metavar="sysdir"
    )
    parser.add_argument(
        "-odir",
        type=pl.Path,
        help="Output directory",
        metavar="odir",
        default=None,
    )

    parser.add_argument(
        "-inp",
        type=str,
        help="Path stem for input .gro and .top files.",
        metavar="inp",
    )

    parser.add_argument(
        "-conc",
        type=np.float_,
        metavar="conc",
        help="Amino acid concentration in mmol L-1",
    )

    parser.add_argument(
        "-uc", type=str, metavar="unit_cell", help="Clay unit cell type"
    )

    parser.add_argument(
        "-pion",
        type=str,
        metavar="pion",
        help="Charge balancing positive ion species in bulk.",
    )

    parser.add_argument(
        "-nion",
        type=str,
        metavar="nion",
        help="Charge balancing positive ion species in bulk.",
        default="Cl",
    )

    parser.add_argument(
        "-ff",
        type=str,
        metavar="force-field",
        help="Force field directory",
        default=FF,
    )

    parser.add_argument(
        "-aa",
        type=str,
        nargs="+",
        default=None,
        metavar="aa",
        help="Amino acid(s)",
    )

    parser.add_argument(
        "-aa-dir",
        type=str,
        metavar="AA_dir",
        help=f'Directory with AA structure folders names {"pK1"!r}-{"pK4"!r}',
        default=AA,
    )

    parser.add_argument(
        "-pH",
        type=np.float64,
        nargs=1,
        default=7.0,
        metavar="pH_value",
        help="pH for experiment",
    )

    for i in ["Ca", "Na", "K", "Mg"]:
        for idx in [1, 2]:
            a = parser.parse_args(
                (
                    f"-aa "
                    f"ala "  # 1
                    f"arg "  # 2
                    f"asn "  # 3
                    f"asp "  # 4
                    f"cys "  # 5
                    f"gln "  # 6
                    f"glu "  # 7
                    f"gly "  # 8
                    f"his "  # 9
                    f"ile "  # 10
                    f"leu "  # 11
                    f"lys "  # 12
                    f"met "  # 13
                    f"phe "  # 14
                    f"pro "  # 15
                    f"ser "  # 16
                    f"thr "  # 17
                    f"trp "  # 18
                    f"tyr "  # 19
                    f"val "  # 20
                    + f"-conc 250 -uc T2 -inp ctl_7 -pion {i} -nion Cl -ff /storage/claycode/package/ClayCode/data/data/FF "
                    f"-sysdir /media/hannahpollak/free/new_NAu/{i}/NAu-{idx}-fe/  "
                    f"-pH 7"
                ).split()
            )
            for aa in a.aa:
                add_aa(
                    aa=[aa],
                    clay_type=a.uc,
                    conc=a.conc,
                    pion=a.pion,
                    nion=a.nion,
                    crdin=a.sysdir / f"{a.inp}.gro",
                    odir=Path(f"{aa}_7/"),
                    pH=a.pH,
                    ff_path=a.ff,
                    posfile="pos.dat",
                    new_outpath=False,
                    overwrite=False,
                )
            add_aa(
                aa=["ctl"],
                clay_type=a.uc,
                conc=0,
                pion=a.pion,
                nion=a.nion,
                crdin=a.sysdir / f"{a.inp}.gro",
                odir=Path(f"{a.sysdir}/ctl_7/"),
                pH=a.pH,
                ff_path=a.ff,
                posfile="pos.dat",
                new_outpath=False,
                overwrite=False,
            )

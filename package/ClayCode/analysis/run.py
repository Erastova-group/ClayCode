import logging
import sys
from argparse import ArgumentParser
from pathlib import Path

from ClayAnalysis.checks import RunChecks

check_logger = logging.getLogger(Path(__file__).stem)
check_logger.setLevel(level=logging.INFO)


parser = ArgumentParser("run", allow_abbrev=False)
subparsers = parser.add_subparsers(title="subparser_name")
check_parser = subparsers.add_parser("-checks")
check_parser.add_argument(
    "-p",
    type=str,
    help="data directory path",
    metavar="root_dir",
    dest="root_dir",
    required=True,
)
check_parser.add_argument(
    "-a",
    type=str,
    help="alternative data directory path",
    metavar="alt_root",
    dest="alt_root",
    default=None,
)
check_parser.add_argument(
    "-t",
    type=int,
    help="expected trajectory length",
    metavar="traj_len",
    dest="traj_len",
)
check_parser.add_argument(
    "-w",
    type=str,
    help="write results to text files to directory",
    metavar="write_dir",
    dest="write_dir",
    default=None,
)
check_parser.add_argument(
    "-s",
    type=str,
    help="Subfolder name",
    metavar="subdir",
    dest="subdir",
    default=None,
)
check_parser.add_argument(
    "-largest_traj",
    action="store_const",
    const="largest",
    default="latest",
    help="Select largest trajectory file instead of most recent",
    dest="trr_sel",
)
check_parser.add_argument(
    "-fix",
    action="store_true",
    default=False,
    required=False,
    help="Apply fixes for failed run checks.",
    dest="fix",
)

check_parser.add_argument(
    "-update",
    action="store_true",
    default=False,
    required=False,
    help="Update even if a dataframe pickle exists for [-in].",
    dest="update",
)

check_parser.add_argument(
    "-in",
    default=None,
    required=False,
    help="Load dataframe from pickle or csv.",
    dest="load",
)
check_parser.add_argument(
    "-out",
    default=None,
    required=False,
    help="Save check data in a directory.",
    dest="savedest",
)
check_parser.add_argument(
    "-rsync",
    default=None,
    required=False,
    help="Copy completed simulation data.",
    dest="rsync",
)
# analysis_parser = subparsers.add_parser('-analysis')
# analysis_choices = {'zdist': None, 'veldist': None, 'multidist': None}
# analysis_parser.add_argument("-n",
#                              type=str,
#                              help="Analysis type selection",
#                              nargs='+',
#                              choices=[*analysis_choices.keys()],
#                              dest='name')
# a_subparsers = analysis_parser.add_subparsers(dest='p_name')
# for analysis_type in analysis_choices.keys():
#     analysis_choices[analysis_type] = a_subparsers.add_parser(f'-analysis_type',
#                                                               parents=[ClayAnalysis.zdist.parser],
#                                                               add_help=False)
if __name__ == "__main__":
    args = check_parser.parse_args(sys.argv[1:])

    r = RunChecks(
        root_dir=args.root_dir, alt_dir=args.alt_root, data_df_file=args.load
    )
    r.run(
        traj_len=args.traj_len, savedest=args.savedest
    )  # , update=args.update)#, rsync_results_file='/storage/sim_data')
    if args.savedest is not None:
        r.save(savedest=args.savedest)
    if args.rsync is not None:
        try:
            rsync_dest = Path(args.rsync).resolve()
            r.write_complete_rsync_script(
                dest=rsync_dest,
                complete_df_json=list(
                    Path(args.savedest).glob("complete*.json")
                ),
                outname=Path(args.savedest) / "rsync_complete.sh",
            )
        except NotImplementedError:
            print("Invalid name for rsync destination")
    if args.fix is True:
        r.fix(
            odir=None,
            rm_tempfiles=True,
            overwrite=False,
            datpath=args.savedest,
        )

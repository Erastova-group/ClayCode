from argparse import ArgumentParser
from config.classes import File

__name__ = 'parser'

__all__ = ['parser']

parser = ArgumentParser('ClayCode', description='Automatically generate atomistic clay models.', add_help=True,
                        allow_abbrev=False)

parser.add_argument('build',
                    required=False,
                    default=False,
                    nargs=0,
                    action='store_true',
                    dest='BUILD')

parser.add_argument('siminp',
                    required=False,
                    default=False,
                    action='store_true',
                    dest='SIMINP')

parser.add_argument('-f',
                    type=File,
                    help='YAML file with input parameters',
                    metavar='yaml_file',
                    dest='PRM_FILE')

parser.add_argument('-comp',
                    type=File,
                    help='CSV file with clay composition',
                    metavar='csv_file',
                    dest='CLAY_COMP')


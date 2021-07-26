import argparse


def init_options():
    parser = argparse.ArgumentParser(
        usage="%(prog)s [command] ...",
        description="Utilidades para testear ETS2 dataset"
    )

    subparser = parser.add_subparsers(dest='command')
    compare_depths(subparser)
    transform_ets2_depth(subparser)
    dataset_tools(subparser)

    return parser


def compare_depths(subparser):
    parser = subparser.add_parser('compare',
                                  description="Compares two depth .npy files")

    parser.add_argument('file1')
    parser.add_argument('file2')
    return parser


def transform_ets2_depth(subparser):
    parser = subparser.add_parser('dump',
                                  description="Generate png image from depth file")

    parser.add_argument('file')
    parser.add_argument('--format', type=str, choices=['ets2', 'monodepth2'], default='ets2')


def dataset_tools(subparser):
    parser = subparser.add_parser('dataset',
                                  description="Dataset commands")

    parser.add_argument('path',
                        help='Path where dataset is stored')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--split',
                       help='Generate split files for training with dataset',
                       type=int,
                       default=None,
                       const=0.95,
                       nargs='?')
    group.add_argument('--stats',
                       help='Generate dataset stats',
                       action='store_true')


def color_tools(subparser):
    parser = subparser.add_parser('color',
                                  description="Color files commands")
    parser.add_argument("file",
                        help="File or directory to process",
                        type=str,
                        required=True)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('fix-header',
                       help='Fix bmp header')
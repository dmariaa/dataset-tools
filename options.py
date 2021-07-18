import argparse


def init_options():
    parser = argparse.ArgumentParser(
        usage="%(prog)s [command] ...",
        description="Utilidades para testear ETS2 dataset"
    )

    subparser = parser.add_subparsers(dest='command')
    compare_depths(subparser)
    transform_ets2_depth(subparser)

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

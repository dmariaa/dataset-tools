import argparse
import glob
import os.path
import sys
import time

import numpy as np
import rawutil


def create_options(parser):
    subparsers = parser.add_subparsers(title='Color files commands',
                                       dest='command',
                                       required=True)

    fix_subparser = subparsers.add_parser('fix', help='Fix color file header')
    fix_subparser.add_argument('files',
                               help='File or directory with .BMP files to fix',
                               type=str)


def fix_bmp_header(args):
    files_arg = args.files

    if os.path.isdir(files_arg):
        files = glob.glob(f"{files_arg}/**/*.bmp", recursive=True)
    else:
        files = [files_arg]

    for file in files:
        f = open(file, 'r+b')
        f.seek(2)
        header_size = rawutil.unpack('<l', f.read(4))[0]
        size = os.fstat(f.fileno()).st_size

        if header_size != size:
            print(f"Fixing: {file} size ({size}) does not match header size ({header_size})")
            f.seek(2)
            f.write(rawutil.pack('<l', size))
            f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='color')
    create_options(parser)
    args = parser.parse_args()

    tic = time.perf_counter()

    if args.command == 'fix':
        fix_bmp_header(args)
    else:
        parser.print_help(sys.stdout)
        exit()

    toc = time.perf_counter()
    print(f"command {args.command} executed in {toc - tic:0.4f} seconds")

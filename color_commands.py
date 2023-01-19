import argparse
import glob
import os.path
import sys
import time

import numpy as np
import rawutil
import cv2
from PIL import Image

from depth_utils import read_depth_file, colorize


def create_options(parser):
    subparsers = parser.add_subparsers(title='Color files commands',
                                       dest='command',
                                       required=True)

    fix_subparser = subparsers.add_parser('fix', help='Fix color file header')
    fix_subparser.add_argument('files',
                               help='File or directory with .BMP files to fix',
                               type=str)

    video_subparser = subparsers.add_parser('video', help='Generate video sequence')
    video_subparser.add_argument('--path',
                                 help='Path to images folder',
                                 type=str,
                                 required=True)
    video_subparser.add_argument('--color-extension',
                                 help='Color files extension, one of "png", "bmp", "jpg" (default)',
                                 choices=['png', 'bmp', 'jpg'],
                                 default='jpg')
    video_subparser.add_argument('--width',
                                 help='Output with, (default 640)',
                                 type=int,
                                 default=640)
    video_subparser.add_argument('--height',
                                 help='Output height, (default 480)',
                                 type=int,
                                 default=480)
    video_subparser.add_argument('--init-frame',
                                 help='Initial frame (default 0)',
                                 type=int,
                                 default=0)
    video_subparser.add_argument('--number-of-frames',
                                 help='Number of frames to generate (default 100)',
                                 type=int,
                                 default=100)
    video_subparser.add_argument('--frame-rate',
                                 help='Frame rate for video output (default 10)',
                                 type=int,
                                 default=10)
    video_subparser.add_argument('--data-type',
                                 help='Data type, one of ets2 (default), monodepth2, densedepth',
                                 type=str,
                                 choices=['ets2', 'monodepth2', 'densedepth'],
                                 default='ets2')


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


def generate_video_frame(color_name, size):
    base_name = os.path.splitext(color_name)[0]

    # Read color image
    color_image = cv2.imread(color_name)

    # Read depth data
    depth_data = read_depth_file(f"{base_name}.depth.raw")
    original_size = (depth_data['header']['width'], depth_data['header']['height'], 3)
    depth_color = np.clip(depth_data['data'], 0, 80)
    depth_color = colorize(depth_color, colormap_name='magma', normalize_data=True, invert=True)
    depth_color = depth_color[:, :3].reshape((original_size[1], original_size[0], original_size[2]))
    depth_color = depth_color[:, :, ::-1]

    # img = Image.frombytes('RGBA', (width, height), depth_color)
    # depth_color_reshaped = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGR)

    # Concat and resize images
    img = cv2.vconcat([color_image, depth_color])
    img = cv2.resize(img, size, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
    return img


def mount_video_frame(color_name, size, data_type):
    (path, file_name) = os.path.split(color_name)
    base_name = os.path.splitext(file_name)[0]

    color_image = cv2.imread(color_name)

    if data_type == 'monodepth2':
        result = cv2.imread(os.path.join(path, 'monodepth2', f"{base_name}_disp.ets2.50000.jpeg"))
        img = cv2.vconcat([color_image, result])
    elif data_type == 'densedepth':
        result = cv2.imread(os.path.join(path, 'ets2-trainer', f"{base_name}.depth.jpg"))
        img = cv2.vconcat([color_image, result])
    else:
        img = color_image

    img = cv2.resize(img, size, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
    return img

def generate_video_sequence(args):
    color_files_path = os.path.join(args.path, f"*.{args.color_extension}")
    color_files = sorted(glob.glob(color_files_path))

    size = (args.width, args.height)
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_stream = cv2.VideoWriter('test.mp4', fourcc, args.frame_rate, size)

    for i, image_file in enumerate(color_files[args.init_frame:args.number_of_frames]):
        start = time.time()

        if args.data_type == 'ets2':
            img = generate_video_frame(image_file, size=size)
        else:
            img = mount_video_frame(image_file, size=size, data_type=args.data_type)

        out_stream.write(img)
        print(f"Generated frame {i}, img size {img.shape}, in {time.time() - start} seconds")

    out_stream.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    create_options(parser)
    args = parser.parse_args()

    tic = time.perf_counter()

    if args.command == 'fix':
        fix_bmp_header(args)
    elif args.command == 'video':
        generate_video_sequence(args)
    else:
        parser.print_help(sys.stdout)
        exit()

    toc = time.perf_counter()
    print(f"command {args.command} executed in {toc - tic:0.4f} seconds")

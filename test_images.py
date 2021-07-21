import os
import glob

from PIL import Image

import depth_utils as du


def linearize_depth(depth, min_value=0.0, max_value=1.0):
    min = depth.min()
    max = depth.max()
    n = (depth - min) / (max - min)  # normalize between 0 and 1
    n = n * (max_value - min_value) + min_value
    return n


def to_world_depth(depth, near=0.1, far=3000.0):
    p33 = far / (far - near)
    p43 = (-far * near) / (far - near)
    return (depth - p43) / p33


def set_zeros_to_background(data, max_value=1000.0):
    data[data == 0] = max_value
    return data

files = glob.glob(os.path.join("test-data", "*.bmp"))

for file in files:
    file_name = os.path.basename(file)
    depth_file_name = f"{os.path.splitext(file)[0]}.depth.raw"

    depth_file = du.read_depth_file(depth_file_name)
    transformed_data = depth_file['data']
    imdata = du.colorize(-set_zeros_to_background(depth_file['data']), colormap_name='magma', normalize_data=True)

    du.generate_histogram(imdata, f"{file_name} Depth Transformed", 50)

    size = (depth_file['header']['width'], depth_file['header']['height'])
    image = Image.frombytes('RGBA', size, imdata)
    file_data = os.path.splitext(file)
    image.save(file_data[0] + '.png')

    print(f"{file_name} => {depth_file['header']} ({transformed_data.min()}, {transformed_data.max()})")

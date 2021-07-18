import math
import os
import struct
import time

import matplotlib.pyplot as plt
import numpy as np
import rawutil
from PIL import Image
from matplotlib import cm
from sklearn import metrics

import options


def normalize(data):
    max = data.max()
    min = data.min()
    range = max - min if max != min else 1e10
    return (data - min) / range


def convert_to_linear(depth, near=0.1, far=3000.0):
    ratio = far / near
    return depth - near / ratio


def convert_to_worlddepth(depth, near=0.1, far=3000.0):
    p33 = far / (far - near)
    p43 = (-far * near) / (far - near)
    return p43 / (depth - p33)


def convert_to_worlddepth2(depth, near=0.1, far=3000.0):
    new_data = []

    sw = 1440
    sh = 816
    proj_matrix = np.array([
        [2 * near / sw, 0, 0, 0],
        [0, 2 * near / sh, 0, 0],
        [0, 0, far / (far - near), 1],
        [0, 0, (-far * near) / (far - near), 0]
    ])

    inv_proj_matrix = np.linalg.inv(proj_matrix)

    for x in depth:
        pos = [0, 0, x, 1]
        unproj_pos = inv_proj_matrix @ pos
        new_data.append(3000.0 - (-unproj_pos[2] / unproj_pos[3]) * 3000.0)

    return np.array(new_data)


def convert_to_colormap2(file, format, colormap_name):
    colormap = cm.get_cmap(colormap_name)
    img_src = Image.open('test-data/capture-0000001673.depth.raw').convert('L')
    img = np.array(img_src)
    img = colormap(img)
    img = np.uint8(img * 255)
    img = Image.fromarray(img)
    img.save('test2.png')


def generate_histogram(data, title, bins='auto'):
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.show()

def convert_to_colormap(file, format, colormap_name):
    colormap = cm.get_cmap(colormap_name)

    if format == 'ets2':
        depth_data = read_depth_file(file)
        data = depth_data['data']
        size = (depth_data['header']['width'], depth_data['header']['height'])
        data.shape = size
        imdata = colormap(1 - data)
        imdata = np.uint8(imdata * 255)
    else:
        file_data = get_npy_data(file)
        data = file_data.squeeze()
        vmax = np.percentile(data, 95)
        vmin = data.min()
        # data = mpl.colors.Normalize(vmin=data.min(), vmax=vmax)
        data = (data - vmin) / (vmax - vmin)
        size = (data.shape[1], data.shape[0])
        imdata = colormap(data)
        imdata = np.uint8(imdata * 255)

    image = Image.frombytes('RGBA', size, imdata)
    file_data = os.path.splitext(file)
    image.save(file_data[0] + '.png')
    plt.imshow(image)
    plt.show()


def read_depth_file(file):
    data = np.fromfile(file, dtype='byte')
    header = get_header(data)
    return {
        'header': header,
        'data': get_data(data, header)
    }


def get_header(data):
    header = {
        "magic": bytes(struct.unpack('bb', data[0:2])).decode('utf-8'),
        "size": struct.unpack('<l', data[2:6])[0],
        "width": struct.unpack('<l', data[6:10])[0],
        "height": struct.unpack('<l', data[10:14])[0],
        "min_val": struct.unpack('<f', data[14:18])[0],
        "max_val": struct.unpack('<f', data[18:22])[0],
        "offset": struct.unpack('<l', data[22:26])[0]
    }
    print(header)
    return header


def get_data(file_data, header):
    denorm = pow(2, 24) - 1
    format = '<%sU' % int((header['size'] - 26) / 3)
    data = np.array(rawutil.unpack(format, file_data[header['offset']:]))

    data = data / denorm

    # data = convert_to_worlddepth(origdata)
    print(f"ORIGINAL FILE DATA [{data.shape}]: min value {np.min(data)}, max value {np.max(data)}")
    generate_histogram(data, "ORIGINAL FILE DATA", bins='auto')

    tdata = convert_to_worlddepth2(data)
    data_range = np.abs(np.max(tdata) - np.min(tdata))
    print(f"TRANSFORMED DATA [{tdata.shape}]: min value {np.min(tdata)}, max value {np.max(tdata)}, {data_range}, {1 / np.max([np.min(tdata), 1e-100])}, {1 / np.max(tdata)}")
    generate_histogram(tdata, "TRANSFORMED DATA")

    minval = header['min_val']
    range = header['max_val'] - minval
    # data = normalize(data)
    # data = (data - minval) / range
    return tdata


def get_npy_data(file):
    data = np.load(file)
    return data


def compare_depths(file1, file2):
    data1 = np.load(file1).squeeze()
    data2 = np.load(file2).squeeze()
    # data1 = (data1 - data1.min()) / (data1.max() - data1.min())
    # data2 = (data2 - data2.min()) / (data2.max() - data2.min())
    print(f"File 1 shape: {data1.shape}")
    print(f"File 2 shape: {data2.shape}")
    mse = metrics.mean_squared_error(data1, data2)
    rmse = math.sqrt(mse)
    return (mse, rmse)


parser = options.init_options()
args = parser.parse_args()

tic = time.perf_counter()

if args.command == 'dump':
    convert_to_colormap(args.file, args.format, 'magma')
elif args.command == 'compare':
    result = compare_depths(args.file1, args.file2)
    print(result)

toc = time.perf_counter()
print(f"command {args.command} executed in {toc - tic:0.4f} seconds")

# file = sys.argv[1]
# tic = time.perf_counter()
# convert_to_colormap(file, 'magma')
# toc = time.perf_counter()
# print(f"Generated colormap raw in {toc - tic:0.4f} seconds")

# tic = time.perf_counter()
# convert_to_colormap2('test-data/capture-0000001673.depth.raw', 'magma')
# toc = time.perf_counter()
# print(f"Generated colormap PIL in {toc - tic:0.4f} seconds")

# test = read_depth_file(file)
# print(test['header'])
# print(len(test['data']))
# print(test['data'][0])

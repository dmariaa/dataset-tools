import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

telemetry_headers = [
    'capture',
    'p.x',
    'p.y',
    'p.z',
    'o.x',
    'o.y',
    'o.z',
    'l_vel.x',
    'l_vel.y',
    'l_vel.z',
    'a_vel.x',
    'a_vel.y',
    'a_vel.z',
    'l_acc.x',
    'l_acc.y',
    'l_acc.z',
    'a_acc.x',
    'a_acc.y',
    'a_acc.z'
]


def create_options(parser):
    subparsers = parser.add_subparsers(title='Dataset commands',
                                       dest='command',
                                       required=True)

    fix_subparser = subparsers.add_parser('split', help='Generate dataset split')
    fix_subparser.add_argument('path',
                               help='Path to the dataset',
                               type=str)

    video_subparser = subparsers.add_parser('stats', help='Generate stats')
    video_subparser.add_argument('path',
                                 help='Path to the dataset',
                                 type=str)


def generate_split(path):
    pass


def generate_stats(args):
    path = args.path

    sessions = []
    dataset_dirs = [f.path for f in os.scandir(path) if f.is_dir()]
    for d in dataset_dirs:
        if os.path.isfile(f"{d}/session.json"):
            session_string = open(f"{d}/session.json", "r").read()
            sessions.append(json.loads(session_string))

    telemetry_data = []
    for i, s in enumerate(sessions):
        session_folder = os.path.join(path, f"{s['date']}-{s['session']}")
        tfd = pd.read_csv(os.path.join(session_folder, "telemetry.txt"),
                          header=None,
                          names=telemetry_headers,
                          # dtype=[('capture', str), ('pos', float, 3)],
                          sep=';')
        tfd['environment'] = s['environment'].split("/")[0]
        tfd['traffic'] = s['traffic']
        tfd['weather'] = s.get('weather','Not specified')
        tfd['position'] = tfd[['p.x', 'p.y', 'p.z']].values.tolist()
        tfd['orientation'] = tfd[['o.x', 'o.y', 'o.z']].values.tolist()
        tfd['vel_l'] = tfd[['l_vel.x', 'l_vel.y', 'l_vel.z']].values.tolist()
        tfd['vel_a'] = tfd[['a_vel.x', 'a_vel.y', 'a_vel.z']].values.tolist()
        tfd['acc_l'] = tfd[['l_acc.x', 'l_acc.y', 'l_acc.z']].values.tolist()
        tfd['acc_a'] = tfd[['a_acc.x', 'a_acc.y', 'a_acc.z']].values.tolist()

        telemetry_data.append(tfd)

    telemetry_data = pd.concat(telemetry_data)
    movement = vector_diff(telemetry_data['position'],
                           telemetry_data['position'].shift(1, fill_value=0))
    static, non_static = movement.groupby(by=lambda x: movement[x] > 0.0).count()
    print(f"Frames totales: {telemetry_data.shape[0]}")
    print(f"Frames dinamicos: {(1-(static / non_static)) * 100:.2f}%")
    print(f"Entorno: ------------------")
    output = telemetry_data.groupby(by='environment').size().to_string(header=None).replace("\n", "\n\t")
    print(f"\t{output}")
    print(f"---------------------------")
    print(f"Niveles de tráfico: -------")
    output = telemetry_data.groupby(by='traffic').size().to_string(header=None).replace("\n", "\n\t")
    print(f"\t{output}")
    print(f"---------------------------")
    print(f"Tiempo: -------------------")
    output = telemetry_data.groupby(by='weather').size().to_string(header=None).replace("\n", "\n\t")
    print(f"\t{output}")
    print(f"---------------------------")



def vector_diff(a, b):
    b[0] = b[1]
    diff = pd.Series(name="Movement", data=np.linalg.norm(np.subtract(b.tolist(), a.tolist()), axis=1))
    return diff


def call_command(options):
    if options.split:
        generate_split(options.path)
    elif options.stats:
        generate_stats(options.path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    create_options(parser)
    args = parser.parse_args()

    tic = time.perf_counter()

    if args.command == 'split':
        pass
    elif args.command == 'stats':
        generate_stats(args)
    else:
        parser.print_help(sys.stdout)
        exit()

    toc = time.perf_counter()
    print(f"command {args.command} executed in {toc - tic:0.4f} seconds")

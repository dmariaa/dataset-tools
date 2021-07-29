import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import depth_utils


class DatasetCommands:
    def __init__(self, path):
        self.path = path
        self.telemetry_headers = [
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
            'a_acc.z',
            'scale'
        ]

    def get_sessions(self):

        sessions = []
        dataset_dirs = [f.path for f in os.scandir(self.path) if f.is_dir()]
        for d in dataset_dirs:
            if os.path.isfile(f"{d}/session.json"):
                session_string = open(f"{d}/session.json", "r").read()
                sessions.append(json.loads(session_string))
        return sessions

    def get_telemetry_data(self, sessions):
        telemetry_data = []
        for i, s in enumerate(sessions):
            session_folder = os.path.join(self.path, f"{s['date']}-{s['session']}")
            tfd = pd.read_csv(os.path.join(session_folder, "telemetry.txt"),
                              header=None,
                              index_col=False,
                              names=self.telemetry_headers,
                              # dtype=[('capture', str), ('pos', float, 3)],
                              sep=';')
            tfd['session'] = f"{s['date']}-{s['session']}"
            tfd['environment'] = s['environment'].split("/")[0]
            tfd['traffic'] = s['traffic']
            tfd['weather'] = s.get('weather','Not specified')
            tfd['position'] = tfd[['p.x', 'p.y', 'p.z']].values.tolist()
            tfd['orientation'] = tfd[['o.x', 'o.y', 'o.z']].values.tolist()
            tfd['vel_l'] = tfd[['l_vel.x', 'l_vel.y', 'l_vel.z']].values.tolist()
            tfd['vel_a'] = tfd[['a_vel.x', 'a_vel.y', 'a_vel.z']].values.tolist()
            tfd['acc_l'] = tfd[['l_acc.x', 'l_acc.y', 'l_acc.z']].values.tolist()
            tfd['acc_a'] = tfd[['a_acc.x', 'a_acc.y', 'a_acc.z']].values.tolist()

            # tfd['min_depth'], tfd['max_depth'] = np.vectorize(self.get_depth_stats)(tfd['session'], tfd['capture'])

            telemetry_data.append(tfd)

        telemetry_data = pd.concat(telemetry_data)
        telemetry_data['movement_delta'] = vector_diff(telemetry_data['position'],
                                                       telemetry_data['position'].shift(1, fill_value=0))
        return telemetry_data

    def generate_split(self, options):
        train_samples = options.training if options.training > 0 else 0.98

        if options.split == 'sfm':
            train_samples = 0.99
        elif options.split == 'depth':
            train_samples = 0.06

        sessions = self.get_sessions()
        telemetry_data = self.get_telemetry_data(sessions)
        non_static_frames = telemetry_data[telemetry_data['movement_delta'] > 0].sample(frac=1)
        non_static_frames = non_static_frames[['session', 'capture']]

        total_frames = non_static_frames.size
        val_samples_number = int(total_frames * (1 - train_samples))
        train_samples_number = total_frames - val_samples_number

        training_frames = non_static_frames.iloc[:-val_samples_number]
        val_frames = non_static_frames.iloc[-val_samples_number:]

        coincidences = pd.merge(training_frames, val_frames, how='inner', on=['session', 'capture'])

        if coincidences.shape[0] > 0:
            raise Exception("Bad split generated, some frames coincide in both sets")

        dest = os.path.join(os.getcwd(), options.dest)

        if not os.path.exists(dest):
            os.makedirs(dest)

        if options.format == 'list':
            np.savez_compressed(os.path.join(dest, "dataset_split"), train=training_frames.to_numpy(), val=val_frames.to_numpy())
        elif options.format == 'files':
            open(os.path.join(dest, "train_files.txt"), 'w').writelines(
                frame[0] + os.path.sep + frame[1] + "\n" for frame in training_frames.to_numpy())
            open(os.path.join(dest, "val_files.txt"), 'w').writelines(
                frame[0] + os.path.sep + frame[1] + "\n" for frame in val_frames.to_numpy())

    def generate_stats(self):
        sessions = self.get_sessions()
        telemetry_data = self.get_telemetry_data(sessions)

        total_frames = telemetry_data.shape[0]
        non_static = telemetry_data[telemetry_data['movement_delta'] > 0]['capture'].count()
        print(f"Frames totales: {total_frames}")
        print(f"Frames dinamicos: {non_static} ({(non_static / total_frames) * 100:.2f}%)")
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

    def get_depth_stats(self, session, frame):
        depth_header = depth_utils.read_depth_header(os.path.join(self.path, session, f"{frame}.depth.raw"))
        max_depth = -depth_header['min_val']
        min_depth = depth_header['max_val']
        return min_depth, max_depth

def vector_diff(a, b):
    b[0] = b[1]
    data = np.linalg.norm(np.subtract(b.tolist(), a.tolist()), axis=1)
    return data


def create_options(parser):
    subparsers = parser.add_subparsers(title='Dataset commands',
                                       dest='command',
                                       required=True)

    fix_subparser = subparsers.add_parser('split', help='Generate dataset split')
    fix_subparser.add_argument('path',
                               help='Path to the dataset',
                               type=str)
    fix_subparser.add_argument('--training',
                               help='Training fraction of samples',
                               type=float,
                               default=-1)
    fix_subparser.add_argument('--split',
                               help='Split to generate, flow for SFM based training, depth for ground-truth depth '
                                    'based training, custom (default) to use --training parameter',
                               choices=['sfm', 'depth', 'custom'])
    fix_subparser.add_argument('--format',
                               help="Split format to generate, 'list' (default) to generate .npz file, 'files' to "
                                    "generate train_files.txt, val_files.txt files",
                               choices=['list', 'files'],
                               default='list')
    fix_subparser.add_argument('--dest',
                               help="Destination folder, default ./split",
                               default='split')

    stats_subparser = subparsers.add_parser('stats', help='Generate stats')
    stats_subparser.add_argument('path',
                                 help='Path to the dataset',
                                 type=str)


def call_command(options):
    dc = DatasetCommands(options.path)

    if options.command == 'split':
        dc.generate_split(options)
    elif options.command == 'stats':
        dc.generate_stats()
    else:
        parser.print_help(sys.stdout)
        exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    create_options(parser)
    args = parser.parse_args()

    tic = time.perf_counter()
    call_command(args)

    toc = time.perf_counter()
    print(f"command {args.command} executed in {toc - tic:0.4f} seconds")

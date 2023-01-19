import argparse
import datetime
import glob
import json
import os
import sys
import time

import PIL.Image
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

    def check_depth(self):
        sessions = self.get_sessions()

        for i, s in enumerate(sessions):
            print(f"processing session {s}")
            session_folder = os.path.join(self.path, f"{s['date']}-{s['session']}")
            for fname in glob.glob(os.path.join(session_folder, "*.depth.raw")):
                f = depth_utils.read_depth_file(fname)
                data = f['data']

                if np.isnan(np.max(data)) or np.isnan(np.min(data)):
                    print(f"file {fname} data is NaN")

    def get_sessions(self):
        sessions = []
        dataset_dirs = sorted([f.path for f in os.scandir(self.path) if f.is_dir()])
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
            d = datetime.datetime.strptime(f"{s['date']} {s['gametime']}", "%Y%m%d %H:%M:%S")
            tfd['time'] = d
            tfd['moment'] = "mañana" if d.strftime("%H") in ["06", "07", "08", "09", "10", "11", "12", "13", "14", "15"] \
                else "tarde" if d.strftime("%H") in ["16", "17", "18", "19", "20"] else "noche"
            tfd['weather'] = s.get('weather', 'Not specified')
            tfd['position'] = tfd[['p.x', 'p.y', 'p.z']].values.tolist()
            tfd['orientation'] = tfd[['o.x', 'o.y', 'o.z']].values.tolist()
            tfd['vel_l'] = tfd[['l_vel.x', 'l_vel.y', 'l_vel.z']].values.tolist()
            tfd['vel_a'] = tfd[['a_vel.x', 'a_vel.y', 'a_vel.z']].values.tolist()
            tfd['acc_l'] = tfd[['l_acc.x', 'l_acc.y', 'l_acc.z']].values.tolist()
            tfd['acc_a'] = tfd[['a_acc.x', 'a_acc.y', 'a_acc.z']].values.tolist()

            v = tfd.tail(1).index[0]
            tfd['corner'] = np.logical_or(tfd.index == v, tfd.index == 0)

            # tfd['min_depth'], tfd['max_depth'] = np.vectorize(self.get_depth_stats)(tfd['session'], tfd['capture'])

            telemetry_data.append(tfd)

        telemetry_data = pd.concat(telemetry_data)
        telemetry_data['movement_delta'] = vector_diff(telemetry_data['position'],
                                                       telemetry_data['position'].shift(1, fill_value=0))
        return telemetry_data

    def get_training_data(self, data, split, static_frames=True, night_frames=False):
        if split == 'depth':
            # this frame has an invalid depth pixel (NaN)
            filtered = data[~((data['session'] == '20210725-000013') & (data['capture'] == 'capture-0000001966'))]
        elif split == 'sfm':
            # remove corners
            filtered = data[~data['corner']]

        # remove static frames
        if not static_frames:
            filtered = filtered[filtered['movement_delta'] > 0]

        # remove night frames
        if not night_frames:
            filtered = filtered[filtered['moment'] != 'noche']

        # filtered = filtered[filtered['session'].str.startswith('202107')]
        return filtered.sample(frac=1)[['session', 'capture']]

    def get_validation_data(self, data):
        filtered = data[~(data['session'].str.startswith('202107'))]
        return filtered

    def generate_test_set(self, options):
        validation_samples = 1000
        sessions = self.get_sessions()
        telemetry_data = self.get_telemetry_data(sessions)
        telemetry_data = self.get_validation_data(telemetry_data)
        non_static_frames = telemetry_data[telemetry_data['movement_delta'] > 0].sample(frac=1)
        frames = non_static_frames[:validation_samples]
        frames = frames[['session', 'capture']]
        print(f"Generated test set: {frames.shape[0]} samples")

        dest = os.path.join(os.getcwd(), options.dest)

        if not os.path.exists(dest):
            os.makedirs(dest)

        if options.format == 'list':
            np.savez_compressed(os.path.join(dest, "test_set"), test=frames.to_numpy())
        elif options.format == 'files':
            open(os.path.join(dest, "test_files.txt"), 'w').writelines(
                frame[0] + os.path.sep + frame[1] + "\n" for frame in frames.to_numpy())

    def generate_split(self, options):
        sessions = self.get_sessions()
        data = self.get_telemetry_data(sessions)
        data = self.get_training_data(data, options.split)

        if options.split == 'sfm':
            train_samples = 0.99
        elif options.split == 'depth':
            train_samples = 0.96
        else:
            train_samples = options.training

        total_frames = data.shape[0]
        val_samples_number = int(total_frames * (1 - train_samples))
        train_samples_number = total_frames - val_samples_number

        training_frames = data.iloc[:-val_samples_number]
        val_frames = data.iloc[-val_samples_number:]

        coincidences = pd.merge(training_frames, val_frames, how='inner', on=['session', 'capture'])

        if coincidences.shape[0] > 0:
            raise Exception("Bad split generated, some frames coincide in both sets")

        print(f"Generated split for {options.split}: train {training_frames.shape[0]}, val {val_frames.shape[0]}")

        dest = os.path.join(os.getcwd(), options.dest)

        if not os.path.exists(dest):
            os.makedirs(dest)

        if options.format == 'list':
            np.savez_compressed(os.path.join(dest, "dataset_split"), train=training_frames.to_numpy(),
                                val=val_frames.to_numpy())
        elif options.format == 'files':
            open(os.path.join(dest, "train_files.txt"), 'w').writelines(
                f"{frame[0]} {int(frame[1][-10:])} l\n" for frame in training_frames.to_numpy())
            open(os.path.join(dest, "val_files.txt"), 'w').writelines(
                f"{frame[0]} {int(frame[1][-10:])} l\n" for frame in val_frames.to_numpy())

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
        print(f"Hora del día: -------------")
        output = telemetry_data.groupby(by='moment').size().to_string(header=None) \
            .replace("\n", "\n\t")
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

    def get_generate_latex(self):
        sessions = self.get_sessions()
        telemetry_data = self.get_telemetry_data(sessions)
        sorted_data = telemetry_data.groupby(by='session', as_index=False)
        data = sorted_data.agg({'capture': 'count', 'moment': 'first', 'time': lambda x: x.dt.strftime("%H:%M:%S")[0],
                                'environment': 'first', 'traffic': 'first', 'weather': 'first'}).sort_values('time')
        data['size'] = (data['capture'] * 3 + data['capture'] * 2) / 1024
        print(data[['session', 'size', 'capture', 'moment', 'environment', 'traffic', 'weather']]
              .style.hide_index().format({'size': '{:.2f} GB'}).to_latex())

    def get_depth_stats(self, session, frame):
        depth_header = depth_utils.read_depth_header(os.path.join(self.path, session, f"{frame}.depth.raw"))
        max_depth = -depth_header['min_val']
        min_depth = depth_header['max_val']
        return min_depth, max_depth

    def check_color(self):
        sessions = self.get_sessions()

        for i, s in enumerate(sessions):
            print(f"processing session {s}")
            session_folder = os.path.join(self.path, f"{s['date']}-{s['session']}")

            for fname in glob.glob(os.path.join(session_folder, "*.jpg")):
                f = PIL.Image.open(fname)
                data = np.array(f)

                if np.isnan(np.max(data)) or np.isnan(np.min(data)):
                    print(f"file {fname} data is NaN")


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
                               help='Training fraction of samples, use with --split custom',
                               type=float,
                               default=0.98)
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

    check_depth_subparser = subparsers.add_parser('check-depth', help='Checks depth files for invalid data')
    check_depth_subparser.add_argument('path',
                                       help='Path to the dataset',
                                       type=str)

    check_depth_subparser = subparsers.add_parser('check-color', help='Checks color files for invalid data')
    check_depth_subparser.add_argument('path',
                                       help='Path to the dataset',
                                       type=str)

    generate_test_subparser = subparsers.add_parser('generate-test', help='Generate test set')
    generate_test_subparser.add_argument('path',
                                         help='Path to the dataset',
                                         type=str)
    generate_test_subparser.add_argument('--format',
                                         help="Split format to generate, 'list' (default) to generate .npz file, "
                                              "'files' to generate test_files.txt",
                                         choices=['list', 'files'],
                                         default='list')
    generate_test_subparser.add_argument('--dest',
                                         help="Destination folder, default ./split",
                                         default='split')


def call_command(options):
    dc = DatasetCommands(options.path)

    if options.command == 'split':
        dc.generate_split(options)
    elif options.command == 'stats':
        dc.get_generate_latex()
        dc.generate_stats()
    elif options.command == 'check-depth':
        dc.check_depth()
    elif options.command == 'check-color':
        dc.check_color()
    elif options.command == 'generate-test':
        dc.generate_test_set(options)
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

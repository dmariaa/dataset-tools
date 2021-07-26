import glob
import json
import os


def generate_split(path):
    pass


def generate_stats(path):
    dataset_dirs = [f.path for f in os.scandir(path) if f.is_dir()]
    for d in dataset_dirs:
        if os.path.isfile(f"{d}/session.json"):
            session_string = open(f"{d}/session.json", "r").read()
            session_data = json.loads(session_string)
            print(f"{session_data['date']}-{session_data['session']}")


def call_command(options):
    if options.split:
        generate_split(options.path)
    elif options.stats:
        generate_stats(options.path)
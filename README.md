# Some useful scripts for the ETS2 Dataset

> :warning: **Highly experimental code**: This code is not meant to be used in production systems, I just build it to 
> get information and other data needed for my degree final project.

This project contains some python code that manipulates and gets information for the ETS Dataset I built for my degree
in videogames design and development final project. Use it at your discretion.

## dataset_tools.py

Some tools to get data from the ETS2 Dataset

```
> python dataset_commands.py --help
usage: dataset_commands.py [-h] {split,stats,check-depth,generate-test} ...

optional arguments:
  -h, --help            show this help message and exit

Dataset commands:
  {split,stats,check-depth,generate-test}
    split               Generate dataset split
    stats               Generate stats
    check-depth         Checks depth files for invalid data
    generate-test       Generate test set

```

## depth_utils.py

Some tools to manipulate .depth.raw files from the dataset.

```
 python depth_utils.py --help
usage: depth_utils.py [command] ...

Utilidades para testear ETS2 dataset

positional arguments:
  {compare,dump,dataset}

optional arguments:
  -h, --help            show this help message and exit
```
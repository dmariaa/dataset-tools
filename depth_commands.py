def create_options(parser):
    subparsers = parser.add_subparsers(title='Depth commands',
                                       dest='command',
                                       required=True)

    stats_subparser = subparsers.add_parser('stats', help='Show some depth stats')
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

    video_subparser = subparsers.add_parser('stats', help='Generate stats')
    video_subparser.add_argument('path',
                                 help='Path to the dataset',
                                 type=str)

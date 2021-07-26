import sys
import time

import options
import dataset_commands
import depth_utils


parser = options.init_options()
args = parser.parse_args()

tic = time.perf_counter()

if args.command == 'dump':
    depth_utils.convert_to_colormap(args.file, args.format, 'magma')
elif args.command == 'compare':
    result = depth_utils.compare_depths(args.file1, args.file2)
    print(result)
elif args.command == 'dataset':
    dataset_commands.call_command(args)
else:
    parser.print_help(sys.stdout)
    exit()

toc = time.perf_counter()
print(f"command {args.command} executed in {toc - tic:0.4f} seconds")
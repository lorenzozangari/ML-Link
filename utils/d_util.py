import os
from pathlib import Path


def print_arguments(args, nl=7):
    print()
    print('Input Arguments: ')
    if isinstance(args, dict):
        for i, (k, v) in enumerate(args.items()):
            print("{} : {}".format(k, v), end="; ")
            if (i + 1) % nl == 0:
                print()
    else:
        for i, arg in enumerate(vars(args)):

            print("{} : {}".format(arg, getattr(args, arg)), end="; ")
            if (i + 1) % nl == 0:
                print()
    print()


def write_results(args, d, name='results', time_list=None):
    pfx = ''
    if hasattr(args, 'no_struct') or hasattr(args, 'no_gnn'):
        no_struct = args.no_struct
        no_gnn = args.no_gnn
        if not no_struct:
            pfx = pfx + 'struct_'
        if not no_gnn:
            pfx = pfx + 'gnn'

    dr = f'{args.save_dir}/{name}/'
    Path(dr).mkdir(parents=True, exist_ok=True)
    file_name = f'{dr}/{args.dataset}_{args.seed}_{pfx}.txt'
    if os.path.exists(file_name):
        append_write = 'a'
    else:
        append_write = 'w'
    file = open(file_name, append_write)
    file.write("*******************************************************\n")
    file.write('Dataset : ' + args.dataset + "\n")
    file.write('ARGS : ' + str(vars(args)) + "\n")
    file.write('\n')
    if time_list is not None:
        t = time_list.mean()
        file.write(f'Average optimization time : {t}\n')
        file.write('\n')
    for key in d:
        value = d[key]
        file.write(f"{key} Test: " + "{:.3f}\n".format(value))
    file.write('\n')
    file.close()

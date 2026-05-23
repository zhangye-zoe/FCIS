import argparse
import json
import os
import os.path as osp
from prettytable import PrettyTable
from collections import OrderedDict


def log_analysis(log_path):
    if osp.isdir(log_path):
        paths = [osp.join(log_path, x) for x in os.listdir(log_path) if osp.splitext(x)[1] == '.json']
        indices = list(range(len(paths)))
        indices = sorted(indices, key=lambda x: osp.getctime(paths[x]), reverse=True)
        paths = [paths[index] for index in indices]
        # The newest path
        log_path = paths[0]

    with open(log_path, 'r') as fp:
        lines = fp.readlines()
        logs = []
        for line in lines:
            line = line.strip()
            logs.append(json.loads(line))

    _ = logs[0]
    logs = logs[1:]

    logs_ = []
    for log in logs:
        if log['mode'] != 'val':
            continue
        logs_.append(log)
    logs_ = logs_[-5:]

    compare_val = -1
    collect_res = {'mean': {}, 'max': {}}
    compare_key = 'mAji'

    # inst_keys = ['imwAji', 'bAji', 'mAji', 'bDQ', 'bSQ', 'bPQ', 'imwDQ', 'imwSQ', 'imwPQ', 'mDQ', 'mSQ', 'mPQ']
    mean_keys = ['imwDice', 'imwAji', 'imwDQ', 'imwSQ', 'imwPQ']
    overall_keys = ['mDice', 'mAji', 'mDQ', 'mSQ', 'mPQ']
    other_keys = ['dir_mDice', 'dir_mPrecision', 'dir_mRecall']
    for log in logs_:
        epoch = log['epoch']
        collect_res[epoch] = {}
        for key in overall_keys + mean_keys:
            if key not in log:
                continue
            collect_res[epoch][key] = log[key]
            if key not in collect_res['mean']:
                collect_res['mean'][key] = [log[key]]
            else:
                collect_res['mean'][key].append(log[key])
        # for mean_k in mean_keys:
        #     collect_res[epoch][mean_k] = log[mean_k]
        #     if mean_k not in collect_res['mean']:
        #         collect_res['mean'][mean_k] = [log[mean_k]]
        #     else:
        #         collect_res['mean'][inst_key].append(log[inst_key])
        # for sem_key in sem_keys:
        #     collect_res[epoch][sem_key] = log[sem_key]
        #     if sem_key not in collect_res['mean']:
        #         collect_res['mean'][sem_key] = [log[sem_key]]
        #     else:
        #         collect_res['mean'][sem_key].append(log[sem_key])
        # for dir_key in dir_keys:
        #     if dir_key not in log:
        #         continue
        #     collect_res[epoch][dir_key] = log[dir_key]
        #     if dir_key not in collect_res['mean']:
        #         collect_res['mean'][dir_key] = [log[dir_key]]
        #     else:
        #         collect_res['mean'][dir_key].append(log[dir_key])

        if log[compare_key] > compare_val:
            collect_res['max'] = collect_res[epoch]
            compare_val = log[compare_key]

    for key in collect_res['mean'].keys():
        collect_res['mean'][key] = round(sum(collect_res['mean'][key]) / len(collect_res['mean'][key]), 2)

    res_table = PrettyTable()
    res = OrderedDict()
    res.update({'names': list(collect_res.keys())})
    res.move_to_end('names', last=False)
    combine_keys = overall_keys + mean_keys + other_keys
    empty_keys = []
    for key in combine_keys:
        res[key] = []
        for k, single_res in collect_res.items():
            if key not in single_res:
                continue
            res[key].append(single_res[key])
        if len(res[key]) == 0:
            empty_keys.append(key)
    # remove empty key
    [res.pop(key) for key in empty_keys]

    for key, val in res.items():
        res_table.add_column(key, val)

    return res_table.get_string()


def parse_args():
    parser = argparse.ArgumentParser(description='Extract related results.')
    parser.add_argument('log_path', help='The json style log file path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    log_path = args.log_path

    print(log_analysis(log_path))


if __name__ == '__main__':
    main()

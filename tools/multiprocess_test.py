import argparse
import os
import os.path as osp
from multiprocessing import Process, Queue

from benchmark_analysis import benchmark_analysis

devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
devices_domain = devices.split(',')
DQ = Queue()
[DQ.put(d) for d in devices_domain]


def run_test(d_num, config_path, ckpt_path):
    print(f'execute: CUDA_VISIBLE_DEVICES={d_num} python tools/test.py {config_path} {ckpt_path}')
    os.system(f'CUDA_VISIBLE_DEVICES={d_num} python tools/test.py {config_path} {ckpt_path}')
    DQ.put(d_num)


def check_processes(process_list):
    if len(process_list) == 0:
        return 0

    alive_len = 0
    for process in process_list:
        if process.is_alive():
            alive_len += 1

    return alive_len


def parse_args():
    parser = argparse.ArgumentParser(description='benchmark models')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('benchmark_folder', help='The analysis log save folder.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    folder = args.benchmark_folder
    config_path = args.config
    ckpts = [osp.join(folder, x) for x in os.listdir(folder) if x.startswith('epoch_')]
    indices = list(range(len(ckpts)))
    indices = sorted(indices, key=lambda x: osp.getctime(ckpts[x]), reverse=True)
    ckpts = [ckpts[index] for index in indices]
    # The 5 newest pths
    ckpts = ckpts[:5]

    process_list = []
    ckpt_len = len(ckpts)
    cur = 0
    while True:
        if DQ.qsize() > 0:
            if cur == ckpt_len:
                if check_processes(process_list) == 0:
                    break
                else:
                    continue
            d_num = DQ.get()
            p = Process(target=run_test, args=(d_num, config_path, ckpts[cur]))
            p.start()
            process_list.append(p)
            cur += 1
        else:
            for i in process_list:
                p.join()
            if cur == ckpt_len:
                break

    # calculate metrics
    eval_dir = f'{folder}/test'
    res_string = benchmark_analysis(eval_dir)
    with open(osp.join(eval_dir, 'res.log'), 'w') as fp:
        fp.write(res_string)
    print(res_string)


if __name__ == '__main__':
    main()

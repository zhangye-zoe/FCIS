import argparse
import copy
import os
import os.path as osp
import time
import warnings
import numpy as np
import mmcv
import torch
from mmcv.runner import init_dist
from mmcv.utils import Config, DictAction, get_git_hash, collect_env, get_logger

from tiseg import __version__
from tiseg.apis import train_segmentor
from tiseg.datasets import build_dataset
from tiseg.models import build_segmentor
from tiseg.models.utils import revert_sync_batchnorm


def parse_args():
    parser = argparse.ArgumentParser(description='Analyse Log')
    parser.add_argument('log_path', help='log file path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    path = args.log_path
    save_dir = osp.dirname(path)
    # print(path)
    # print(save_dir)
    results = []
    results_mDice = []
    results_dir_mDice = []
    last_epoch_num = 5
    with open(path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            # print(lines)
            if not lines:
                break
            if ("dir_mdice" in lines):
                results_dir_mDice.append(lines[:-1])
            if ("mDice |" in lines):
                for i in range(2):
                    lines = file_to_read.readline()
                results_mDice.append(lines[:-1])
            if ("imwAji |" in lines):
                for i in range(2):
                    lines = file_to_read.readline()
                results.append(lines[:-1])
                # print(lines[:-1])
    save_name = "last" + str(last_epoch_num) + "epoch_result.txt"
    with open(osp.join(save_dir, save_name), "w") as f:
        all_ans = []
        for i in range(last_epoch_num):
            res = results[-last_epoch_num:][i]
            f.writelines(res + '\n')
            print(res)
            ans = []
            l = 0
            while True:
                r = res[l + 1:].find('|')
                if (r == -1):
                    break
                r = r + l + 1
                ans.append(float(res[l + 1:r]))
                l = r
            res = results_mDice[-last_epoch_num:][i]
            f.writelines(res + '\n')
            print(res)
            l = 0
            while True:
                r = res[l + 1:].find('|')
                if (r == -1):
                    break
                r = r + l + 1
                ans.append(float(res[l + 1:r]))
                break
                l = r
            if len(results_dir_mDice) > 0:
                res = results_dir_mDice[-last_epoch_num:][i]
                f.writelines(res + '\n')
                print(res)
                l = res.find("dir_mdice")
                r = res[l + 1:].find(',') + l + 1
                ans.append(float(res[l + 10:r]))

            all_ans.append(ans)  #mDice

        ans = np.array(all_ans)
        average_res = np.mean(ans, axis=0)

        c = 0
        id = 0
        for i in range(last_epoch_num):
            if ans[i][2] > c:  #mAji
                c = ans[i][2]
                id = i
        best_res = ans[id]

        average_res = np.round(average_res, 2).tolist()
        best_res = np.round(best_res, 2).tolist()
        f.write('average_res:')
        f.writelines(str(average_res) + '\n')
        f.write('best_res:')
        f.writelines(str(best_res) + '\n')
        print('average_res:', average_res)
        print('best_res:', best_res)


if __name__ == '__main__':
    main()

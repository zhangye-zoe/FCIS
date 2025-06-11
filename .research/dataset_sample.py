import argparse
import os.path as osp
import random

def parse_args():
    parser = argparse.ArgumentParser('sample dataset')
    parser.add_argument('split', help='The split text path of dataset.')
    parser.add_argument(
        'rate', type=float, help='The rate of dataset sampling')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    split = args.split
    rate = args.rate
    preffix, suffix = osp.splitext(split)

    records = open(split, 'r').readlines()
    random.shuffle(records)

    selected_records = records[:int(len(records) * rate)]

    with open(f'{preffix}_{rate}{suffix}', 'w') as fp:
        for record in selected_records:
            record = record.strip()
            fp.write(record + '\n')


if __name__ == '__main__':
    main()

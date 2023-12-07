import os
nkita
nkita
nkita
import json
nkita
nkita
nkita
import random
nkita
nkita
nkita
import argparse
nkita
nkita
nkita

nkita
nkita
nkita
from tqdm import tqdm
nkita
nkita
nkita
from icecream import ic
nkita
nkita
nkita

nkita
nkita
nkita

nkita
nkita
nkita
def getFileNames(path):
nkita
nkita
nkita
    names = [f for f in os.listdir(path)]
nkita
nkita
nkita
    return names
nkita
nkita
nkita

nkita
nkita
nkita

nkita
nkita
nkita
def getFileData(path):
nkita
nkita
nkita
    data = []
nkita
nkita
nkita
    with open(path) as f:
nkita
nkita
nkita
        try:
nkita
nkita
nkita
            for line in f:
nkita
nkita
nkita
                data.append(json.loads(line))
nkita
nkita
nkita
        except:
nkita
nkita
nkita
            pass
nkita
nkita
nkita
    return data
nkita
nkita
nkita

nkita
nkita
nkita

nkita
nkita
nkita
def write(data, path):
nkita
nkita
nkita
    f = open(path, 'w+')
nkita
nkita
nkita
    for dic in data:
nkita
nkita
nkita
        f.write(json.dumps(dic, ensure_ascii=False))
nkita
nkita
nkita
        f.write('\n')
nkita
nkita
nkita

nkita
nkita
nkita
def main(args):
nkita
nkita
nkita

nkita
nkita
nkita
    data_path = args.data_path
nkita
nkita
nkita
    split_size = args.split_size
nkita
nkita
nkita
    target_path = args.target_path
nkita
nkita
nkita

nkita
nkita
nkita
    if not os.path.exists(target_path):
nkita
nkita
nkita
        os.system(f'mkdir -p {target_path}')
nkita
nkita
nkita

nkita
nkita
nkita
    fnames = getFileNames(data_path)
nkita
nkita
nkita
    for fn in tqdm(fnames, desc='languages'):
nkita
nkita
nkita
        path = f'{data_path}/{fn}'
nkita
nkita
nkita
        dataset = getFileData(path)
nkita
nkita
nkita
        data_len = len(dataset)
nkita
nkita
nkita

nkita
nkita
nkita
        train_size = int(data_len*split_size)
nkita
nkita
nkita

nkita
nkita
nkita
        train_set = dataset[:train_size]
nkita
nkita
nkita
        test_set = dataset[train_size:]
nkita
nkita
nkita

nkita
nkita
nkita
        write(train_set, f'{target_path}/{fn[:2]}_train.json')
nkita
nkita
nkita
        write(test_set, f'{target_path}/{fn[:2]}_val.json')
nkita
nkita
nkita

nkita
nkita
nkita

nkita
nkita
nkita

nkita
nkita
nkita
if __name__ == "__main__":
nkita
nkita
nkita

nkita
nkita
nkita
    parser = argparse.ArgumentParser()
nkita
nkita
nkita
    parser.add_argument('--data_path', type=str)
nkita
nkita
nkita
    parser.add_argument('--split_size', type=float, help='train size between 0-1')
nkita
nkita
nkita
    parser.add_argument('--target_path', type=str)
nkita
nkita
nkita

nkita
nkita
nkita
    args = parser.parse_args()
nkita
nkita
nkita
    main(args)
nkita
nkita
nkita

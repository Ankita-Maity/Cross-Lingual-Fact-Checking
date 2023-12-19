import os
import json
import random
import argparse
from tqdm import tqdm
from icecream import ic

def getFileNames(path):
    names = [f for f in os.listdir(path)]
    return names

def getFileData(path):
    data = []
    with open(path) as f:
        try:
            for line in f:
                data.append(json.loads(line))
        except:
            pass
    return data

def write(data, path):
    f = open(path, 'w+')
    for dic in data:
        f.write(json.dumps(dic, ensure_ascii=False))
        f.write('\n')
        
def main(args):
    data_path = args.data_path
    split_size = args.split_size
    target_path = args.target_path
    if not os.path.exists(target_path):
        os.system(f'mkdir -p {target_path}')
    fnames = getFileNames(data_path)
    for fn in tqdm(fnames, desc='languages'):
        path = f'{data_path}/{fn}'
        dataset = getFileData(path)
        data_len = len(dataset)
        train_size = int(data_len*split_size)
        train_set = dataset[:train_size]
        test_set = dataset[train_size:]
        write(train_set, f'{target_path}/{fn[:2]}_train.json')
        write(test_set, f'{target_path}/{fn[:2]}_val.json')
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--split_size', type=float, help='train size between 0-1')
    parser.add_argument('--target_path', type=str)
    args = parser.parse_args()
    main(args)

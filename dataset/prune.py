import os
nkita
nkita
nkita
import json
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
from collections import defaultdict
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
    fnames = getFileNames(data_path)
nkita
nkita
nkita

nkita
nkita
nkita
    for fn in tqdm(fnames):
nkita
nkita
nkita
        path = f'{data_path}/{fn}'
nkita
nkita
nkita
        data = getFileData(path)
nkita
nkita
nkita
        dataset = []
nkita
nkita
nkita

nkita
nkita
nkita
        for d in data:
nkita
nkita
nkita
            facts = d['xalign_facts']
nkita
nkita
nkita

nkita
nkita
nkita
            new_facts = []
nkita
nkita
nkita
            for f in facts:
nkita
nkita
nkita
                new_facts.append([f[0], f[1]])
nkita
nkita
nkita

nkita
nkita
nkita
            dataset.append({
nkita
nkita
nkita
                'title': d['title'],
nkita
nkita
nkita
                'qid': d['qid'],
nkita
nkita
nkita
                'xalign_sent': d['xalign_sent'],
nkita
nkita
nkita
                'xalign_facts': new_facts,
nkita
nkita
nkita
                'xwgen_refs': d['xwgen_refs'],
nkita
nkita
nkita
                'content': d['content']
nkita
nkita
nkita
            })
nkita
nkita
nkita

nkita
nkita
nkita
        fout = open(path, 'w+')
nkita
nkita
nkita
        for dic in dataset:
nkita
nkita
nkita
            fout.write(json.dumps(dic, ensure_ascii=False))
nkita
nkita
nkita
            fout.write('\n')
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
    parser.add_argument('--data_path')
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

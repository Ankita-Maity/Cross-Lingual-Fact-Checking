import os
nkita
nkita
nkita
import argparse
nkita
nkita
nkita
import json
nkita
nkita
nkita
import pandas as pd
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
def write(path, data):
nkita
nkita
nkita
    df = pd.DataFrame.from_dict(data)
nkita
nkita
nkita
    df.to_csv(path)
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
    output_path = args.output_path
nkita
nkita
nkita

nkita
nkita
nkita
    if not os.path.exists(output_path):
nkita
nkita
nkita
        os.mkdir(output_path)
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

nkita
nkita
nkita
    sent_stats = defaultdict(lambda: defaultdict(int))
nkita
nkita
nkita
    fact_stats = defaultdict(lambda: defaultdict(int))
nkita
nkita
nkita
    article_stats = defaultdict(lambda: defaultdict(int))
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

nkita
nkita
nkita
        num_articles = set()
nkita
nkita
nkita
        num_sent = 0
nkita
nkita
nkita
        num_facts = 0
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
            num_articles.add(d['qid'])
nkita
nkita
nkita

nkita
nkita
nkita
            facts = d['xalign_facts']
nkita
nkita
nkita
            sent = d['xalign_sent']
nkita
nkita
nkita

nkita
nkita
nkita
            num_sent += 1
nkita
nkita
nkita
            for f in facts:
nkita
nkita
nkita
                num_facts += 1
nkita
nkita
nkita

nkita
nkita
nkita
        lang = fn[:2]
nkita
nkita
nkita
        dom = fn[3:-5]
nkita
nkita
nkita
        sent_stats[lang][dom] = num_sent
nkita
nkita
nkita
        fact_stats[lang][dom] = num_facts
nkita
nkita
nkita
        article_stats[lang][dom] = len(num_articles)
nkita
nkita
nkita

nkita
nkita
nkita
    write(f'{output_path}/art_stats.csv', article_stats)
nkita
nkita
nkita
    write(f'{output_path}/sent_stats.csv', sent_stats)
nkita
nkita
nkita
    write(f'{output_path}/fact_stats.csv', fact_stats)
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
    parser.add_argument('--output_path')
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

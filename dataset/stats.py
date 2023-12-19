import os
import argparse
import json
import pandas as pd
from tqdm import tqdm
from icecream import ic
from collections import defaultdict

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

def write(path, data):
    df = pd.DataFrame.from_dict(data)
    df.to_csv(path)
    
def main(args):
    data_path = args.data_path
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    fnames = getFileNames(data_path)
    sent_stats = defaultdict(lambda: defaultdict(int))
    fact_stats = defaultdict(lambda: defaultdict(int))
    article_stats = defaultdict(lambda: defaultdict(int))
    for fn in tqdm(fnames):
        path = f'{data_path}/{fn}'
        data = getFileData(path)
        num_articles = set()
        num_sent = 0
        num_facts = 0
        for d in data:
            num_articles.add(d['qid'])
            facts = d['xalign_facts']
            sent = d['xalign_sent']
            num_sent += 1
            for f in facts:
                num_facts += 1
        lang = fn[:2]
        dom = fn[3:-5]
        sent_stats[lang][dom] = num_sent
        fact_stats[lang][dom] = num_facts
        article_stats[lang][dom] = len(num_articles)
    write(f'{output_path}/art_stats.csv', article_stats)
    write(f'{output_path}/sent_stats.csv', sent_stats)
    write(f'{output_path}/fact_stats.csv', fact_stats)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path')
    parser.add_argument('--output_path')
    args = parser.parse_args()
    main(args)

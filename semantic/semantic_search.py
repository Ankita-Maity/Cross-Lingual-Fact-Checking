import os
import json
import string
import argparse
from tqdm import tqdm
from icecream import ic
from wikimapper import WikiMapper
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util

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
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    fnames = getFileNames(data_path)
    model = SentenceTransformer('sentence-transformers/LaBSE')
    for fn in tqdm(fnames, desc='file names'):
        data = getFileData(f'{data_path}/{fn}')
        all_stored = []
        for d in tqdm(data, desc='data'):
            facts = d['article']
            title = d['title']
            doc = d['references']
            if len(doc) == 0:
                continue
            doc_emb = model.encode(doc)
            for f in facts:
                triplets = f['fact']
                for t in triplets:
                    query = f'{title} {t[0]} {t[1]}'
                    query_emb = model.encode(query)
                    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
                    doc_score_pairs = list(zip(doc, scores))
                    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
                    all_stored.append({
                        "fact": t,
                        "sentences": doc_score_pairs,
                        "title": title,
                    })
        write(all_stored, f'{output_path}/{fn}')
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path')
    parser.add_argument('--output_path')
    args = parser.parse_args()
    main(args)

import os
nkita
import json
nkita
import string
nkita
import argparse
nkita

nkita
from tqdm import tqdm
nkita
from icecream import ic
nkita
from wikimapper import WikiMapper
nkita
from collections import defaultdict
nkita

nkita
from sentence_transformers import SentenceTransformer, util
nkita

nkita

nkita
def getFileNames(path):
nkita
    names = [f for f in os.listdir(path)]
nkita
    return names
nkita

nkita

nkita
def getFileData(path):
nkita
    data = []
nkita
    with open(path) as f:
nkita
        try:
nkita
            for line in f:
nkita
                data.append(json.loads(line))
nkita
        except:
nkita
            pass
nkita
    return data
nkita

nkita

nkita
def write(data, path):
nkita
    f = open(path, 'w+')
nkita
    for dic in data:
nkita
        f.write(json.dumps(dic, ensure_ascii=False))
nkita
        f.write('\n')
nkita

nkita
def main(args):
nkita

nkita
    data_path = args.data_path
nkita
    output_path = args.output_path
nkita

nkita
    if not os.path.exists(output_path):
nkita
        os.mkdir(output_path)
nkita

nkita
    fnames = getFileNames(data_path)
nkita
    model = SentenceTransformer('sentence-transformers/LaBSE')
nkita

nkita
    for fn in tqdm(fnames, desc='file names'):
nkita
        data = getFileData(f'{data_path}/{fn}')
nkita
        all_stored = []
nkita

nkita
        for d in tqdm(data, desc='data'):
nkita

nkita
            facts = d['article']
nkita
            title = d['title']
nkita
            doc = d['references']
nkita

nkita
            if len(doc) == 0:
nkita
                continue
nkita
            doc_emb = model.encode(doc)
nkita

nkita
            for f in facts:
nkita
                triplets = f['fact']
nkita
                for t in triplets:
nkita
                    query = f'{title} {t[0]} {t[1]}'
nkita
                    query_emb = model.encode(query)
nkita

nkita
                    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
nkita
                    doc_score_pairs = list(zip(doc, scores))
nkita
                    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
nkita

nkita
                    all_stored.append({
nkita
                        "fact": t,
nkita
                        "sentences": doc_score_pairs,
nkita
                        "title": title,
nkita
                    })
nkita
        write(all_stored, f'{output_path}/{fn}')
nkita

nkita

nkita
if __name__ == "__main__":
nkita

nkita
    parser = argparse.ArgumentParser()
nkita
    parser.add_argument('--data_path')
nkita
    parser.add_argument('--output_path')
nkita

nkita
    args = parser.parse_args()
nkita
    main(args)
nkita

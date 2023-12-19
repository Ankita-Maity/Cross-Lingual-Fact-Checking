import os
import json
import string
import argparse
from tqdm import tqdm
from icecream import ic
from wikimapper import WikiMapper
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

def main(args):
    xalign_path = args.xalign_path
    xwikigen_path = args.xwikigen_path
    index_path = args.index_path
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.system(f'mkdir -p {output_path}')
    LANG = {'hi', 'bn', 'ta', 'pa', 'or', 'en'}
    DOM = {'sportsman', 'writers', 'politicians'}
    PUNCTUATIONS = string.punctuation + ' '
    for ln in tqdm(LANG, desc='languages'):
        wikigen = f'{xwikigen_path}/{ln}'
        ic(ln)
        gennames = getFileNames(wikigen)
        # getting wikigen data
        gendataset = []
        genfset = []
        gentitles = set()
        for fn in tqdm(gennames, desc='gen dataset'):
            if fn[3:-5] in DOM:
                gendataset.append(getFileData(f'{wikigen}/{fn}'))
                genfset.append(fn[3:-5])
        for d, fn in tqdm(zip(gendataset, genfset), desc='wikiref titles'):
            for article in d:
                title = article['title']
                title = title.translate(str.maketrans('', '', PUNCTUATIONS)).strip()
                gentitles.add(title)
        # getting xalign data
        xalign = f'{xalign_path}/{ln}'
        alnames = getFileNames(xalign)
        aldataset = []
        altitles = set()
        mapper = f'{index_path}/index_{ln}wiki-latest.db'
        mapper = WikiMapper((mapper))
        for fn in tqdm(alnames, desc='xalign files'):
            if 'test' in fn:
                aldataset.append(getFileData(f'{xalign}/{fn}'))
        for d in tqdm(aldataset, desc='xalign titles'):
            for sent in tqdm(d, desc='sentences'):
                title = mapper.id_to_titles(sent['qid'])
                if len(title) != 0:
                    altitles.add(title[0].translate(str.maketrans('', '', PUNCTUATIONS)).strip())
        intersection = gentitles.intersection(altitles)
        ic(ln, len(intersection))
        tempref = defaultdict(list)
        tempcont = defaultdict(list)
        for d, fn in tqdm(zip(gendataset, genfset), desc='titles from wikiref'):
            for article in d:
                title = article['title']
                title = title.translate(str.maketrans('', '', PUNCTUATIONS)).strip()
                if title in intersection:
                    sections = article['sections']
                    for section in sections:
                        refs = section['references']
                        cont = section['content']
                        if len(cont) != 0:
                            tempcont[title].append(cont)
                        for ref in refs:
                            tempref[title].append(ref)
        tempal = defaultdict(list)
        for d in tqdm(aldataset, desc='xalign'):
            for sent in tqdm(d, desc='article'):
                qid = sent['qid']
                title = mapper.id_to_titles(qid)
                if len(title) != 0:
                    title = title[0].translate(str.maketrans('','',PUNCTUATIONS)).strip()
                    if title in intersection:
                        tempal[title].append({
                            'sentence': sent['sentence'],
                            'fact': sent['facts'],
                            'qid': qid
                        })
                        # tempsent[title].append(sent['sentence'])
                        # tempal[title].append(sent['facts'])
                        # tempqid[title].add(qid)
        output = []
        for key in list(intersection):
            for stuff in tempal[key]:
                output.append({
                    "title": key,
                    "qid": stuff['qid'],
                    "xalign_sent": stuff['sentence'],
                    "xalign_facts": stuff['fact'],
                    "xwgen_refs": tempref[key],
                    "content": tempcont[key]
                })
        fout = open(f'{output_path}/{ln}_test.json', 'w+')
        ic(ln, len(output))
        for dic in output:
            fout.write(json.dumps(dic, ensure_ascii=False))
            fout.write('\n')
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--xalign_path')
    parser.add_argument('--xwikigen_path')
    parser.add_argument('--index_path')
    parser.add_argument('--output_path')
    args = parser.parse_args()
    main(args)

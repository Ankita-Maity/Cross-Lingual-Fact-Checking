import os
nkita
nkita
nkita
import json
nkita
nkita
nkita
import string
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
from wikimapper import WikiMapper
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
    xalign_path = args.xalign_path
nkita
nkita
nkita
    xwikigen_path = args.xwikigen_path
nkita
nkita
nkita
    index_path = args.index_path
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
        os.system(f'mkdir -p {output_path}')
nkita
nkita
nkita

nkita
nkita
nkita
    LANG = {'hi', 'bn', 'ta', 'pa', 'or', 'en'}
nkita
nkita
nkita
    DOM = {'sportsman', 'writers', 'politicians'}
nkita
nkita
nkita
    PUNCTUATIONS = string.punctuation + ' '
nkita
nkita
nkita

nkita
nkita
nkita
    for ln in tqdm(LANG, desc='languages'):
nkita
nkita
nkita
        wikigen = f'{xwikigen_path}/{ln}'
nkita
nkita
nkita
        ic(ln)
nkita
nkita
nkita
        gennames = getFileNames(wikigen)
nkita
nkita
nkita

nkita
nkita
nkita
        # getting wikigen data
nkita
nkita
nkita
        gendataset = []
nkita
nkita
nkita
        genfset = []
nkita
nkita
nkita
        gentitles = set()
nkita
nkita
nkita
        for fn in tqdm(gennames, desc='gen dataset'):
nkita
nkita
nkita
            if fn[3:-5] in DOM:
nkita
nkita
nkita
                gendataset.append(getFileData(f'{wikigen}/{fn}'))
nkita
nkita
nkita
                genfset.append(fn[3:-5])
nkita
nkita
nkita

nkita
nkita
nkita
        for d, fn in tqdm(zip(gendataset, genfset), desc='wikiref titles'):
nkita
nkita
nkita
            for article in d:
nkita
nkita
nkita
                title = article['title']
nkita
nkita
nkita
                title = title.translate(str.maketrans('', '', PUNCTUATIONS)).strip()
nkita
nkita
nkita
                gentitles.add(title)
nkita
nkita
nkita

nkita
nkita
nkita
        # getting xalign data
nkita
nkita
nkita
        xalign = f'{xalign_path}/{ln}'
nkita
nkita
nkita
        alnames = getFileNames(xalign)
nkita
nkita
nkita
        aldataset = []
nkita
nkita
nkita
        altitles = set()
nkita
nkita
nkita
        mapper = f'{index_path}/index_{ln}wiki-latest.db'
nkita
nkita
nkita
        mapper = WikiMapper((mapper))
nkita
nkita
nkita

nkita
nkita
nkita
        for fn in tqdm(alnames, desc='xalign files'):
nkita
nkita
nkita
            if 'test' in fn:
nkita
nkita
nkita
                aldataset.append(getFileData(f'{xalign}/{fn}'))
nkita
nkita
nkita

nkita
nkita
nkita
        for d in tqdm(aldataset, desc='xalign titles'):
nkita
nkita
nkita
            for sent in tqdm(d, desc='sentences'):
nkita
nkita
nkita
                title = mapper.id_to_titles(sent['qid'])
nkita
nkita
nkita

nkita
nkita
nkita
                if len(title) != 0:
nkita
nkita
nkita
                    altitles.add(title[0].translate(str.maketrans('', '', PUNCTUATIONS)).strip())
nkita
nkita
nkita

nkita
nkita
nkita
        intersection = gentitles.intersection(altitles)
nkita
nkita
nkita
        ic(ln, len(intersection))
nkita
nkita
nkita

nkita
nkita
nkita
        tempref = defaultdict(list)
nkita
nkita
nkita
        tempcont = defaultdict(list)
nkita
nkita
nkita

nkita
nkita
nkita
        for d, fn in tqdm(zip(gendataset, genfset), desc='titles from wikiref'):
nkita
nkita
nkita
            for article in d:
nkita
nkita
nkita
                title = article['title']
nkita
nkita
nkita
                title = title.translate(str.maketrans('', '', PUNCTUATIONS)).strip()
nkita
nkita
nkita
                if title in intersection:
nkita
nkita
nkita
                    sections = article['sections']
nkita
nkita
nkita
                    for section in sections:
nkita
nkita
nkita
                        refs = section['references']
nkita
nkita
nkita
                        cont = section['content']
nkita
nkita
nkita

nkita
nkita
nkita
                        if len(cont) != 0:
nkita
nkita
nkita
                            tempcont[title].append(cont)
nkita
nkita
nkita

nkita
nkita
nkita
                        for ref in refs:
nkita
nkita
nkita
                            tempref[title].append(ref)
nkita
nkita
nkita

nkita
nkita
nkita
        tempal = defaultdict(list)
nkita
nkita
nkita

nkita
nkita
nkita
        for d in tqdm(aldataset, desc='xalign'):
nkita
nkita
nkita
            for sent in tqdm(d, desc='article'):
nkita
nkita
nkita
                qid = sent['qid']
nkita
nkita
nkita
                title = mapper.id_to_titles(qid)
nkita
nkita
nkita
                if len(title) != 0:
nkita
nkita
nkita
                    title = title[0].translate(str.maketrans('','',PUNCTUATIONS)).strip()
nkita
nkita
nkita
                    if title in intersection:
nkita
nkita
nkita
                        tempal[title].append({
nkita
nkita
nkita
                            'sentence': sent['sentence'],
nkita
nkita
nkita
                            'fact': sent['facts'],
nkita
nkita
nkita
                            'qid': qid
nkita
nkita
nkita
                        })
nkita
nkita
nkita
                        # tempsent[title].append(sent['sentence'])
nkita
nkita
nkita
                        # tempal[title].append(sent['facts'])
nkita
nkita
nkita
                        # tempqid[title].add(qid)
nkita
nkita
nkita

nkita
nkita
nkita
        output = []
nkita
nkita
nkita
        for key in list(intersection):
nkita
nkita
nkita
            for stuff in tempal[key]:
nkita
nkita
nkita
                output.append({
nkita
nkita
nkita
                    "title": key,
nkita
nkita
nkita
                    "qid": stuff['qid'],
nkita
nkita
nkita
                    "xalign_sent": stuff['sentence'],
nkita
nkita
nkita
                    "xalign_facts": stuff['fact'],
nkita
nkita
nkita
                    "xwgen_refs": tempref[key],
nkita
nkita
nkita
                    "content": tempcont[key]
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
        fout = open(f'{output_path}/{ln}_test.json', 'w+')
nkita
nkita
nkita
        ic(ln, len(output))
nkita
nkita
nkita
        for dic in output:
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
    parser.add_argument('--xalign_path')
nkita
nkita
nkita
    parser.add_argument('--xwikigen_path')
nkita
nkita
nkita
    parser.add_argument('--index_path')
nkita
nkita
nkita
    parser.add_argument('--output_path')
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

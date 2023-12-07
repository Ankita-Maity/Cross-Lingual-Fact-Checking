import os
nkita
nkita
nkita
import json
nkita
nkita
nkita
import stanza
nkita
nkita
nkita

nkita
nkita
nkita
from icecream import ic
nkita
nkita
nkita
from tqdm import tqdm
nkita
nkita
nkita
from langdetect import detect
nkita
nkita
nkita
from indicnlp.tokenize.sentence_tokenize import sentence_split
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
# @profile
nkita
nkita
nkita
def selectSentences(text,qid,entity):
nkita
nkita
nkita

nkita
nkita
nkita
    if ref.strip() == '':
nkita
nkita
nkita
        return []
nkita
nkita
nkita
    reflang = detect(ref)
nkita
nkita
nkita
    if reflang not in BASELANG:
nkita
nkita
nkita
        return []
nkita
nkita
nkita
    selected_sentences = []
nkita
nkita
nkita
    sentences = sentence_split(ref, lang=reflang)
nkita
nkita
nkita
    for sent in sentences:
nkita
nkita
nkita
        if reflang == 'hi':
nkita
nkita
nkita
            doc = hinlp(sent)
nkita
nkita
nkita
        elif reflang == 'en':
nkita
nkita
nkita
            doc = ennlp(sent)
nkita
nkita
nkita
        elif reflang == 'ta':
nkita
nkita
nkita
            doc = tanlp(sent)
nkita
nkita
nkita
        elif reflang == 'te':
nkita
nkita
nkita
            doc = tenlp(sent)
nkita
nkita
nkita
        elif reflang == 'mr':
nkita
nkita
nkita
            doc = mrnlp(sent)
nkita
nkita
nkita
        else:
nkita
nkita
nkita
            pass
nkita
nkita
nkita
        hasNoun = False
nkita
nkita
nkita
        hasVerb = False
nkita
nkita
nkita
        for s in doc.sentences:
nkita
nkita
nkita
            sen = []
nkita
nkita
nkita
            for w in s.words:
nkita
nkita
nkita
                if w.upos == 'PROPN':
nkita
nkita
nkita
                    hasNoun = True
nkita
nkita
nkita
                elif w.upos == 'VERB':
nkita
nkita
nkita
                    hasVerb = True
nkita
nkita
nkita
                sen.append(w.text)
nkita
nkita
nkita
            if (hasNoun or hasVerb) and len(sen) >= 5 and len(sen) <= 100:
nkita
nkita
nkita
                selected_sentences.append({
nkita
nkita
nkita
                    'qid': qid,
nkita
nkita
nkita
                    'refs': ' '.join(sen),
nkita
nkita
nkita
                    'entity': entity })
nkita
nkita
nkita
            else:
nkita
nkita
nkita
                pass
nkita
nkita
nkita
    return selected_sentences
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
    LANG = {'hi','ta','mr'}
nkita
nkita
nkita
    BASELANG = {'en','hi','ta','mr','te'}
nkita
nkita
nkita

nkita
nkita
nkita
    # language specific tokenizer and pos tagger
nkita
nkita
nkita
    ennlp = stanza.Pipeline(lang='en', processors='tokenize,pos,mwt')
nkita
nkita
nkita
    hinlp = stanza.Pipeline(lang='hi', processors='tokenize,pos')
nkita
nkita
nkita
    tanlp = stanza.Pipeline(lang='ta', processors='tokenize,pos,mwt')
nkita
nkita
nkita
    tenlp = stanza.Pipeline(lang='te', processors='tokenize,pos')
nkita
nkita
nkita
    mrnlp = stanza.Pipeline(lang='mr', processors='tokenize,pos,mwt')
nkita
nkita
nkita

nkita
nkita
nkita
    DNAME = f'../Datasets/intersec/intersection/'
nkita
nkita
nkita
    fnames = getFileNames(DNAME)
nkita
nkita
nkita
    dataset = []
nkita
nkita
nkita
    fset = []
nkita
nkita
nkita
    # extracting specific file names and dataset
nkita
nkita
nkita
    for fn in fnames:
nkita
nkita
nkita
        if fn[0:2] in LANG:
nkita
nkita
nkita
            dataset.append(getFileData(f'{DNAME}/{fn}'))
nkita
nkita
nkita
            fset.append(fn[0:2])
nkita
nkita
nkita

nkita
nkita
nkita
    print(fnames)
nkita
nkita
nkita

nkita
nkita
nkita
    for d, fn in zip(dataset, fset):
nkita
nkita
nkita
        print(f"Processing lang: {fn}")
nkita
nkita
nkita
        cnt = 0
nkita
nkita
nkita
        # for each article in dataset
nkita
nkita
nkita
        for article in tqdm(d):
nkita
nkita
nkita
            cnt += 1
nkita
nkita
nkita
            refs = article['reftext']
nkita
nkita
nkita
            articleQid = article['qid']
nkita
nkita
nkita
            articleEntity = article['title']
nkita
nkita
nkita
            references = []
nkita
nkita
nkita
            refcnt = 0
nkita
nkita
nkita
            # for each reference in article
nkita
nkita
nkita
            for ref in refs:
nkita
nkita
nkita
                refcnt += 1
nkita
nkita
nkita
                # select specific reference sentences
nkita
nkita
nkita
                ref_sentences = selectSentences(ref,articleQid,articleEntity)
nkita
nkita
nkita
                if len(ref_sentences) != 0:
nkita
nkita
nkita
                    references.extend(ref_sentences)
nkita
nkita
nkita
                if refcnt == 3:
nkita
nkita
nkita
                    break
nkita
nkita
nkita
            with open(f'dataset_{fn}.json','a+') as fout:
nkita
nkita
nkita
                for dic in references:
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
            if cnt == 2:
nkita
nkita
nkita
                break
nkita
nkita
nkita

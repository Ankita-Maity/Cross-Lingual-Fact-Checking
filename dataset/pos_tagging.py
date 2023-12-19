import os



import json



import stanza







from icecream import ic



from tqdm import tqdm



from langdetect import detect



from indicnlp.tokenize.sentence_tokenize import sentence_split







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







# @profile



def selectSentences(text,qid,entity):







    if ref.strip() == '':



        return []



    reflang = detect(ref)



    if reflang not in BASELANG:



        return []



    selected_sentences = []



    sentences = sentence_split(ref, lang=reflang)



    for sent in sentences:



        if reflang == 'hi':



            doc = hinlp(sent)



        elif reflang == 'en':



            doc = ennlp(sent)



        elif reflang == 'ta':



            doc = tanlp(sent)



        elif reflang == 'te':



            doc = tenlp(sent)



        elif reflang == 'mr':



            doc = mrnlp(sent)



        else:



            pass



        hasNoun = False



        hasVerb = False



        for s in doc.sentences:



            sen = []



            for w in s.words:



                if w.upos == 'PROPN':



                    hasNoun = True



                elif w.upos == 'VERB':



                    hasVerb = True



                sen.append(w.text)



            if (hasNoun or hasVerb) and len(sen) >= 5 and len(sen) <= 100:



                selected_sentences.append({



                    'qid': qid,



                    'refs': ' '.join(sen),



                    'entity': entity })



            else:



                pass



    return selected_sentences







if __name__ == "__main__":



    LANG = {'hi','ta','mr'}



    BASELANG = {'en','hi','ta','mr','te'}







    # language specific tokenizer and pos tagger



    ennlp = stanza.Pipeline(lang='en', processors='tokenize,pos,mwt')



    hinlp = stanza.Pipeline(lang='hi', processors='tokenize,pos')



    tanlp = stanza.Pipeline(lang='ta', processors='tokenize,pos,mwt')



    tenlp = stanza.Pipeline(lang='te', processors='tokenize,pos')



    mrnlp = stanza.Pipeline(lang='mr', processors='tokenize,pos,mwt')







    DNAME = f'../Datasets/intersec/intersection/'



    fnames = getFileNames(DNAME)



    dataset = []



    fset = []



    # extracting specific file names and dataset



    for fn in fnames:



        if fn[0:2] in LANG:



            dataset.append(getFileData(f'{DNAME}/{fn}'))



            fset.append(fn[0:2])







    print(fnames)







    for d, fn in zip(dataset, fset):



        print(f"Processing lang: {fn}")



        cnt = 0



        # for each article in dataset



        for article in tqdm(d):



            cnt += 1



            refs = article['reftext']



            articleQid = article['qid']



            articleEntity = article['title']



            references = []



            refcnt = 0



            # for each reference in article



            for ref in refs:



                refcnt += 1



                # select specific reference sentences



                ref_sentences = selectSentences(ref,articleQid,articleEntity)



                if len(ref_sentences) != 0:



                    references.extend(ref_sentences)



                if refcnt == 3:



                    break



            with open(f'dataset_{fn}.json','a+') as fout:



                for dic in references:



                    fout.write(json.dumps(dic, ensure_ascii=False))



                    fout.write('\n')







            if cnt == 2:



                break




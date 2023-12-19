import os



import json



import argparse







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











def main(args):







    data_path = args.data_path



    fnames = getFileNames(data_path)







    for fn in tqdm(fnames):



        path = f'{data_path}/{fn}'



        data = getFileData(path)



        dataset = []







        for d in data:



            facts = d['xalign_facts']







            new_facts = []



            for f in facts:



                new_facts.append([f[0], f[1]])







            dataset.append({



                'title': d['title'],



                'qid': d['qid'],



                'xalign_sent': d['xalign_sent'],



                'xalign_facts': new_facts,



                'xwgen_refs': d['xwgen_refs'],



                'content': d['content']



            })







        fout = open(path, 'w+')



        for dic in dataset:



            fout.write(json.dumps(dic, ensure_ascii=False))



            fout.write('\n')







if __name__ == "__main__":







    parser = argparse.ArgumentParser()



    parser.add_argument('--data_path')







    args = parser.parse_args()



    main(args)




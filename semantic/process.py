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





def write(data, path):

    f = open(path, 'w+')

    for dic in data:

        f.write(json.dumps(dic, ensure_ascii=False))

        f.write('\n')







def main(args):



    processed_path = args.processed_path

    intersection_path = args.intersection_path

    output_path = args.output_path



    if not os.path.exists(output_path):

        os.mkdir(output_path)



    pr_names = set(getFileNames(processed_path))

    in_names = set(getFileNames(intersection_path))



    fnames = list(pr_names.intersection(in_names))



    for fn in fnames:



        pr_data = getFileData(f'{processed_path}/{fn}')

        in_data = getFileData(f'{intersection_path}/{fn}')



        art_wise = []



        per_art = defaultdict(list)

        for d in pr_data:

            per_art[d['title']].append(d['xwgen_refs'])



        # ic([list(per_art.keys())[0]])

        per_in = defaultdict(list)

        for d in in_data:

            per_in[d['title']].append({'fact': d['xalign_facts'], 'sent': d['xalign_sent']})

            # ['title', 'qid', 'xalign_sent', 'xalign_facts', 'xwgen_refs', 'content']



            # art_wise.append({

            #     "article": d,

            #     "references": per_art[d['title']]

            # })



        for title in per_in.keys():

            art_wise.append({

                "article": per_in[title],

                "references": per_art[title],

                "title": title

            })



        write(art_wise, f'{output_path}/{fn}')





if __name__ == "__main__":



    parser = argparse.ArgumentParser()

    parser.add_argument('--processed_path')

    parser.add_argument('--intersection_path')

    parser.add_argument('--output_path')



    args = parser.parse_args()

    main(args)


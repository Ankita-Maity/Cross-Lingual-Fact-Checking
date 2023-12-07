import json

from icecream import ic
from torch.utils.data import Dataset

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer
import pandas as pd
import json
import torch

class Dataset1(Dataset):
    def __init__(self, data_path, tokenizer, max_source_length, max_target_length, is_mt5):
        fp = open(data_path, 'r')
        self.df = [json.loads(line, strict=False) for line in fp.readlines()]
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.is_mt5 = is_mt5
        self.languages_map = {
            'bn': 'bn_IN',
            'de': 'de_DE',
            'en': 'en_XX',
            'es': 'es_XX',
            'fr': 'fr_XX',
            'gu': 'gu_IN',
            'hi': 'hi_IN',
            'it': 'it_IT',
            'kn': 'kn_IN',
            'ml': 'ml_IN',
            'mr': 'mr_IN',
            'or': 'or_IN',
            'pa': 'pa_IN',
            'ta': 'ta_IN',
            'te': 'te_IN',
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        lang = self.df[idx]['lang']
        title = self.df[idx]['title']
        xalign_sent = self.df[idx]['xalign_sent']
        xalign_facts = self.df[idx]['xalign_facts']

        if lang not in self.languages_map:
            lang='en'
        # xlang = lang
        lang = self.languages_map[lang]

        fact_text = ""
        for f in xalign_facts:
            relation = f[0]
            relation = relation.strip().replace(' ','_')
            obj = f[1]
            fact_text = f'{fact_text} {relation} {obj} <SEP>'

        input_text = f'{lang} {title} {xalign_sent}'

        target_text = fact_text

        input_text = input_text.strip()
        target_text = target_text.strip()

        # ic(input_text, target_text)
        # self.tokenizer.add_special_tokens({'sep_token': self.septoken})

        input_encoding = self.tokenizer(input_text, return_tensors='pt', max_length=self.max_source_length ,padding='max_length', truncation=True)
        target_encoding = self.tokenizer(lang + target_text, return_tensors='pt', max_length=self.max_target_length ,padding='max_length', truncation=True)

        input_ids, attention_mask = input_encoding['input_ids'], input_encoding['attention_mask']
        labels = target_encoding['input_ids']

        if self.is_mt5 == 1:
            labels[labels == self.tokenizer.pad_token_id] = -100    # for ignoring the cross-entropy loss at padding locations


        return {'input_ids': input_ids.squeeze(), 'attention_mask': attention_mask.squeeze(), 'labels': labels.squeeze(), 'lang': lang}

class DataModule(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer_name_or_path)

    def setup(self, stage=None):
        # ic("loading train")
        self.train = Dataset1(self.hparams.train_path, self.tokenizer, self.hparams.max_source_length, self.hparams.max_target_length, self.hparams.is_mt5)
        # ic("loading val")
        self.val = Dataset1(self.hparams.val_path, self.tokenizer, self.hparams.max_source_length, self.hparams.max_target_length, self.hparams.is_mt5)
        # ic("loading test")
        self.test = Dataset1(self.hparams.test_path, self.tokenizer, self.hparams.max_source_length, self.hparams.max_target_length, self.hparams.is_mt5)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.hparams.train_batch_size, num_workers=1,shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.hparams.val_batch_size, num_workers=1,shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.hparams.test_batch_size, num_workers=1,shuffle=False)

    def predict_dataloader(self):
        return self.test_dataloader()

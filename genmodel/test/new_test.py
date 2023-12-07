from torch.utils.data import Dataset, DataLoader
nkita
nkita
nkita
import pytorch_lightning as pl
nkita
nkita
nkita
from transformers import AutoTokenizer
nkita
nkita
nkita
import pandas as pd
nkita
nkita
nkita
import json
nkita
nkita
nkita
import pytorch_lightning as pl
nkita
nkita
nkita
from pytorch_lightning.loggers import WandbLogger
nkita
nkita
nkita
from indicnlp.transliterate import unicode_transliterate
nkita
nkita
nkita
from transformers import MBartForConditionalGeneration, MT5ForConditionalGeneration, AutoConfig, AutoModelForSeq2SeqLM, MBartTokenizer
nkita
nkita
nkita
import torch
nkita
nkita
nkita
import argparse
nkita
nkita
nkita
from rouge import Rouge
nkita
nkita
nkita
import sys
nkita
nkita
nkita
sys.setrecursionlimit(1024 * 1024 + 10)
nkita
nkita
nkita

nkita
nkita
nkita
class Dataset1(Dataset):
nkita
nkita
nkita
    def __init__(self, data_path, tokenizer, max_source_length, max_target_length, is_mt5):
nkita
nkita
nkita
        fp = open(data_path, 'r')
nkita
nkita
nkita
        self.df = [json.loads(line, strict=False) for line in fp.readlines()]
nkita
nkita
nkita
        self.tokenizer = tokenizer
nkita
nkita
nkita
        self.max_source_length = max_source_length
nkita
nkita
nkita
        self.max_target_length = max_target_length
nkita
nkita
nkita
        self.is_mt5 = is_mt5
nkita
nkita
nkita
        self.languages_map = {
nkita
nkita
nkita
            'bn': {0:'bn_IN'},
nkita
nkita
nkita
            'de': {0:'de_DE'},
nkita
nkita
nkita
            'en': {0:'en_XX'},
nkita
nkita
nkita
            'es': {0:'es_XX'},
nkita
nkita
nkita
            'fr': {0:'fr_XX'},
nkita
nkita
nkita
            'gu': {0:'gu_IN'},
nkita
nkita
nkita
            'hi': {0:'hi_IN'},
nkita
nkita
nkita
            'it': {0:'it_IT'},
nkita
nkita
nkita
            'kn': {0:'kn_IN'},
nkita
nkita
nkita
            'ml': {0:'ml_IN'},
nkita
nkita
nkita
            'mr': {0:'mr_IN'},
nkita
nkita
nkita
            'or': {0:'or_IN'},
nkita
nkita
nkita
            'pa': {0:'pa_IN'},
nkita
nkita
nkita
            'ta': {0:'ta_IN'},
nkita
nkita
nkita
            'te': {0:'te_IN'}
nkita
nkita
nkita
        }
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
    def __len__(self):
nkita
nkita
nkita
        return len(self.df)
nkita
nkita
nkita

nkita
nkita
nkita
    def __getitem__(self, idx):
nkita
nkita
nkita

nkita
nkita
nkita
        lang = self.df[idx]['lang']
nkita
nkita
nkita
        title = self.df[idx]['title']
nkita
nkita
nkita
        xalign_sent = self.df[idx]['xwgen_refs']
nkita
nkita
nkita
        # xalign_facts = self.df[idx]['xalign_facts']
nkita
nkita
nkita

nkita
nkita
nkita
        if lang not in self.languages_map:
nkita
nkita
nkita
            lang='en'
nkita
nkita
nkita

nkita
nkita
nkita
        lang = self.languages_map[lang][0]
nkita
nkita
nkita

nkita
nkita
nkita
        # fact_text = ""
nkita
nkita
nkita
        # for f in xalign_facts:
nkita
nkita
nkita
        #     relation = f[0]
nkita
nkita
nkita
        #     relation = relation.strip().replace(' ','_')
nkita
nkita
nkita
        #     obj = f[1]
nkita
nkita
nkita
        #     fact_text = f'{fact_text} {relation} {obj} <SEP>'
nkita
nkita
nkita

nkita
nkita
nkita
        input_text = f'{lang} {title} {xalign_sent}'
nkita
nkita
nkita
        # target_text = fact_text
nkita
nkita
nkita

nkita
nkita
nkita
        input_text = input_text.strip()
nkita
nkita
nkita
        # target_text = target_text.strip()
nkita
nkita
nkita

nkita
nkita
nkita
        # if self.is_mt5:
nkita
nkita
nkita
        #     self.tokenizer.add_special_tokens({'sep_token': '</s>'})
nkita
nkita
nkita

nkita
nkita
nkita

nkita
nkita
nkita
        input_encoding = self.tokenizer(input_text, return_tensors='pt', max_length=self.max_source_length ,padding='max_length', truncation=True)
nkita
nkita
nkita
        # target_encoding = self.tokenizer(lang + target_text, return_tensors='pt', max_length=self.max_target_length ,padding='max_length', truncation=True)
nkita
nkita
nkita

nkita
nkita
nkita
        input_ids, attention_mask = input_encoding['input_ids'], input_encoding['attention_mask']
nkita
nkita
nkita
        # labels = target_encoding['input_ids']
nkita
nkita
nkita

nkita
nkita
nkita
        # if self.is_mt5:
nkita
nkita
nkita
        #     labels[labels == self.tokenizer.pad_token_id] = -100    # for ignoring the cross-entropy loss at padding locations
nkita
nkita
nkita

nkita
nkita
nkita

nkita
nkita
nkita
        # return {'input_ids': input_ids.squeeze(), 'attention_mask': attention_mask.squeeze(), 'labels': labels.squeeze(), 'lang': lang}
nkita
nkita
nkita
        return {'input_ids': input_ids.squeeze(), 'attention_mask': attention_mask.squeeze(), 'lang': lang}
nkita
nkita
nkita

nkita
nkita
nkita

nkita
nkita
nkita
class DataModule(pl.LightningDataModule):
nkita
nkita
nkita
    def __init__(self, *args, **kwargs):
nkita
nkita
nkita
        super().__init__()
nkita
nkita
nkita
        self.save_hyperparameters()
nkita
nkita
nkita
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer_name_or_path)
nkita
nkita
nkita

nkita
nkita
nkita
    def setup(self, stage=None):
nkita
nkita
nkita
        self.train = Dataset1(self.hparams.train_path, self.tokenizer, self.hparams.max_source_length, self.hparams.max_target_length, self.hparams.is_mt5)
nkita
nkita
nkita
        self.val = Dataset1(self.hparams.val_path, self.tokenizer, self.hparams.max_source_length, self.hparams.max_target_length, self.hparams.is_mt5)
nkita
nkita
nkita
        self.test = Dataset1(self.hparams.test_path, self.tokenizer, self.hparams.max_source_length, self.hparams.max_target_length, self.hparams.is_mt5)
nkita
nkita
nkita

nkita
nkita
nkita
    def train_dataloader(self):
nkita
nkita
nkita
        return DataLoader(self.train, batch_size=self.hparams.train_batch_size, num_workers=1,shuffle=True)
nkita
nkita
nkita

nkita
nkita
nkita
    def val_dataloader(self):
nkita
nkita
nkita
        return DataLoader(self.val, batch_size=self.hparams.val_batch_size, num_workers=1,shuffle=False)
nkita
nkita
nkita

nkita
nkita
nkita
    def test_dataloader(self):
nkita
nkita
nkita
        return DataLoader(self.test, batch_size=self.hparams.test_batch_size, num_workers=1,shuffle=False)
nkita
nkita
nkita

nkita
nkita
nkita
    def predict_dataloader(self):
nkita
nkita
nkita
        return self.test_dataloader()
nkita
nkita
nkita

nkita
nkita
nkita

nkita
nkita
nkita
class Summarizer(pl.LightningModule):
nkita
nkita
nkita
    def __init__(self, *args, **kwargs):
nkita
nkita
nkita
        super().__init__()
nkita
nkita
nkita
        # print(self.hparams)
nkita
nkita
nkita
        self.save_hyperparameters()
nkita
nkita
nkita
        self.rouge = Rouge()
nkita
nkita
nkita
        # self.config = AutoConfig.from_pretrained(self.hparams.config)
nkita
nkita
nkita
        # print(self.hparams)
nkita
nkita
nkita
        if self.hparams.is_mt5:
nkita
nkita
nkita
            self.model = MT5ForConditionalGeneration.from_pretrained(self.hparams.model_name_or_path)
nkita
nkita
nkita
        else:
nkita
nkita
nkita
            self.model = MBartForConditionalGeneration.from_pretrained(self.hparams.model_name_or_path)
nkita
nkita
nkita

nkita
nkita
nkita
        self.languages_map = {
nkita
nkita
nkita
            'bn': {0:'bn_IN'},
nkita
nkita
nkita
            'de': {0:'de_DE'},
nkita
nkita
nkita
            'en': {0:'en_XX'},
nkita
nkita
nkita
            'es': {0:'es_XX'},
nkita
nkita
nkita
            'fr': {0:'fr_XX'},
nkita
nkita
nkita
            'gu': {0:'gu_IN'},
nkita
nkita
nkita
            'hi': {0:'hi_IN'},
nkita
nkita
nkita
            'it': {0:'it_IT'},
nkita
nkita
nkita
            'kn': {0:'kn_IN'},
nkita
nkita
nkita
            'ml': {0:'ml_IN'},
nkita
nkita
nkita
            'mr': {0:'mr_IN'},
nkita
nkita
nkita
            'or': {0:'or_IN'},
nkita
nkita
nkita
            'pa': {0:'pa_IN'},
nkita
nkita
nkita
            'ta': {0:'ta_IN'},
nkita
nkita
nkita
            'te': {0:'te_IN'}
nkita
nkita
nkita
        }
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
    def forward(self, input_ids, attention_mask):
nkita
nkita
nkita
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
nkita
nkita
nkita
        return outputs
nkita
nkita
nkita

nkita
nkita
nkita
    def _step(self, batch):
nkita
nkita
nkita
        input_ids, attention_mask, src_lang = batch['input_ids'], batch['attention_mask'], batch['lang']
nkita
nkita
nkita
        outputs = self(input_ids, attention_mask)
nkita
nkita
nkita
        loss = outputs[0]
nkita
nkita
nkita
        return loss
nkita
nkita
nkita

nkita
nkita
nkita
    def _generative_step(self, batch):
nkita
nkita
nkita

nkita
nkita
nkita
        if not self.hparams.is_mt5:
nkita
nkita
nkita
            try:
nkita
nkita
nkita
                token_id = self.hparams.tokenizer.lang_code_to_id[batch['lang'][0]]
nkita
nkita
nkita
                self.hparams.tokenizer.tgt_lang = batch['lang'][0]
nkita
nkita
nkita
            except:
nkita
nkita
nkita
                token_id = 250044
nkita
nkita
nkita
                self.hparams.tokenizer.tgt_lang = 'ta_IN'
nkita
nkita
nkita

nkita
nkita
nkita
            generated_ids = self.model.generate(
nkita
nkita
nkita
                input_ids=batch['input_ids'],
nkita
nkita
nkita
                attention_mask=batch['attention_mask'],
nkita
nkita
nkita
                use_cache=True,
nkita
nkita
nkita
                num_beams=self.hparams.eval_beams,
nkita
nkita
nkita
                forced_bos_token_id=token_id,
nkita
nkita
nkita
                max_length=self.hparams.tgt_max_seq_len #understand above 3 arguments
nkita
nkita
nkita
                )
nkita
nkita
nkita
        else:
nkita
nkita
nkita
            self.hparams.tokenizer.tgt_lang = batch['lang'][0]
nkita
nkita
nkita
            generated_ids = self.model.generate(
nkita
nkita
nkita
                input_ids=batch['input_ids'],
nkita
nkita
nkita
                attention_mask=batch['attention_mask'],
nkita
nkita
nkita
                use_cache=True,
nkita
nkita
nkita
                num_beams=self.hparams.eval_beams,
nkita
nkita
nkita
                max_length=self.hparams.tgt_max_seq_len #understand above 3 arguments
nkita
nkita
nkita
                )
nkita
nkita
nkita

nkita
nkita
nkita

nkita
nkita
nkita
        input_text = self.hparams.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
nkita
nkita
nkita
        pred_text = self.hparams.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
nkita
nkita
nkita
        # if self.hparams.is_mt5:
nkita
nkita
nkita
        #     batch['labels'][batch['labels'] == -100] = self.hparams.tokenizer.pad_token_id
nkita
nkita
nkita
        # ref_text = self.hparams.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
nkita
nkita
nkita

nkita
nkita
nkita
        return input_text, pred_text, batch['lang']
nkita
nkita
nkita

nkita
nkita
nkita

nkita
nkita
nkita
    def training_step(self, batch, batch_idx):
nkita
nkita
nkita
        loss = self._step(batch)
nkita
nkita
nkita
        self.log("train_loss", loss, on_epoch=True)
nkita
nkita
nkita
        return {'loss': loss}
nkita
nkita
nkita

nkita
nkita
nkita
    def validation_step(self, batch, batch_idx):
nkita
nkita
nkita
        loss = self._step(batch)
nkita
nkita
nkita
        # input_text, pred_text, ref_text = self._generative_step(batch)
nkita
nkita
nkita
        self.log("val_loss", loss, on_epoch=True)
nkita
nkita
nkita
        return
nkita
nkita
nkita

nkita
nkita
nkita
    def validation_epoch_end(self, outputs):
nkita
nkita
nkita

nkita
nkita
nkita
        return
nkita
nkita
nkita

nkita
nkita
nkita

nkita
nkita
nkita
    def predict_step(self, batch, batch_idx):
nkita
nkita
nkita
        input_text, pred_text, src_lang = self._generative_step(batch)
nkita
nkita
nkita
        return {'input_text': input_text, 'pred_text': pred_text}
nkita
nkita
nkita

nkita
nkita
nkita
    def get_native_text_from_unified_script(self, unified_text, lang):
nkita
nkita
nkita
        return unicode_transliterate.UnicodeIndicTransliterator.transliterate(unified_text, "hi", lang)
nkita
nkita
nkita

nkita
nkita
nkita
    def process_for_rouge(self, text, lang):
nkita
nkita
nkita
        native_text = text
nkita
nkita
nkita
        if lang!='en':
nkita
nkita
nkita
            # convert unified script to native langauge text
nkita
nkita
nkita
            native_text = self.get_native_text_from_unified_script(text, lang)
nkita
nkita
nkita
        native_text = native_text.strip()
nkita
nkita
nkita
        # as input and predicted text are already space tokenized
nkita
nkita
nkita
        native_text = ' '.join([x for x in native_text.split()])
nkita
nkita
nkita
        return native_text
nkita
nkita
nkita

nkita
nkita
nkita
    def test_step(self, batch, batch_idx):
nkita
nkita
nkita
        loss = self._step(batch)
nkita
nkita
nkita
        input_text, pred_text, lang = self._generative_step(batch)
nkita
nkita
nkita
        return {'test_loss': loss, 'input_text': input_text, 'pred_text': pred_text, 'lang': lang}
nkita
nkita
nkita

nkita
nkita
nkita
    def test_epoch_end(self, outputs):
nkita
nkita
nkita
        input_texts = []
nkita
nkita
nkita
        pred_texts = []
nkita
nkita
nkita
        # ref_texts = []
nkita
nkita
nkita
        langs = []
nkita
nkita
nkita
        for x in outputs:
nkita
nkita
nkita
            if x['pred_text'][0] == '':
nkita
nkita
nkita
                x['pred_text'][0] = 'pred_text'
nkita
nkita
nkita
            # if x['ref_text'][0] == '':
nkita
nkita
nkita
            #     x['ref_text'][0] = 'ref_text'
nkita
nkita
nkita
            input_texts.extend(x['input_text'])
nkita
nkita
nkita
            pred_texts.extend(x['pred_text'])
nkita
nkita
nkita
            # ref_texts.extend(x['ref_text'])
nkita
nkita
nkita
            langs.extend(x['lang'])
nkita
nkita
nkita

nkita
nkita
nkita

nkita
nkita
nkita
        df_to_write = pd.DataFrame()
nkita
nkita
nkita
        df_to_write['input_texts'] = input_texts
nkita
nkita
nkita
        df_to_write['lang'] = langs
nkita
nkita
nkita
        # df_to_write['ref_text'] = ref_texts
nkita
nkita
nkita
        df_to_write['pred_text'] = pred_texts
nkita
nkita
nkita
        df_to_write.to_csv(method + '_' + model_name.replace('google/', '').replace('facebook/', '') + '.csv', index=False)
nkita
nkita
nkita

nkita
nkita
nkita
        t = model_name.replace('google/', '').replace('facebook/', '')
nkita
nkita
nkita
        logger.log_text(f'{method}_{t}_predictions', dataframe=df_to_write)
nkita
nkita
nkita

nkita
nkita
nkita

nkita
nkita
nkita
    def configure_optimizers(self):
nkita
nkita
nkita
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
nkita
nkita
nkita

nkita
nkita
nkita
    @staticmethod
nkita
nkita
nkita
    def add_model_specific_args(parent_parser):
nkita
nkita
nkita
        parser = parent_parser.add_argument_group('Bart Fine-tuning Parameters')
nkita
nkita
nkita
        parser.add_argument('--learning_rate', default=2e-5, type=float)
nkita
nkita
nkita
        parser.add_argument('--model_name_or_path', default='bart-base', type=str)
nkita
nkita
nkita
        parser.add_argument('--eval_beams', default=3, type=int)
nkita
nkita
nkita
        parser.add_argument('--tgt_max_seq_len', default=128, type=int)
nkita
nkita
nkita
        parser.add_argument('--tokenizer', default='bart-base', type=str)
nkita
nkita
nkita
        return parent_parser
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
    parser = argparse.ArgumentParser(description='Input parameters for extractive stage')
nkita
nkita
nkita
    parser.add_argument('--batch_size', default=1, type=int, help='test_batch_size')
nkita
nkita
nkita
    parser.add_argument('--train_path', default=None, help='path to input json file for a given domain in given language')
nkita
nkita
nkita
    parser.add_argument('--val_path', default=None, help='path to intermediate output json file for a given domain in given language')
nkita
nkita
nkita
    parser.add_argument('--test_path', default=None, help='path to output json file for a given domain in given language')
nkita
nkita
nkita
    parser.add_argument('--config', default=None, help='which config file to use')
nkita
nkita
nkita
    parser.add_argument('--tokenizer', default='facebook/mbart-large-50', help='which tokenizer to use')
nkita
nkita
nkita
    parser.add_argument('--model', default='facebook/mbart-large-50', help='which model to use')
nkita
nkita
nkita
    # parser.add_argument('--target_lang', default='hi_IN', help='what is the target language')
nkita
nkita
nkita
    parser.add_argument('--ckpt_path', help='ckpt path')
nkita
nkita
nkita
    parser.add_argument('--exp_name', help='experimet name')
nkita
nkita
nkita
    parser.add_argument('--is_mt5', type=int, help='is the model mt5')
nkita
nkita
nkita
    parser.add_argument('--prediction_path', default='preds.txt', help='path to save prediction file')
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
    prediction_path = args.prediction_path
nkita
nkita
nkita

nkita
nkita
nkita
    ckpt_path = args.ckpt_path
nkita
nkita
nkita
    ckpt_path_1 = ckpt_path.split('/')[-1]
nkita
nkita
nkita

nkita
nkita
nkita
    method = args.exp_name
nkita
nkita
nkita
    model_name = args.exp_name
nkita
nkita
nkita
    is_mt5 = args.is_mt5
nkita
nkita
nkita

nkita
nkita
nkita
    print('-----------------------------------------------------------------------------------------------------------')
nkita
nkita
nkita
    print(method, model_name)
nkita
nkita
nkita

nkita
nkita
nkita
    train_path = args.train_path
nkita
nkita
nkita
    test_path = args.test_path
nkita
nkita
nkita
    val_path= args.val_path
nkita
nkita
nkita

nkita
nkita
nkita
    if 'mt5' in model_name:
nkita
nkita
nkita
        tokenizer = 'google/mt5-small'
nkita
nkita
nkita
        model_name = 'google/mt5-small'
nkita
nkita
nkita
    else:
nkita
nkita
nkita
        tokenizer = 'facebook/mbart-large-50'
nkita
nkita
nkita
        model_name = 'facebook/mbart-large-50'
nkita
nkita
nkita

nkita
nkita
nkita
    dm_hparams = dict(
nkita
nkita
nkita
            train_path=train_path,
nkita
nkita
nkita
            val_path=val_path,
nkita
nkita
nkita
            test_path=test_path,
nkita
nkita
nkita
            tokenizer_name_or_path=tokenizer,
nkita
nkita
nkita
            is_mt5=is_mt5,
nkita
nkita
nkita
            max_source_length=512,
nkita
nkita
nkita
            max_target_length=512,
nkita
nkita
nkita
            train_batch_size=1,
nkita
nkita
nkita
            val_batch_size=1,
nkita
nkita
nkita
            test_batch_size=args.batch_size
nkita
nkita
nkita
            )
nkita
nkita
nkita
    dm = DataModule(**dm_hparams)
nkita
nkita
nkita

nkita
nkita
nkita
    model_hparams = dict(
nkita
nkita
nkita
            learning_rate=2e-5,
nkita
nkita
nkita
            model_name_or_path=model_name,
nkita
nkita
nkita
            eval_beams=4,
nkita
nkita
nkita
            is_mt5=is_mt5,
nkita
nkita
nkita
            tgt_max_seq_len=512,
nkita
nkita
nkita
            tokenizer=dm.tokenizer,
nkita
nkita
nkita
        )
nkita
nkita
nkita

nkita
nkita
nkita
    model = Summarizer(**model_hparams)
nkita
nkita
nkita
    logger=WandbLogger(name='inference_' + method +  '_' + model_name, save_dir='./', project='factver', log_model=False)
nkita
nkita
nkita
    trainer = pl.Trainer(gpus=1, logger=logger)
nkita
nkita
nkita

nkita
nkita
nkita
    model = model.load_from_checkpoint(ckpt_path)
nkita
nkita
nkita
    results = trainer.test(model=model, datamodule=dm, verbose=True)
nkita
nkita
nkita
    print('-----------------------------------------------------------------------------------------------------------')
nkita
nkita
nkita

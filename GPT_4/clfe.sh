unzip clfe_data.zip
nkita
cd clfe_data
nkita

nkita
for lang in bn en hi or pa ta; do
nkita
python3 GPT_4/run_annotation.py --language ${lang} --val_csv ${lang}_val.csv --test_csv ${lang}_test.csv --output_csv ${lang}_annot.csv
nkita
done
nkita

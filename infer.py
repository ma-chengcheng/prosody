from process import load_chinese_dataset
from pytorch_transformers import BertTokenizer

chinese_data_set = load_chinese_dataset()
train_chinese_data_set = chinese_data_set['train']

sent, tag = train_chinese_data_set[0]
sent = ["[CLS]"] + sent + ["[SEP]"]
tag = ['<pad>'] + tag + ['<pad>']
print(sent)
print(tag)

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
tokens = tokenizer.tokenize(''.join(sent))
x = tokenizer.convert_tokens_to_ids(tokens)
print(x)

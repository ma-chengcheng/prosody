import re
import random
import torch
import random
import numpy as np
from torch.utils import data
from pytorch_transformers import BertTokenizer


class Dataset(data.Dataset):
    """据集类"""

    def __init__(self, tagged_sents, tag_to_index, config, word_to_embid=None):
        """
        初始化
        :param tagged_sents: 标记句子
        :return
        """
        sents, tags = [], []

        for sent, tag in tagged_sents:
            sents.append(["[CLS]"] + sent + ["[SEP]"])
            tags.append(["<pad>"] + tag + ["<pad>"])

        self.sents, self.tags = sents, tags

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

        self.tag_to_index = tag_to_index

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, id):
        """

        :param id:
        :return:
        """
        sent, tag = self.sents[id], self.tags[id]

        x, y = [], []
        for w in sent:
            token = self.tokenizer.tokenize(w)
            xx = self.tokenizer.convert_tokens_to_ids(token)
            x.extend(xx)

        y = [self.tag_to_index[t] for t in tag]

        # print(len(sent), len(tag))
        # print(sent, tag)
        # print(len(x), len(y))
        # print(x, y)

        assert len(x) == len(y)
        seqlen = len(y)

        return sent, tag, x, y, seqlen


def load_english_dataset():
    """
    加载英文数据集
    :return: 英文数据
    """
    splits = dict()
    splits['train'] = list()
    for split in ['train_360', 'train_100', 'dev', 'test']:
        filename = split
        tagged_sents = list()
        with open('./data/' + filename + '.txt') as f:
            lines = f.readlines()

            sent = []  # 句子
            tag_boundaries = []  #
            for i, line in enumerate(lines):
                split_line = line.split('\t')
                if split_line[0] != "<file>":
                    word = split_line[0]
                    tag_boundary = split_line[2]

                    sent.append(word)
                    tag_boundaries.append(tag_boundary)

                elif split_line[0] == "<file>" or i + 1 == len(lines):
                    assert len(sent) == len(tag_boundaries)
                    tagged_sents.append((sent, tag_boundaries))
                    sent = []
                    tag_boundaries = []

        #  句子随机打乱
        random.shuffle(tagged_sents)

        if split in ['train_360', 'train_100']:
            splits['train'].extend(tagged_sents)
        else:
            splits[split] = tagged_sents

    # 打印数据信息
    print('Training sentences: {}'.format(len(splits["train"])))
    print('Dev sentences: {}'.format(len(splits["dev"])))
    print('Test sentences: {}'.format(len(splits["test"])))

    return splits


def load_chinese_dataset():
    """
    加载中文数据集
    :return: 中文数据
    """
    tagged_sents = list()
    with open('./data/000001-010000.txt') as f:
        flag = True
        for line in f.readlines():
            if flag:
                tagged_sent = line.strip().split('\t')[-1]
                i = 0
                sent = list()
                tag = list()
                while i != len(tagged_sent):
                    c = tagged_sent[i]
                    if c == '#':
                        c += tagged_sent[i + 1]
                        if c == '#1' or c == '#2':
                            tag[-1] = '1'
                        else:
                            tag[-1] = '2'
                        i += 1
                    else:
                        sent.append(c)
                        tag.append('0')
                        if re.search('[^\u4e00-\u9fa5]', sent[-1]):
                            tag[-1] = 'NA'
                    i += 1
                assert len(sent) == len(tag)
                tagged_sents.append((sent, tag))

                # try:
                #     tagged_sents.append((' '.join(sent), ' '.join(tag)))
                # except TypeError:
                    # print('sent: {} tag: {}'.format(sent, tag))
            flag = not flag
    random.shuffle(tagged_sents)
    splits = dict()
    splits['train'] = tagged_sents[:8000]
    splits['dev'] = tagged_sents[8000:9000]
    splits['test'] = tagged_sents[9000:]

    # 打印数据信息
    print('Training sentences: {}'.format(len(splits["train"])))
    print('Dev sentences: {}'.format(len(splits["dev"])))
    print('Test sentences: {}'.format(len(splits["test"])))

    tag_to_index = {'0': 0, '1': 1, '2': 2, 'NA': 3, '<pad>': 3}
    index_to_tag = {0: '0', 1: '1', 2: '2', 3: 'NA', 4: '<pad>'}

    return splits, tag_to_index, index_to_tag


def pad(batch):
    """
    :param batch:
    :return:
    """
    # Pad sentences to the longest sample
    f = lambda x: [sample[x] for sample in batch]
    sents = f(0)
    tags = f(1)
    seqlens = f(4)
    maxlen = np.array(seqlens).max()

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]  # 0: <pad>
    x = f(2, maxlen)
    y = f(3, maxlen)

    f = torch.LongTensor
    return sents, tags, f(x), f(y), seqlens


# dataset_english = load_english_dataset()
#
# for item in Dataset(dataset_english['train']):
#     print(item)
#     break

# dataset_chinese = load_chinese_dataset()
# for item in Dataset(dataset_chinese['train'], language='zh'):
#     print(item)
#     break

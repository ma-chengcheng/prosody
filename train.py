"""
main
~~~~~~~~~~~~~~~
用于模型的训练
"""
import os
import errno
import glob
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
import process
from process import Dataset, load_chinese_dataset
from my_model import Bert
from argparse import ArgumentParser

parser = ArgumentParser(description='Prosody prediction')

parser.add_argument('--datadir',
                    type=str,
                    default='./data')
parser.add_argument('--train_set',
                    type=str,
                    choices=['train_100',
                             'train_360'],
                    default='train_360')
parser.add_argument('--batch_size',
                    type=int,
                    default=1)
parser.add_argument('--epochs',
                    type=int,
                    default=2)
parser.add_argument('--model',
                    type=str,
                    choices=['BertUncased',
                             'BertCased',
                             'BertLSTM',
                             'LSTM',
                             'BiLSTM',
                             'BertRegression',
                             'LSTMRegression',
                             'WordMajority',
                             'ClassEncodings',
                             'BertAllLayers'],
                    default='BertUncased')
parser.add_argument('--nclasses',
                    type=int,
                    default=3)
parser.add_argument('--hidden_dim',
                    type=int,
                    default=600)
parser.add_argument('--embedding_file',
                    type=str,
                    default='embeddings/glove.840B.300d.txt')
parser.add_argument('--layers',
                    type=int,
                    default=1)
parser.add_argument('--save_path',
                    type=str,
                    default='results.txt')
parser.add_argument('--log_every',
                    type=int,
                    default=10)
parser.add_argument('--learning_rate',
                    type=float,
                    default=0.00005)
parser.add_argument('--weight_decay',
                    type=float,
                    default=0)
parser.add_argument('--gpu',
                    type=int,
                    default=None)
parser.add_argument('--fraction_of_train_data',
                    type=float,
                    default=1
                    )
parser.add_argument("--optimizer",
                    type=str,
                    choices=['rprop',
                             'adadelta',
                             'adagrad',
                             'rmsprop',
                             'adamax',
                             'asgd',
                             'adam',
                             'sgd'],
                    default='adam')
parser.add_argument('--include_punctuation',
                    action='store_false',
                    dest='ignore_punctuation')
parser.add_argument('--sorted_batches',
                    action='store_true',
                    dest='sorted_batches')
parser.add_argument('--mask_invalid_grads',
                    action='store_true',
                    dest='mask_invalid_grads')
parser.add_argument('--invalid_set_to',
                    type=float,
                    default=-2) # -2 = log(0.01)
parser.add_argument('--log_values',
                    action='store_true',
                    dest='log_values')
parser.add_argument('--weighted_mse',
                    action='store_true',
                    dest='weighted_mse')
parser.add_argument('--shuffle_sentences',
                    action='store_true',
                    dest='shuffle_sentences')
parser.add_argument('--seed',
                    type=int,
                    default=1234)


def make_dirs(name):
    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise


def main():

    config = parser.parse_args()

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    # 是否使用gpu 
    if config.gpu is not None:
        torch.cuda.set_device(config.gpu)
        device = torch.device('cuda:{}'.format(config.gpu))
        torch.cuda.manual_seed(config.seed)
        print("\nTraining on GPU[{}] (torch.device({})).".format(config.gpu, device))
    else:
        device = torch.device('cpu')
        print("GPU not available so training on CPU (torch.device({})).".format(device))
        device = 'cpu'

    #   优化器
    optim_algorithm = optim.Adam

    splits, tag_to_index, index_to_tag = load_chinese_dataset()
    # splits, tag_to_index, index_to_tag, vocab = prosody_dataset.load_dataset(config)

    # 选择要加载的模型
    model = Bert(device, config, labels=len(tag_to_index))

    model.to(device)

    # 词嵌入
    word_to_embid = None

    # 获取不同类型的数据集合 
    train_dataset = Dataset(splits["train"], tag_to_index, config, word_to_embid)
    eval_dataset = Dataset(splits["dev"], tag_to_index, config, word_to_embid)
    test_dataset = Dataset(splits["test"], tag_to_index, config, word_to_embid)

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=not(config.sorted_batches),  # will manually shuffle if sorted_batches desired
                                 num_workers=1,
                                 collate_fn=process.pad)
    dev_iter = data.DataLoader(dataset=eval_dataset,
                               batch_size=config.batch_size,
                               shuffle=False,
                               num_workers=1,
                               collate_fn=process.pad)
    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                num_workers=1,
                                collate_fn=process.pad)

    optimizer = optim_algorithm(model.parameters(),
                                lr=config.learning_rate,
                                weight_decay=config.weight_decay)

    # 损失函数 
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 输出模型参数数量 
    params = sum([p.numel() for p in model.parameters()])
    print('Parameters: {}'.format(params))

    print('\nTraining started...\n')
    best_dev_acc = 0
    best_dev_epoch = 0

    for epoch in range(config.epochs):
        print("Epoch: {}".format(epoch+1))
        train(model, train_iter, optimizer, criterion, device, config)
        valid(model, dev_iter, criterion, index_to_tag, device, config, best_dev_acc, best_dev_epoch, epoch+1)
    # test(model, test_iter, criterion, index_to_tag, device, config)


def train(model, iterator, optimizer, criterion, device, config):
    """
    训练模型
    :param model: 模型
    :param iterator: 数据
    :param optimizer: 优化器
    :param criterion: 损失函数
    :param device: 设备
    :param config: 配置
    :return:
    """
    model.train()
    for i, batch in enumerate(iterator):
        sent, tag, x, y, seqlen = batch
        # words, x, is_main_piece, tags, y, seqlens, _, _ = batch

        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)

        logits, _ = model(x)  # logits: (N, T, VOCAB), y: (N, T)

        logits = logits.view(-1, logits.shape[-1]) # (N*T, VOCAB)
        y = y.view(-1)  # (N*T,)
        loss = criterion(logits.to(device), y.to(device))

        loss.backward()
        optimizer.step()

        if i % config.log_every == 0 or i+1 == len(iterator):
            print("Training step: {}/{}, loss: {:<.4f}".format(i+1, len(iterator), loss.item()))


def valid(model, iterator, criterion, index_to_tag, device, config, best_dev_acc, best_dev_epoch, epoch):
    """
    验证模型
    :param model: 模型
    :param iterator: 数据
    :param criterion: 损失函数
    :param index_to_tag: 下标转标签
    :param device: 
    :param config: 配置
    :param best_dev_acc: 最佳准确度
    :param best_dev_epoch: 最佳的epoch
    :param epoch: 当前epoch
    :return:
    """

    model.eval()
    dev_losses = []
    # Words, Is_main_piece, Tags, Y, Y_hat = [], [], [], [], []
    Y, Y_hat = [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            sent, tag, x, y, seqlen = batch
            x = x.to(device)
            y = y.to(device)

            logits, y_hat = model(x)  # y_hat: (N, T)
            labels = y

            logits = logits.view(-1, logits.shape[-1])  # (N*T, VOCAB)
            labels = labels.view(-1)  # (N*T,)
            loss = criterion(logits.to(device), labels.to(device))

            dev_losses.append(loss.item())

            # Words.extend(words)
            # Is_main_piece.extend(is_main_piece)
            # Tags.extend(tags)
            print(y.cpu().numpy().tolist())
            print(y_hat.cpu().numpy().tolist())
            exit()
            Y.extend(y.cpu().numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    # true = []
    # predictions = []
    # for words, is_main_piece, tags, y_hat in zip(Words, Is_main_piece, Tags, Y_hat):
    #     y_hat = [hat for head, hat in zip(is_main_piece, y_hat) if head == 1]
    #     preds = [index_to_tag[hat] for hat in y_hat]
    #
    #     tagslice = tags.split()[1:-1]
    #     predsslice = preds[1:-1]
    #     assert len(preds) == len(words.split()) == len(tags.split())
    #
    #     for t, p in zip(tagslice, predsslice):
    #         if config.ignore_punctuation:
    #             if t != 'NA':
    #                 true.append(t)
    #                 predictions.append(p)
    #         else:
    #             true.append(t)
    #             predictions.append(p)
    #
    # # calc metric
    # y_true = np.array(true)
    # y_pred = np.array(predictions)
    acc = 100. * (np.array(Y_hat) == np.array(Y)).astype(np.int32).sum() / len(Y_hat)

    if acc > best_dev_acc:
        best_dev_acc = acc
        best_dev_epoch = epoch
        dev_snapshot_path = 'best_model_{}_devacc_{}_epoch_{}.pt'.format(config.model, round(best_dev_acc, 2), best_dev_epoch)

        # save model, delete previous snapshot
        torch.save(model, dev_snapshot_path)
        for f in glob.glob('best_model_*'):
            if f != dev_snapshot_path:
                os.remove(f)

    print('Validation accuracy: {:<5.2f}%, Validation loss: {:<.4f}\n'.format(round(acc, 2), np.mean(dev_losses)))


def test(model, iterator, criterion, index_to_tag, device, config):
    """
    测试模型
    :param model: 模型
    :param iterator: 数据
    :param criterion: 损失函数
    :param index_to_tag: 下标转标签
    :param device: gpu或cpu训练
    :param config: 配置
    :return:
    """
    print('Calculating test accuracy and printing predictions to file {}'.format(config.save_path))
    print("Output file structure: <word>\t <tag>\t <prediction>\n")

    model.eval()
    test_losses = []

    Words, Is_main_piece, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_main_piece, tags, y, seqlens, _, _ = batch
            x = x.to(device)
            y = y.to(device)

            logits, y_hat = model(x)  # y_hat: (N, T)
            labels = y

            if config.model == 'ClassEncodings':
                logits = logits.view(-1, logits.shape[-1])  # (N*T, VOCAB)
                labels = labels.view(-1, labels.shape[-1])  # also (N*T, VOCAB)
                loss = criterion(logits.to(device), labels.to(device))
            else:
                logits = logits.view(-1, logits.shape[-1])  # (N*T, VOCAB)
                labels = labels.view(-1)  # (N*T,)
                loss = criterion(logits, labels)

            test_losses.append(loss.item())

            Words.extend(words)
            Is_main_piece.extend(is_main_piece)
            Tags.extend(tags)
            Y.extend(y.cpu().numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    true = []
    predictions = []
    # gets results and save
    with open(config.save_path, 'w') as results:
        for words, is_main_piece, tags, y_hat in zip(Words, Is_main_piece, Tags, Y_hat):
            y_hat = [hat for head, hat in zip(is_main_piece, y_hat) if head == 1]
            preds = [index_to_tag[hat] for hat in y_hat]
            if config.model != 'LSTM' and config.model != 'BiLSTM':
                tagslice = tags.split()[1:-1]
                predsslice = preds[1:-1]
                wordslice = words.split()[1:-1]
                assert len(preds) == len(words.split()) == len(tags.split())
            else:
                tagslice = tags.split()
                predsslice = preds
                wordslice = words.split()
            for w, t, p in zip(wordslice, tagslice, predsslice):
                results.write("{}\t{}\t{}\n".format(w, t, p))
                if config.ignore_punctuation:
                    if t != 'NA':
                        true.append(t)
                        predictions.append(p)
                else:
                    true.append(t)
                    predictions.append(p)
            results.write("\n")

    # calc metric
    y_true = np.array(true)
    y_pred = np.array(predictions)

    acc = 100. * (y_true == y_pred).astype(np.int32).sum() / len(y_true)
    print('Test accuracy: {:<5.2f}%, Test loss: {:<.4f} after {} epochs.\n'.format(round(acc, 2), np.mean(test_losses),
                                                                                   config.epochs))

    final_snapshot_path = 'final_model_{}_testacc_{}_epoch_{}.pt'.format(config.model,
                                                                 round(acc, 2), config.epochs)
    torch.save(model, final_snapshot_path)


if __name__ == "__main__":
    main()

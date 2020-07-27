from torch import nn
import torch
import torch.optim as optim
from torch.utils import data
from pytorch_transformers import BertModel
from process import Dataset, load_english_dataset, load_chinese_dataset

batch_size = 1


class BertLSTM(nn.Module):
    def __init__(self, device='cpu', language='en', labels=None):
        super().__init__()

        if language == 'en':
            self.bert = BertModel.from_pretrained('bert-base-cased')
        else:
            self.bert = BertModel.from_pretrained('bert-base-chinese')

        self.fc = nn.Linear(1536, 4)
        self.device = device
        self.lstm = nn.LSTM(input_size=768,
                            hidden_size=768,
                            num_layers=1,
                            dropout=0,
                            bidirectional=True)

    def forward(self, x):

        # x = x.to(self.device)

        self.bert.eval()
        with torch.no_grad():
            print(self.bert(x))
            enc = self.bert(x)[0]

        enc = enc.permute(1, 0, 2).to(self.device)
        enc = self.lstm(enc)[0]
        enc = enc.permute(1, 0, 2)
        logits = self.fc(enc).to(self.device)
        y_hat = logits.argmax(-1)
        return logits, y_hat


# dataset_english = load_english_dataset()
#
# for item in Dataset(dataset_english['train']):
#     print(item)
#     break
# english_dataset = load_english_dataset()
# english_dataset_train = Dataset(english_dataset['train'])
# english_dataset_dev = Dataset(english_dataset['dev'])
# english_dataset_test = Dataset(english_dataset['test'])


chinese_dataset = load_chinese_dataset()
chinese_dataset_train = Dataset(chinese_dataset['train'], language='cn')
chinese_dataset_dev = Dataset(chinese_dataset['dev'], language='cn')
chinese_dataset_test = Dataset(chinese_dataset['test'], language='cn')


train_iter = data.DataLoader(dataset=chinese_dataset_train,
                             batch_size=batch_size,
                             num_workers=1)
dev_iter = data.DataLoader(dataset=chinese_dataset_train,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1)
test_iter = data.DataLoader(dataset=chinese_dataset_train,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=1)

model = BertLSTM(language='zh')
criterion = nn.CrossEntropyLoss()
optim_algorithm = optim.Adam
optimizer = optim_algorithm(model.parameters(),
                            lr=0.001,
                            weight_decay=0)


model.train()
for i, batch in enumerate(chinese_dataset_train):
    # print(batch)

    sent, tag, x, y, _ = batch

    x = torch.tensor(x)
    y = torch.tensor(y)
    print(x)
    print(y)

    optimizer.zero_grad()
    logits, y_hat = model(x)
    loss = criterion(logits, y)

    loss.backward()
    optimizer.step()

    print("Training step: {}/{}, loss: {:<.4f}".format(i + 1, len(iterator), loss.item()))

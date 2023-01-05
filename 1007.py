import math
import numpy as np
import pandas as pd
import os 
import csv
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

def same_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.banchmark = False
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_valid_split(data_set, valid_radio, seed):
    valid_set_size = int(valid_radio * len(data_set))
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)

def predict(test_loader, model, device):
    model.eval()
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds

class stockDataTest(Dataset):
    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]
    
    def __len__(self):
        return len(self.x)

class My_model(torch.nn.Module):
    def __init__(self, input_dim):
        super(My_model, self).__init__()
        self.layers = torch.nn.Sequential(
            nn.Linear(164, 164),
            nn.ReLU(),
            nn.Linear(164, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1)
        return x

def select_feat(train_data, valid_data, test_data, select_all=True):
    y_train, y_valid = train_data[0:,-1:], valid_data[0:,-1:]
    raw_x_train = train_data[0:,:164]
    raw_x_valid = valid_data[0:,:164]
    raw_x_test = test_data[0:,:164]

    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        feat_idx = [i for i in range(163)]
    
    return raw_x_train[:,feat_idx], raw_x_valid[:,feat_idx], raw_x_test[:,feat_idx], y_train, y_valid



def trainer(train_loader, valid_loader, model, config, device):
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, foreach=None, maximize=False, capturable=False)
    # optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)
    writer = SummaryWriter()
    if not os.path.isdir('./models'):
        os.mkdir('./models')

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0

    for epoch in range(n_epochs):
        model.train()
        loss_record = []
        train_pbar = tqdm(train_loader, position=0, leave = True)

        for x, y in train_pbar:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred,y)
            loss.backward()
            optimizer.step()
            step += 1
            loss_record.append(loss.detach().item())       
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss':loss.detach().item()})

 
        mean_train_loss = sum(loss_record)/len(loss_record)
        writer.add_scalar('Loss/Train', mean_train_loss, step)
        model.eval()

        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred,y)
            loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record)/len(loss_record)
        print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss: .4f}')
        writer.add_scalar('Loss/valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path'],)
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 5201314,
    'select_all': True,
    'valid_ratio': 0.7,
    'n_epochs': 10000000,
    'batch_size': 100,
    'learning_rate': (1e-5)/10,
    'early_stop': 10000,
    'save_path': './models/model.ckpt'
}

same_seed(config['seed'])
train_data, test_data = pd.read_csv('data_valid1007.csv', encoding="cp1252").values, pd.read_csv('data_test1007.csv', encoding="cp1252").values

train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])

print(f"""train_data size: {train_data.shape}
valid_data size: {valid_data.shape}
test_data size: {test_data.shape}""")

x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, test_data, config['select_all'])
x_train, x_valid, x_test, y_train, y_valid = x_train.astype(int), x_valid.astype(int), x_test.astype(int), y_train.astype(int), y_valid.astype(int)

print(f'number of features: {x_train.shape[1]}')

train_dataset = stockDataTest(x_train, y_train)
valid_dataset = stockDataTest(x_valid, y_valid)
test_dataset = stockDataTest(x_test)


train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)


def save_pred(preds, file):
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id','tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow(i,p)

print(x_train.shape[1])

model = My_model(input_dim=x_train.shape[1]).to(device)
preds = predict(test_loader, model, device)
model.load_state_dict(torch.load(config['save_path']))
trainer(train_loader, valid_loader, model, config, device)


save_pred(preds, 'pred.csv')
from sklearn import linear_model
from data_processor import DataProcessor
from models.lstm_classifier import LSTMClassifier
from torch.optim import Adam
import time
import joblib

import torch.nn as nn
import torch
import numpy as np


# path config
model_path = './ckpt'

# data config
tiker = 'AAPL'
is_local = False
period = '10y'
interval = '1d'
features = ['Open', 'Close', 'High', 'Low', 'Volume']
technical_indicators = ['CROCP', 'OROCP', 'LROCP', 'HROCP', 'VROCP', 'MA', 'SMA', 'EMA', 'WMA', 'RSI', 'RSIROCP']
target = 'Trend'
seq_len = 10
train_ratio = 0.7
test_ratio = 0.2
scale = True

# traditional model config
C = 1.0
penalty = 'l2'
tol = 1e-4

# deep model config
batch_size = 32
input_size = len(features) + len(technical_indicators)
hidden_size = 128
num_layers = 2
num_classes = 1
lr = 1e-4
n_epochs = 1000
model_name = 'LSTMClassifier'
model_framework = 'PyTorch'

# initialize
model = None
optimizer = None
loss_fn = nn.BCELoss()
train_loader, val_loader, test_loader = None, None, None
X_train, X_val, X_test, y_train, y_val, y_test = None, None, None, None, None, None
data_processor = None


def build_data():
    global data_processor
    data_processor = DataProcessor(tiker, is_local, period, interval, 
                                   features, technical_indicators, 
                                   target, batch_size, seq_len, 
                                   train_ratio, test_ratio, scale)
    if model_framework == 'PyTorch':
        global train_loader, val_loader, test_loader
        train_loader, val_loader, test_loader = data_processor.get_data_loader()
    elif model_framework == 'Sklearn':
        global X_train, X_val, X_test, y_train, y_val, y_test
        X_train, X_val, X_test, y_train, y_val, y_test = data_processor.get_data()


def build_model(path=None):
    global model
    if model_framework == 'Sklearn':
        if path:
            model = joblib.load(path)
        elif model_name == 'LogisticRegression':
            model = linear_model.LogisticRegression(C=C, penalty=penalty, tol=tol,
                                                    solver='lbfgs', max_iter=1000, 
                                                    verbose=3, random_state=42)
    elif model_framework == 'PyTorch':
        if model_name == 'LSTMClassifier':
            global optimizer
            model = LSTMClassifier(input_size=input_size, 
                                hidden_size=hidden_size, 
                                num_layers=num_layers, 
                                num_classes=num_classes)
            optimizer = Adam(model.parameters(), lr=lr)

        model.load_state_dict(torch.load(path))


def evaluate(data_loader):
    model.eval()
    y_pred, acc = [], []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            y_pred_ = model(X_batch.float()).numpy()
            acc.extend((y_pred_ > 0.5) == y_batch.numpy().reshape(-1, 1))
            y_pred.extend(y_pred_)
        acc = np.mean(acc)
    return y_pred, acc


def train_deep_model():
    print(f'Training {model_name} model...')
    for epoch in range(1, n_epochs + 1):
        # train
        model.train()
        y_pred_train, acc = [], []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch.float())
            loss = loss_fn(y_pred, y_batch.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            y_pred_train.extend(y_pred.detach().numpy())
            acc.extend((y_pred.detach().numpy() > 0.5) == y_batch.numpy().reshape(-1, 1))
        if epoch % 100 == 99:
            # validate
            train_acc = np.mean(acc)
            _, val_acc = evaluate(val_loader)
            print(f'Epoch: {epoch+1}, Train Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
    torch.save(model.state_dict(), f'{model_path}/{model_name}_{time.strftime("%Y%m%d%H%M%S", time.localtime())}.ckpt')
    # test
    y_pred_test, test_acc = evaluate(test_loader)
    print(f'Test Acc: {test_acc:.4f}')
    return y_pred_test


def train_traditional_model():
    # train
    print(f'Training {model_name} model...')
    print(X_train.shape, y_train.shape)
    model.fit(X_train, y_train)
    # validate
    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val)
    print(f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
    # save
    joblib.dump(model, f'{model_path}/{model_name}_{time.strftime("%Y%m%d%H%M%S", time.localtime())}.pkl')
    # test
    test_acc = model.score(X_test, y_test)
    print(f'Test Acc: {test_acc:.4f}')
    

def train():
    if model_framework == 'PyTorch':
        y_pred_test = train_deep_model()
        return y_pred_test
    if model_framework == 'Sklearn':
        train_traditional_model()
        return model.predict(X_test)


def print_info():
    print(f'Tiker: {tiker}')
    print(f'Period: {period}')
    print(f'Interval: {interval}')
    print(f'Features: {features}')
    print(f'Technical Indicators: {technical_indicators}')
    print(f'Target: {target}')
    print(f'Sequence Length: {seq_len}')
    print(f'Train Ratio: {train_ratio}')
    print(f'Test Ratio: {test_ratio}')
    print(f'Batch Size: {batch_size}')
    print(f'Input Size: {input_size}')
    print(f'Hidden Size: {hidden_size}')
    print(f'Number of Layers: {num_layers}')
    print(f'Number of Classes: {num_classes}')
    print(f'Learning Rate: {lr}')
    print(f'Number of Epochs: {n_epochs}')
    print(f'Model Name: {model_name}')
    print(f'Model Framework: {model_framework}')
    

def run():
    print_info()
    build_data()
    build_model()
    return train()
    
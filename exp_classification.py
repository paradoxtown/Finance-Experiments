from sklearn import linear_model, svm, ensemble, neural_network
from sklearn.metrics import roc_auc_score
from data_processor import DataProcessor
from models.lstm_classifier import LSTMClassifier
from models.att_lstm_classifier import AttLSTMClassifier
from torch.optim import Adam
import time
import joblib

import torch.nn as nn
import torch
import numpy as np

import simulator as sim

# path config
model_path = './ckpt'

# data config
tiker = 'AAPL'
is_local = False
period = None
interval = '1d'
features = ['Open', 'Close', 'High', 'Low', 'Volume']
technical_indicators = ['CROCP', 'OROCP', 'LROCP', 'HROCP', 'VROCP', 'MA', 'SMA', 'EMA', 'WMA', 'RSI', 'RSIROCP']
target = 'Trend'
seq_len = 1
train_ratio = 0.7
test_ratio = 0.2
scale = True

# traditional model config
C = 10.0
penalty = 'l2'
tol = 1e-6
max_iter = 1000

# deep model config
batch_size = 512
input_size = 5
hidden_size = 128
num_layers = 1
num_classes = 1
lr = 1e-3
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
    data_processor = None
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
    model = None
    input_size = len(features) + len(technical_indicators)
    if model_framework == 'Sklearn':
        if path:
            model = joblib.load(path)
        elif model_name == 'LogisticRegression':
            model = linear_model.LogisticRegression(C=C, penalty=penalty, tol=tol,
                                                    solver='lbfgs', max_iter=max_iter, 
                                                    random_state=42)
        elif model_name == 'Lasso':
            model = linear_model.Lasso(alpha=0.1, max_iter=max_iter, tol=tol, random_state=42)
        elif model_name == 'SVM':
            model = svm.SVC(C=C, kernel='rbf', gamma='auto', tol=tol, 
                            max_iter=max_iter, random_state=42)
        elif model_name == 'RandomForest':
            model = ensemble.RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42)
        elif model_name == 'MLP':
            model = neural_network.MLPClassifier(hidden_layer_sizes=(100, 100), verbose=True, 
                                                 batch_size=32, early_stopping=True, tol=tol,
                                                 learning_rate_init=lr, max_iter=max_iter, random_state=42)
        else:
            raise ValueError(f'Invalid model name: {model_name} with Sklearn framework')
    elif model_framework == 'PyTorch':
        if model_name == 'LSTMClassifier':
            model = LSTMClassifier(input_size=input_size, 
                                   hidden_size=hidden_size, 
                                   num_layers=num_layers, 
                                   num_classes=num_classes)

        elif model_name == 'AttLSTMClassifier':
            model = AttLSTMClassifier(input_size=input_size,
                                      hidden_size=hidden_size,
                                      num_layers=num_layers,
                                      num_classes=num_classes)
        else:
            raise ValueError(f'Invalid model name: {model_name} with PyTorch framework')
        if path:
            model.load_state_dict(torch.load(path))


def evaluate(data_loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            y_true.extend(y_batch.numpy())
            y_pred_ = model(X_batch.float()).numpy()
            y_pred.extend(y_pred_ > 0.5)
    y_pred = np.squeeze(y_pred)
    acc = np.mean(y_pred == y_true)
    auc = roc_auc_score(y_true, y_pred)
    return y_pred, acc, auc


def train_deep_model():
    global optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 betas=(0.9, 0.999), eps=1e-08,
                                 weight_decay=1e-5)
    
    print(f'Training {model_name} model...')
    print(model)
    print('----------------------------------------')
    for epoch in range(1, n_epochs + 1):
        # train
        model.train()
        acc = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch.float())
            loss = loss_fn(y_pred, y_batch.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            acc.extend((y_pred.detach().numpy() > 0.5) == y_batch.numpy().reshape(-1, 1))
        if epoch % 100 == 99:
            # validate
            train_acc = np.mean(acc)
            _, val_acc, val_auc = evaluate(val_loader)
            print(f'Epoch: {epoch+1}, Train Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}, \
            Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}, lr: {optimizer.param_groups[0]["lr"]:.6f}')
    torch.save(model.state_dict(), f'{model_path}/{model_name}_{time.strftime("%Y%m%d%H%M%S", time.localtime())}.ckpt')
    # test
    y_pred_test, test_acc, test_auc = evaluate(test_loader)
    print(f'Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}')
    
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
    y_pred = model.predict(X_test)
    test_acc = model.score(X_test, y_test)
    test_auc = roc_auc_score(y_test, y_pred)
    print(f'Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}')
    
    return y_pred
    

def train():
    if model_framework == 'PyTorch':
        return train_deep_model()
    if model_framework == 'Sklearn':
        return train_traditional_model()


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
    print(f'C: {C}')
    print(f'Batch Size: {batch_size}')
    print(f'Input Size: {input_size}')
    print(f'Hidden Size: {hidden_size}')
    print(f'Number of Layers: {num_layers}')
    print(f'Number of Classes: {num_classes}')
    print(f'Learning Rate: {lr}')
    print(f'Number of Epochs: {n_epochs}')
    print(f'Max Iterations: {max_iter}')
    print(f'Model Name: {model_name}')
    print(f'Model Framework: {model_framework}')
    print('----------------------------------------')
    

def run():
    build_data()
    build_model()
    print_info()
    y_pred = train()
    
    sim_data = data_processor.get_simulate_data()
    sim_data['y_pred'] = y_pred
    bt = sim.backtest(sim_data)
    sim_result = bt.run()
    print('----------------------------------------')
    print(sim_result)
    return y_pred

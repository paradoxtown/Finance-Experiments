import copy
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
os.environ["WANDB_NOTEBOOK_NAME"] = "experiments.ipynb"

from sklearn import linear_model, svm, ensemble, neural_network
from sklearn.metrics import roc_auc_score
from data_processor import DataProcessor
from models.gru_classifier import GRUClassifier
from models.lstm_classifier import LSTMClassifier
from models.att_lstm_classifier import AttLSTMClassifier
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam, SGD
import time
import joblib

import torch.nn as nn
import torch
import numpy as np

import wandb
import simulator as sim

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_wandb = True

# set random seed
SEED=42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False

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
early_stopping = True

# deep model config
batch_size = 512
input_size = 5
hidden_size = 128
num_layers = 2
num_classes = 1
lr = 1e-3
n_epochs = 1000
model_name = 'LSTMClassifier'
model_framework = 'PyTorch'
best = False
optimizer_name = 'Adam'

# initialize
model = None
optimizer = None
lr_scheduler = None
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
    global model, input_size
    model = None
    input_size = len(features) + len(technical_indicators)
    if model_framework == 'Sklearn':
        if path:
            model = joblib.load(path)
        elif model_name == 'LogisticRegression':
            model = linear_model.LogisticRegression(C=C, penalty=penalty, tol=tol, multi_class='ovr',
                                                    solver='lbfgs', max_iter=max_iter, random_state=42)
        elif model_name == 'Lasso':
            model = linear_model.Lasso(alpha=0.1, max_iter=max_iter, tol=tol, random_state=42)
        elif model_name == 'SVM':
            model = svm.LinearSVC(C=C, max_iter=max_iter, random_state=42)
        elif model_name == 'RandomForest':
            model = ensemble.RandomForestClassifier(max_depth=3, n_estimators=5, min_samples_leaf=20, random_state=42)
        elif model_name == 'MLP':
            model = neural_network.MLPClassifier(hidden_layer_sizes=(32, 64, 32), verbose=False, tol=tol, 
                                                 batch_size=batch_size, early_stopping=early_stopping, 
                                                 solver='adam', shuffle=False, learning_rate_init=lr, 
                                                 alpha=0.001, max_iter=max_iter, random_state=42)
        else:
            raise ValueError(f'Invalid model name: {model_name} with Sklearn framework')
    elif model_framework == 'PyTorch':
        if model_name == 'LSTMClassifier':
            model = LSTMClassifier(input_size=input_size, 
                                   hidden_size=hidden_size, 
                                   num_layers=num_layers, 
                                   num_classes=num_classes)
        elif model_name == 'GRUClassifier':
            model = GRUClassifier(input_size=input_size,
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
    global model
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_true.extend(y_batch.cpu().data.numpy())
            y_pred_ = model(X_batch.float()).cpu().data.numpy()
            y_pred.extend(y_pred_ > 0.5)
    y_pred = np.squeeze(y_pred)
    acc = np.mean(y_pred == y_true)
    auc = roc_auc_score(y_true, y_pred)
    return y_pred, acc, auc


def train_deep_model():
    global optimizer, lr_scheduler, model
    if optimizer_name == 'Adam':
        optimizer = Adam(model.parameters(), lr=lr,
                        betas=(0.9, 0.999), eps=1e-08,
                        weight_decay=1e-5)
    elif optimizer_name == 'SGD':
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
    lr_scheduler = StepLR(optimizer, step_size=(n_epochs // 3), gamma=0.1)
    
    print(f'Training {model_name} model...')
    print(model)
    print('----------------------------------------')
    model.to(device)
    best_model = None
    best_acc = 0
    for epoch in range(1, n_epochs + 1):
        # train
        model.train()
        acc = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.float().to(device), y_batch.float().to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
            acc.extend((y_pred.cpu().data.numpy() > 0.5) == y_batch.cpu().data.numpy().reshape(-1, 1))
        if epoch % 100 == 99:
            # validate
            train_acc = np.mean(acc)
            _, val_acc, val_auc = evaluate(val_loader)
            if val_acc > best_acc:
                best_acc = val_acc
                best_model = copy.deepcopy(model)
            print(f'Epoch: {epoch+1}, Train Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}, \
            Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}, lr: {optimizer.param_groups[0]["lr"]:.6f}')
        lr_scheduler.step()
    if best: model = best_model
    torch.save(model.state_dict(), f'{model_path}/{model_name}_{time.strftime("%Y%m%d%H%M%S", time.localtime())}.ckpt')
    # test
    y_pred_test, test_acc, test_auc = evaluate(test_loader)
    print(f'Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}')
    if use_wandb: wandb.log({'test_acc': test_acc, 'test_auc': test_auc})
    
    return y_pred_test, test_acc, test_auc


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
    if use_wandb: wandb.log({"test_acc": test_acc, "test_auc": test_auc})
    
    return y_pred, test_acc, test_auc
    

def train():
    if model_framework == 'PyTorch':
        return train_deep_model()
    if model_framework == 'Sklearn':
        return train_traditional_model()


def print_info():
    if use_wandb:
        if model_framework == 'PyTorch':
            wandb.init(
                project = "finance-experiment-5y",
                config = {
                    "learning_rate": lr,
                    "batch_size": batch_size,
                    "n_epochs": n_epochs,
                    "tech_indicators": technical_indicators,
                    "model_name": model_name,
                    "model_framework": model_framework,
                    "seq_len": seq_len,
                    "input_size": input_size,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "num_classes": num_classes,
                    "optimizer": optimizer.__class__.__name__,
                    "lr_scheduler": lr_scheduler.__class__.__name__,
                    "loss_fn": loss_fn.__class__.__name__,
                }
            )
        elif model_framework == 'Sklearn':
            wandb.init(
                project = "finance-experiment-5y",
                config = {
                    "model_name": model_name,
                    "model_framework": model_framework,
                    "C": C,
                    "penalty": penalty,
                    "max_iter": max_iter,
                    "tech_indicators": technical_indicators
                }
            )


def run_baseline():
    build_data()
    sim_data = data_processor.get_simulate_data()
    # random
    y_pred = np.random.randint(0, 2, len(sim_data))
    sim_data['y_pred'] = y_pred
    bt = sim.backtest(sim_data)
    sim_result = bt.run()
    wandb.init(
        project = "finance-experiment-5y",
        config = {
            "model_name": 'baseline-random',
            "model_framework": 'baseline'
        }
    )
    wandb.log({"Sharpe Ratio": sim_result['Sharpe Ratio'],
               "Sortino Ratio": sim_result['Sortino Ratio'],
               "Calmar Ratio": sim_result['Calmar Ratio'],
               "Return (Ann.) [%]": sim_result['Return (Ann.) [%]']})
    wandb.finish()
    
    # all one
    sim_data.drop('y_pred', axis=1, inplace=True)
    sim_data['y_pred'] = [1] * len(sim_data)
    bt = sim.backtest(sim_data)
    sim_result = bt.run()
    wandb.init(
        project = "finance-experiment-5y",
        config = {
            "model_name": 'baseline-all-positive',
            "model_framework": 'baseline'
        }
    )
    wandb.log({"Sharpe Ratio": sim_result['Sharpe Ratio'],
               "Sortino Ratio": sim_result['Sortino Ratio'],
               "Calmar Ratio": sim_result['Calmar Ratio'],
               "Return (Ann.) [%]": sim_result['Return (Ann.) [%]']})
    wandb.finish()    
    
    # y_true
    sim_data.drop('y_pred', axis=1, inplace=True)
    sim_data['y_pred'] = sim_data['y_true']
    bt = sim.backtest(sim_data)
    sim_result = bt.run()
    wandb.init(
        project = "finance-experiment-5y",
        config = {
            "model_name": 'baseline-true',
            "model_framework": 'baseline'
        }
    )
    wandb.log({"Sharpe Ratio": sim_result['Sharpe Ratio'],
               "Sortino Ratio": sim_result['Sortino Ratio'],
               "Calmar Ratio": sim_result['Calmar Ratio'],
               "Return (Ann.) [%]": sim_result['Return (Ann.) [%]']})
    wandb.finish()
    

def run():
    global data_processor
    build_data()
    build_model()
    print_info()
    y_pred, acc, auc = train()
    
    sim_data = data_processor.get_simulate_data()
    sim_data['y_pred'] = y_pred
    bt = sim.backtest(sim_data)
    sim_result = bt.run()
    bt.plot(filename=f'./simulation/{model_name}_{time.strftime("%Y%m%d%H%M%S", time.localtime())}.html')
    if use_wandb:
        wandb.log({"Sharpe Ratio": sim_result['Sharpe Ratio'],
                "Sortino Ratio": sim_result['Sortino Ratio'],
                "Calmar Ratio": sim_result['Calmar Ratio'],
                "Return (Ann.) [%]": sim_result['Return (Ann.) [%]']})
        wandb.finish()
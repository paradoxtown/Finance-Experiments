import yfinance as yf
import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader

from talib import abstract
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataProcessor(object):
    def __init__(self, tiker, is_local=False, period='10y', interval='1d', 
                 features=['Open', 'Close', 'High', 'Close', 'Volume'], 
                 technical_indicators=['CROCP', 'OROCP', 'LROCP', 'HROCP', 'VROCP', 'MA', 'RSI', 'RSIROCP'],
                 target='Trend', batch_size=32, seq_len=10, train_ratio=0.7,
                 test_ratio=0.2, scale=True):
        self.tiker = tiker
        if is_local:
            self.hist = pd.read_csv(f'./datasets/{tiker}.csv')
        else:
            if period:
                self.hist = yf.Ticker(tiker).history(period=period, interval=interval)
            else:
                self.hist = yf.Ticker(tiker).history(start='2013-07-01', end='2023-07-01', interval=interval)
        
        self.hist = self.hist[features]
        
        if technical_indicators:
            self.get_technical_indicators(technical_indicators)
        
        self.features = features + technical_indicators
        
        if target == 'Trend':
            self.get_trend()
        elif target == 'Return':
            self.get_return()
            
        self.test_ratio = test_ratio
        self.val_ratio = 1 - train_ratio - test_ratio
        
        self.batch_szie = batch_size
        self.seq_len = seq_len
        
        self.scale = scale
    
    def get_technical_indicators(self, technical_indicators, timeperiod=14):
        for indicator in technical_indicators:
            if indicator == 'CROCP':
                self.hist['CROCP'] = abstract.ROCP(self.hist['Close'], timeperiod=timeperiod)
            if indicator == 'OROCP':
                self.hist['OROCP'] = abstract.ROCP(self.hist['Open'], timeperiod=timeperiod)
            if indicator == 'LROCP':
                self.hist['LROCP'] = abstract.ROCP(self.hist['Low'], timeperiod=timeperiod)
            if indicator == 'HROCP':
                self.hist['HROCP'] = abstract.ROCP(self.hist['High'], timeperiod=timeperiod)
            if indicator == 'VROCP':
                self.hist['VROCP'] = abstract.ROCP(self.hist['Volume'], timeperiod=timeperiod)
            if indicator == 'MA':
                self.hist['MA'] = abstract.MA(self.hist['Close'], timeperiod=timeperiod)
            if indicator == 'SMA':
                self.hist['SMA'] = abstract.SMA(self.hist['Close'], timeperiod=timeperiod)
            if indicator == 'EMA':
                self.hist['EMA'] = abstract.EMA(self.hist['Close'], timeperiod=timeperiod)
            if indicator == 'WMA':
                self.hist['WMA'] = abstract.WMA(self.hist['Close'], timeperiod=timeperiod)
            if indicator == 'RSI':
                self.hist['RSI'] = abstract.RSI(self.hist['Close'], timeperiod=timeperiod)
            if indicator == 'RSIROCP':
                self.hist['RSIROCP'] = abstract.ROCP(self.hist['RSI'], timeperiod=timeperiod)
        self.hist.dropna(inplace=True)
    
    def get_trend(self):
        self.hist['Target'] = self.hist['Close'].diff()
        self.hist['Target'] = self.hist['Target'].apply(lambda x: 1 if x > 0 else 0)
        self.hist['Target'] = self.hist['Target'].shift(-1, fill_value=1)
    
    def get_return(self):
        # todo
        self.hist['Target'] = abstract.ROCP(self.hist['Close'])
    
    def get_data(self):
        X_train, X_test, y_train, y_test = \
            train_test_split(self.hist[self.features], self.hist['Target'], test_size=self.test_ratio, shuffle=False, random_state=42)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=self.val_ratio, shuffle=False, random_state=42)
        
        if self.scale:
            scaler = StandardScaler()
            scaler.fit(X_train)
            
            X_train = scaler.transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
        
        if self.seq_len > 1:
            X_train, y_train = self.get_time_series(X_train, y_train)
            X_val, y_val = self.get_time_series(X_val, y_val)
            X_test, y_test = self.get_time_series(X_test, y_test)
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_time_series(self, X, y):
        Xs, ys = [], []
        for i in range(len(X) - self.seq_len + 1):
            v = X[i:(i + self.seq_len)]
            Xs.append(v)
            ys.append(y[i + self.seq_len - 1])
        return np.array(Xs), np.array(ys)

    def get_data_loader(self):
        X_train, X_val, X_test, y_train, y_val, y_test = self.get_data()
        train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()), 
                                  batch_size=self.batch_szie, shuffle=False)
        val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()), 
                                batch_size=self.batch_szie, shuffle=False)
        test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()), 
                                 batch_size=self.batch_szie, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def get_simulate_data(self):
        _, sim_data, _, y_true = train_test_split(self.hist[self.features], self.hist['Target'], test_size=self.test_ratio, shuffle=False)
        sim_data['y_true'] = y_true
        return sim_data[self.seq_len-1:]
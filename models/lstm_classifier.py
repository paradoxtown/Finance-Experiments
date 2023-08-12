from torch import nn
import torch


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = 0 if num_layers == 1 else 0.5
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=self.dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        nn.init.xavier_uniform_(h0)
        nn.init.xavier_uniform_(c0)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take the last time step output
        out = self.sigmoid(out)
        return out


if __name__ == '__main__':
    import sys
    sys.path.append('..')
    from data_processor import DataProcessor
    from sklearn.metrics import roc_auc_score
    import numpy as np
    import time
    
    SEED=42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.use_deterministic_algorithms(True)
    
    tiker = 'AAPL'
    is_local = False
    period = None
    interval = '1d'
    features = ['Open', 'Close', 'High', 'Low', 'Volume']
    technical_indicators = ['CROCP', 'OROCP', 'LROCP', 'HROCP', 'VROCP', 'MA', 'SMA', 'EMA', 'WMA', 'RSI', 'RSIROCP']
    target = 'Trend'
    seq_len = 10
    train_ratio = 0.7
    test_ratio = 0.2
    scale = True
    
    batch_size = 512
    input_size = 16
    hidden_size = 64
    num_layers = 2
    num_classes = 1
    lr = 1e-3
    n_epochs = 1000
    model_path = '../ckpt'
    model_name = 'LSTMClassifier'
    
    data_processor = DataProcessor(tiker, is_local, period, interval, 
                                features, technical_indicators, 
                                target, batch_size, seq_len, 
                                train_ratio, test_ratio, scale)
    train_loader, val_loader, test_loader = data_processor.get_data_loader()
    model = LSTMClassifier(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            num_classes=num_classes)

    def evaluate(data_loader):
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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                betas=(0.9, 0.999), eps=1e-08,
                                weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=(n_epochs // 3), gamma=0.1)
    loss_fn = nn.BCELoss()
    
    print(model)
    print('----------------------------------------')
    model.to(device)
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
            print(f'Epoch: {epoch+1}, Train Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}, \
            Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}, lr: {optimizer.param_groups[0]["lr"]:.6f}')
        lr_scheduler.step()
    torch.save(model.state_dict(), f'{model_path}/{model_name}_{time.strftime("%Y%m%d%H%M%S", time.localtime())}.ckpt')
    # test
    y_pred_test, test_acc, test_auc = evaluate(test_loader)
    print(f'Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}')
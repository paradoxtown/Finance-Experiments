from torch.nn import functional as F
from torch import nn
import torch


class AttLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(AttLSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.Tanh = nn.Tanh()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.batch_norm = nn.BatchNorm1d(num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        nn.init.xavier_uniform_(h0)
        nn.init.xavier_uniform_(c0)
        
        att = F.scaled_dot_product_attention(x, x, x, dropout_p=0.2)
        out = self.Tanh(att)
        out, _ = self.lstm(out, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.batch_norm(out)
        out = self.sigmoid(out)
        return out

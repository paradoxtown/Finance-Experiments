from backtesting import Strategy, Backtest
import numpy as np


class MyStrategy(Strategy):
    price_delta = .004
    
    def init(self):
        self.I(lambda x: x, self.data['y_true'], name='y_true')
        self.I(lambda x: x, self.data['y_pred'], name='y_pred')

    def next(self):
        forcast = self.data['y_pred'][-1]
        
        # upper, lower = close * (1 + np.r_[1, -1] * self.price_delta)
        
        if forcast == 1 and not self.position.is_long:
            self.buy(size=.2)
        elif forcast == 0 and not self.position.is_short:
            self.sell(size=.2)
            


def backtest(data):
    return Backtest(data, MyStrategy, cash=10000, commission=.002)
from backtesting import Strategy, Backtest


class MyStrategy(Strategy):
    def init(self):
        self.I(lambda x: x, self.data['y_true'], name='y_true')
        self.I(lambda x: x, self.data['y_pred'], name='y_pred')

    def next(self):
        forecast = self.data['y_pred'][-1]
        
        if forecast == 1 and not self.position.is_long:
            self.buy(size=.2)
        elif forecast == 0 and not self.position.is_short:
            self.sell(size=.2)

class LongOnly(Strategy):
    def init(self):
        self.I(lambda x: x, self.data['y_true'], name='y_true')
        self.I(lambda x: x, self.data['y_pred'], name='y_pred')
    
    def next(self):
        forecast = self.data['y_pred'][-1]
        
        if forecast == 1:
            self.buy(size=.2)


def backtest(data):
    return Backtest(data, MyStrategy, cash=10000, commission=.002)
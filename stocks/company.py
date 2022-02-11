import sys
import os

import yfinance as yf

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from stock_prediction import StockIndicatorsBase
from plots import TechnicalIndicatorsChartPlotter


class Company(StockIndicatorsBase, TechnicalIndicatorsChartPlotter):
    def __init__(self, symbol, start_date=None, end_date=None, period=None):
        self.symbol = symbol
        if (period):
            self.stock_data = yf.Ticker(symbol).history(period=period)
            self.period = period
            self.start_date = (str(self.stock_data.index[0])).split(' ')[0]
            self.end_date = (str(self.stock_data.index[-1])).split(' ')[0]
        else:
            self.start_date = start_date
            self.end_date = end_date
            self.stock_data = yf.Ticker(symbol).history(start=start_date, end=end_date)

        self.technical_indicators = None
        self.prices = self.stock_data['Close']

    def set_technical_indicators(self):
        self._set_technical_indicators()

    def draw_plots(self):
        self.plot_macd()
        self.plot_rsi()
        self.plot_bollinger_bands()


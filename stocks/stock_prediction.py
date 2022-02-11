#importing variables
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
import numpy as np


class StockIndicatorsBase():

    def generate_buy_sell_signals(self, condition_buy, condition_sell, stock_dataframe, strategy):
        """Generates buy and sell signals"""
        last_signal = None
        indicators = []
        buy = []
        sell = []
        for i in range(0, len(stock_dataframe)):
            # if buy condition is true and last signal was not Buy
            if condition_buy(i, stock_dataframe) and last_signal != 'Buy':
                last_signal = 'Buy'
                indicators.append(last_signal)
                buy.append(stock_dataframe['Close'].iloc[i])
                sell.append(np.nan)
            # if sell condition is true and last signal was Buy
            elif condition_sell(i, stock_dataframe) and last_signal == 'Buy':
                last_signal = 'Sell'
                indicators.append(last_signal)
                buy.append(np.nan)
                sell.append(stock_dataframe['Close'].iloc[i])
            else:
                indicators.append(last_signal)
                buy.append(np.nan)
                sell.append(np.nan)

        stock_dataframe[f'{strategy}_Last_Signal'] = np.array(last_signal)
        stock_dataframe[f'{strategy}_Indicator'] = np.array(indicators)
        stock_dataframe[f'{strategy}_Buy'] = np.array(buy)
        stock_dataframe[f'{strategy}_Sell'] = np.array(sell)

    def _set_technical_indicators(self):
        """Sets technical indicators"""
        self.technical_indicators = pd.DataFrame()
        self.technical_indicators['Close'] = self.prices

        self.get_macd()
        self.get_rsi()
        self.get_bollinger_bands()

    def get_macd(self, window_slow=26, window_fast=12, signal=9):
        """Returns MACD dataframe and sets MACD, MACD_Histogram, MACD_Signal in technical_indicators
        dataframe

        """
        dataframe = self.technical_indicators
        window_slow = window_slow
        signal = signal
        window_fast = window_fast

        macd = MACD(self.prices, window_slow, window_fast, signal)
        dataframe['MACD'] = macd.macd()
        dataframe['MACD_Histogram'] = macd.macd_diff()
        dataframe['MACD_Signal'] = macd.macd_signal()

        self.generate_buy_sell_signals(
            lambda x, dataframe: dataframe['MACD'].values[x] < dataframe['MACD_Signal'].iloc[x],
            lambda x, dataframe: dataframe['MACD'].values[x] > dataframe['MACD_Signal'].iloc[x],
            dataframe,
            'MACD'
        )
        return dataframe

    def get_rsi(self, rsi_time_period=20, low_rsi=40, high_rsi=70):
        """Returns RSI dataframe and sets RSI in technical_indicators
        dataframe

        """
        close_prices = self.prices
        dataframe = self.technical_indicators

        rsi_time_period = rsi_time_period
        low_rsi = low_rsi
        high_rsi = high_rsi

        rsi_indicator = RSIIndicator(close_prices, rsi_time_period)
        dataframe['RSI'] = rsi_indicator.rsi()

        self.generate_buy_sell_signals(
            lambda x, dataframe: dataframe['RSI'].values[x] < low_rsi,
            lambda x, dataframe: dataframe['RSI'].values[x] > high_rsi,
            dataframe, 'RSI')

        return dataframe

    def get_bollinger_bands(self, window=20):
        """Returns Bollinger Bands dataframe and sets Bollinger_Bands_Middle, Bollinger_Bands_Upper,
        and Bollinger_Bands_Lower in technical_indicators dataframe

        """

        close_prices = self.prices
        dataframe = self.technical_indicators

        window = window

        indicator_bb = BollingerBands(close=close_prices, window=window, window_dev=2)

        # Add Bollinger Bands features
        dataframe['Bollinger_Bands_Middle'] = indicator_bb.bollinger_mavg()
        dataframe['Bollinger_Bands_Upper'] = indicator_bb.bollinger_hband()
        dataframe['Bollinger_Bands_Lower'] = indicator_bb.bollinger_lband()

        self.generate_buy_sell_signals(
            lambda x, signal: signal['Close'].values[x] < signal['Bollinger_Bands_Lower'].values[x],
            lambda x, signal: signal['Close'].values[x] > signal['Bollinger_Bands_Upper'].values[x],
            dataframe, 'Bollinger_Bands')

        return dataframe



import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 

from .trading_env import TradingEnv, Actions, Positions


class MyForexEnv(TradingEnv):

    def __init__(self, df, window_size, frame_bound, unit_side='left',trade_fee=0.0003):
        assert len(frame_bound) == 2
        assert unit_side.lower() in ['left', 'right']

        self.frame_bound = frame_bound
        self.unit_side = unit_side.lower()
        super().__init__(df, window_size)
        
        self.trade_fee = trade_fee # unit


    def _process_data(self):
        prices = self.df['EURUSD'].to_numpy()

        prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        prices = prices[self.frame_bound[0]-self.window_size:self.frame_bound[1]]
        
        ## Feature extraction
        signal_features = pd.DataFrame()
        # Use return log of prices instead of raw data
        signal_features['Timestamp'] = self.df['Timestamp']
        signal_features['EURUSD'] = self.df['EURUSD'].apply(lambda x: math.log(x/self.df['EURUSD'][0]))
        signal_features['GBPUSD'] = self.df['GBPUSD'].apply(lambda x: math.log(x/self.df['GBPUSD'][0]))
        signal_features['USDJPY'] = self.df['USDJPY'].apply(lambda x: math.log(x/self.df['USDJPY'][0]))
        signal_features['XAUUSD'] = self.df['XAUUSD'].apply(lambda x: math.log(x/self.df['XAUUSD'][0]))
        signal_features['DJI'] = self.df['DJI'].apply(lambda x: math.log(x/self.df['DJI'][0]))
        signal_features['DXY'] = self.df['DXY'].apply(lambda x: math.log(x/self.df['DXY'][0]))
        signal_features['NDXT'] = self.df['NDXT'].apply(lambda x: math.log(x/self.df['NDXT'][0]))
        signal_features['SP500'] = self.df['SP500'].apply(lambda x: math.log(x/self.df['SP500'][0]))
        
        # Create 7 and 21 days Moving Average
        signal_features['ma7'] = signal_features['EURUSD'].rolling(window=7).mean()
        signal_features['ma21'] = signal_features['EURUSD'].rolling(window=21).mean()

        # Create MACD
        signal_features["26ema"] = signal_features['EURUSD'].ewm(span=26).mean()
        signal_features["12ema"] = signal_features['EURUSD'].ewm(span=12).mean()
        signal_features['MACD'] = (signal_features["12ema"]-signal_features["26ema"])

        #Create Bollinger Bands
        signal_features['20sd'] = signal_features['EURUSD'].rolling(20).std()
        signal_features['upper_band'] = signal_features['ma21'] + (signal_features['20sd'] *2)
        signal_features['lower_band'] = signal_features['ma21'] - (signal_features['20sd'] *2)

        # Create Exponential moving average
        signal_features['ema'] = signal_features['EURUSD'].ewm(com=0.5).mean()

        # Create Momentum
        signal_features['momentum'] = signal_features['EURUSD']-1
        
        signal_features.fillna(method='backfill',inplace=True)
        #signal_features.dropna(inplace=True)
        
        scaler = MinMaxScaler()
        for (columnName, columnData) in signal_features.iteritems():
            if columnName == 'Timestamp': 
                continue
            signal_features[columnName] = scaler.fit_transform(signal_features[columnName].values.reshape(-1,1))

        return prices, signal_features


    def _calculate_reward(self, action):
        step_reward = 0.0  # pip
    
        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            price_diff = current_price - last_trade_price

            if self._position == Positions.Short:
                step_reward += -price_diff * 10000.0
            elif self._position == Positions.Long:
                step_reward += price_diff * 10000.0
                
        self.cumulative_reward += step_reward
        return self.cumulative_reward


    def _update_profit(self, action):
        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade or self._done:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            if self.unit_side == 'left':
                if self._position == Positions.Short:
                    quantity = self._total_profit * (last_trade_price - self.trade_fee)
                    self._total_profit = quantity / current_price

            elif self.unit_side == 'right':
                if self._position == Positions.Long:
                    quantity = self._total_profit / last_trade_price
                    self._total_profit = quantity * (current_price - self.trade_fee)


    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:
            position = None
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Short
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Long

            current_price = self.prices[current_tick - 1]
            last_trade_price = self.prices[last_trade_tick]

            if self.unit_side == 'left':
                if position == Positions.Short:
                    quantity = profit * (last_trade_price - self.trade_fee)
                    profit = quantity / current_price

            elif self.unit_side == 'right':
                if position == Positions.Long:
                    quantity = profit / last_trade_price
                    profit = quantity * (current_price - self.trade_fee)

            last_trade_tick = current_tick - 1

        return profit

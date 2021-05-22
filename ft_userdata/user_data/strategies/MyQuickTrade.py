from freqtrade.strategy.interface import IStrategy
from typing import Any, Dict, List, Tuple, Callable
from pandas import DataFrame
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal, Real  # noqa


class MyQuickTrade(IStrategy):
    INTERFACE_VERSION = 2

    minimal_roi = {"0": 0.02}

    stoploss = -0.02

    timeframe = '5m'

    window_size = 6

    window_perc_change = 0.2

    # Sell signal
    use_sell_signal = False
    # sell_profit_only = False
    # it doesn't meant anything, just to guarantee there is a minimal profit.
    # sell_profit_offset = 0.001
    ignore_roi_if_buy_signal = False

    trailing_stop = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 50

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    def populate_indicators(self, dataframe: DataFrame,
                            metadata: dict) -> DataFrame:
        dataframe['rolling_min'] = dataframe.close.rolling(
            self.window_size).min()
        dataframe['perc_change'] = (dataframe['close'] -
                                    dataframe['rolling_min']) / (
                                        dataframe['rolling_min'])

        return dataframe

    def apply_sparsify(self, x):
        # only keep first buy within window
        max_value = x.max()
        if max_value == 1:
            max_loc = x.argmax()
            x[:] = 0
            x.loc[x.index[max_loc]] = 1
        return x.astype(int)

    def populate_buy_trend(self, dataframe: DataFrame,
                           metadata: dict) -> DataFrame:
        dataframe.loc[(dataframe['perc_change'] > self.window_perc_change),
                      'buy'] = 1
        # only keep first buying opportunity
        dataframe['first_buy'] = dataframe['buy'].rolling(2).sum()
        dataframe.loc[dataframe.first_buy != 1, 'buy'] = 0

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame,
                            metadata: dict) -> DataFrame:
        return dataframe

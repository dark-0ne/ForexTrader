from gym.envs.registration import register
from copy import deepcopy

from . import datasets


register(
    id='forex-v0',
    entry_point='gym_anytrading.envs:ForexEnv',
    kwargs={
        'df': deepcopy(datasets.FOREX_EURUSD_1H_ASK),
        'window_size': 24,
        'frame_bound': (24, len(datasets.FOREX_EURUSD_1H_ASK))
    }
)

register(
    id='forex-v1',
    entry_point='gym_anytrading.envs:MyForexEnv',
    kwargs={
        'df': deepcopy(datasets.FOREX_EURUSD_1H_2019Jan_Dec),
        'window_size': 24,
        'frame_bound': (24, len(datasets.FOREX_EURUSD_1H_2019Jan_Dec))
    }
)
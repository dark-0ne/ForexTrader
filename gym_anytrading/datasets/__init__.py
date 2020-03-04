from .utils import load_dataset as _load_dataset
from .utils import load_custom_dataset as _load_custom_dataset



# Load FOREX datasets
FOREX_EURUSD_1H_ASK = _load_dataset('FOREX_EURUSD_1H_ASK', 'Time')
FOREX_EURUSD_1H_2019Jan_Dec = _load_custom_dataset('Jan2019-Dec2019_EURUSD')


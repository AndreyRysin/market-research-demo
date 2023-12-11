import numpy as np
from easydict import EasyDict as edict
from numpy import ndarray
from pandas import DataFrame

from calculators._basecalc import Calculator


class RollingMean(Calculator):
    "Rolling mean (aka Moving average - MA)"
    def __init__(self, config: edict, drop_corr_cols: bool, silent: bool = False) -> None:
        super().__init__(config, drop_corr_cols, silent)
        self.num_origin_cols = 1
        self.is_rolling = True

    def calculate(self, ts: DataFrame) -> None:
        for win in self.roll_wins:
            self.out.loc[:, str(win)] = ts.iloc[:, 0].rolling(win).mean()


class RollingArgmin(Calculator):
    "Rolling argmin"
    def __init__(self, config: edict, drop_corr_cols: bool, silent: bool = False) -> None:
        super().__init__(config, drop_corr_cols, silent)
        self.num_origin_cols = 1
        self.is_rolling = True

    def aux_fn(self, arr: ndarray, win: int) -> ndarray:
        out = np.full(arr.shape[0], fill_value=np.nan, dtype=float)
        for i in range(win, arr.shape[0] + 1):
            out[i - 1] = np.argmin(arr[i - win:i])
        return out / win

    def calculate(self, ts: DataFrame) -> None:
        for win in self.roll_wins:
            self.out.loc[:, str(win)] = self.aux_fn(ts.iloc[:, 0].values, win)

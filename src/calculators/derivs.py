from easydict import EasyDict as edict
from pandas import DataFrame

from calculators._basecalc import Calculator


class Direct(Calculator):
    "a'[n] = a[n]"
    def __init__(self, config: edict, drop_corr_cols: bool, silent: bool = False) -> None:
        super().__init__(config, drop_corr_cols, silent)
        self.num_origin_cols = 1

    def calculate(self, ts: DataFrame) -> None:
        self.out.loc[:, 'none'] = ts.iloc[:, 0]


class Abs(Calculator):
    "a'[n] = abs(a[n])"
    def __init__(self, config: edict, drop_corr_cols: bool, silent: bool = False) -> None:
        super().__init__(config, drop_corr_cols, silent)
        self.num_origin_cols = 1

    def calculate(self, ts: DataFrame) -> None:
        self.out.loc[:, 'none'] = ts.iloc[:, 0].abs()


class DiffOne(Calculator):
    "a'[n] = a[n] - a[n-1]"
    def __init__(self, config: edict, drop_corr_cols: bool, silent: bool = False) -> None:
        super().__init__(config, drop_corr_cols, silent)
        self.num_origin_cols = 1

    def calculate(self, ts: DataFrame) -> None:
        self.out.loc[:, 'none'] = ts.iloc[:, 0].diff(1)

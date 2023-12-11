from typing import Iterable, Tuple

from easydict import EasyDict as edict
from numpy import floating, int32, signedinteger, uint32, unsignedinteger
from pandas import DataFrame

from data_modules.feature_frame import FeatureFrame
from data_modules.rolling_correlation_evaluator import RollingCorrelationEvaluator


class Calculator:
    EPS = 1e-10

    def __init__(self, config: edict, drop_corr_cols: bool, silent: bool = False) -> None:
        self.cfg = config
        self.silent = silent
        self.calc_name = self.__class__.__name__
        self.calc_description = self.__doc__
        self.num_origin_cols = 1
        self.is_rolling = False
        self.is_ewm = False
        self.rce = RollingCorrelationEvaluator(self.cfg, multiprocessing=False)
        self.drop_corr_cols = drop_corr_cols

    def calculate(self, ts: DataFrame) -> FeatureFrame:
        self.out.loc[:, 'none'] = ts.iloc[:, 0]

    def __call__(self, input_tuple: Tuple[DataFrame, Iterable[int]]) -> Tuple[FeatureFrame, str]:
        # extract input tuple
        ts = input_tuple[0]
        roll_wins = input_tuple[1]
        self._check_ts(ts)
        # init
        self.roll_wins = self._prepare_roll_wins(roll_wins)
        self.origin_str = str(', '.join(ts.columns))
        if not self.silent:
            print('{:30}{}'.format(self.calc_name, self.origin_str))
        self.out = FeatureFrame(data=None, index=ts.index)
        # calculate
        self.calculate(ts)
        # check the output
        if not isinstance(self.out, FeatureFrame):
            raise TypeError("Type of the output must be `FeatureFrame`")
        self._check_and_cast_dtypes()
        len_inf = self.out.drop_inf()
        len_zero_std = self.out.drop_zero_std()
        len_low_stdmean = self.out.drop_low_stdmean(self.cfg)
        corr_cols = []
        if self.out.shape[1] > 1 and self.drop_corr_cols:
            corr_cols = self.rce.get_corr_cols(self.out)
            self.out = FeatureFrame(self.out.drop(columns=corr_cols))
        self.num_dropped_cols = len_inf + len_zero_std + len_low_stdmean + len(corr_cols)
        if self.silent and self.num_dropped_cols > 0:  # The condition `if self.silent` is correct (i.e. without `not`)
            print('{:30}{}'.format(self.calc_name, self.origin_str))
        # return
        return self.out, self.origin_str

    @property
    def num_dropped_columns(self) -> int:
        return self.num_dropped_cols

    def _check_ts(self, ts: DataFrame) -> None:
        if isinstance(ts, DataFrame):
            if ts.columns.shape[0] == self.num_origin_cols:
                return
            else:
                raise RuntimeError("Number of columns in the input dataframe must be {}, but {} is given".format(
                    self.num_origin_cols, ts.columns.shape[0]
                ))
        else:
            raise TypeError("Type of the input (`ts`) must be `DataFrame`")

    def _prepare_roll_wins(self, roll_wins: Iterable[int]) -> Iterable[int] | Iterable[float]:
        if self.is_rolling and self.is_ewm:
            raise ValueError('Only one of `is_rolling` and `is_ewm` can be True, not both')
        elif not self.is_rolling and not self.is_ewm:
            return None
        else:
            if isinstance(roll_wins, Iterable):
                if len(roll_wins) > 0:
                    if self.is_rolling and not self.is_ewm:
                        for win in roll_wins:
                            if not (isinstance(win, int) and win > 0):
                                raise ValueError('All the `roll_wins` values must be positive integers')
                    elif not self.is_rolling and self.is_ewm:
                        for win in roll_wins:
                            if not (isinstance(win, float) and (0.0 < win <= 1.0)):
                                raise ValueError('All the `roll_wins` values must be floats within (0.0, 1.0]')
                    return roll_wins
                else:
                    raise RuntimeError('There must be at least one element in `roll_wins`')
            else:
                raise TypeError('`roll_wins` must be iterable')

    def _check_and_cast_dtypes(self) -> None:
        for i, dtype in enumerate(self.out.dtypes):
            if issubclass(dtype.type, floating):
                self.out[self.out.columns[i]] = self.out[self.out.columns[i]].astype(self.cfg.float_dtype)
            elif issubclass(dtype.type, signedinteger):
                self.out[self.out.columns[i]] = self.out[self.out.columns[i]].astype(int32)
            elif issubclass(dtype.type, unsignedinteger):
                self.out[self.out.columns[i]] = self.out[self.out.columns[i]].astype(uint32)
            else:
                raise TypeError(f'dtype of `{self.out.columns[i]}` is {dtype} but must be float or int')

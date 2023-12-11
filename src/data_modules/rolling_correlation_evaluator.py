from multiprocessing import Pool
from typing import Iterable, Tuple

import numpy as np
import torch
from easydict import EasyDict as edict
from pandas import DataFrame, Index, Series
from torch import BoolTensor, FloatTensor
from tqdm import tqdm

from data_modules.pearson_numpy import PearsonNumpy
from data_modules.pearson_torch import PearsonTorch
from data_modules.utils import num_batches


class RollingCorrelationEvaluator:
    def __init__(self, config: edict, multiprocessing: bool = True) -> None:
        self.cfg = config
        self.original_cols = Index(list(self.cfg.original_cols.values()))
        self.sample_index = Index(range(
            self.cfg.strong_corr_sample_start_index,
            self.cfg.strong_corr_sample_stop_index,
        ))
        self.multiprocessing = multiprocessing

    def get_corr_cols(self, df: DataFrame) -> Index:
        """
        Returns the list (pandas.Index) of correlated columns (their names)
        of the given dataframe `df`.

        This is a wrapper for the `_get_corr_cols_*` methods.
        Depending on `df` properties and the config parameters as well the
        proper method is chosen automatically.
        """
        if not isinstance(df, DataFrame):
            raise TypeError(f'`df` type is `{type(df)}` but DataFrame is expected')
        self.datetime_cols = df.columns[df.columns.str.contains('dt')]
        if df.shape[1] > self.cfg.num_cols_to_use_sparse_matrices:
            corr_cols = self._get_corr_cols_sparse(df)
        elif df.shape[1] > self.cfg.num_cols_to_apply_matrix_computation:
            corr_cols = self._get_corr_cols_torch(df)
        else:
            corr_cols = self._get_corr_cols_pandas(df)
        corr_cols = np.unique(np.array(corr_cols).astype(str))
        self._print_corr_cols(corr_cols)
        return Index(corr_cols)

    def _get_corr_cols_torch(self, df: DataFrame) -> Iterable:
        "For applying to the entire table"
        # init
        pearson = PearsonTorch()
        corr_cols = []
        rcorr_isnan = BoolTensor([])
        for win in sorted(self.cfg.strong_corr_windows):
            # compose columns list
            cols = []
            for col in df.columns:
                if not (col in corr_cols) and not (col in self.original_cols) and not (col in self.datetime_cols):
                    cols.append(col)
            cols = Index(cols)
            # init
            win = int(win)
            device_str = 'cuda' if len(cols) > int(self.cfg.num_cols_to_use_cuda) else 'cpu'
            pearson.set_device(device_str)
            device = torch.device(device_str)
            sum_wincorrs = torch.zeros((len(cols), len(cols))).to(device)
            sum_wincorrs_centered = torch.zeros_like(sum_wincorrs)
            len_wincorrs = self.sample_index.shape[0] - win + 1
            df_tensor = FloatTensor(df.loc[df.index[self.sample_index], cols].values)
            # compute correlation matrix
            for i in tqdm(range(len_wincorrs)):
                wincorr = pearson.compute_corr_matrix(df_tensor[i:i + win].to(device))
                sum_wincorrs += wincorr
            mean_wincorrs = sum_wincorrs / len_wincorrs
            for i in tqdm(range(len_wincorrs)):
                wincorr = pearson.compute_corr_matrix(df_tensor[i:i + win].to(device))
                sum_wincorrs_centered += torch.square(wincorr - mean_wincorrs)
            rcorr_stds = torch.sqrt(sum_wincorrs_centered / len_wincorrs)
            rcorr_means = torch.abs(mean_wincorrs)
            # compute boolean matrices
            rcorr_isstrong = (
                (rcorr_stds < float(self.cfg.strong_corr_std_bound))
                * (rcorr_means > float(self.cfg.strong_corr_mean_bound))
            )  # AND
            if win == sorted(self.cfg.strong_corr_windows)[-1]:
                rcorr_isnan = torch.isnan(rcorr_stds) + torch.isnan(rcorr_means)  # OR
            # evaluation
            corr_cols_index = pearson.compute_true_columns_optimal_set(rcorr_isstrong)
            corr_cols.extend(cols[corr_cols_index.cpu().numpy()])
        # evaluation (continue)
        nan_cols_index = pearson.compute_true_columns_optimal_set(rcorr_isnan)
        corr_cols.extend(cols[nan_cols_index.cpu().numpy()])
        # return
        return corr_cols

    def _get_corr_cols_sparse(self, df: DataFrame) -> Iterable:
        "For applying to the entire table"
        # init
        pearson = PearsonTorch()
        corr_cols = []
        rcorr_isnan_sparses = []
        batch_size = int(self.cfg.sparse_matrices_batch_size)
        for win in sorted(self.cfg.strong_corr_windows):
            # compose columns list
            cols = []
            for col in df.columns:
                if not (col in corr_cols) and not (col in self.original_cols) and not (col in self.datetime_cols):
                    cols.append(col)
            cols = Index(cols)
            # init
            win = int(win)
            device_str = 'cuda'
            pearson.set_device(device_str)
            device = torch.device(device_str)
            rcorr_isstrong_sparses = []
            n_batches = num_batches(len(cols), batch_size)
            len_wincorrs = self.sample_index.shape[0] - win + 1
            df_tensor = FloatTensor(df.loc[df.index[self.sample_index], cols].values)
            # loop over batches
            for b in tqdm(range(n_batches)):
                # batch init
                cols_start = b * batch_size
                cols_end = (b + 1) * batch_size
                sum_wincorrs_batch = torch.zeros((len(cols[cols_start:cols_end]), len(cols))).to(device)
                sum_wincorrs_centered_batch = torch.zeros_like(sum_wincorrs_batch)
                # compute batch correlation matrix
                for i in range(len_wincorrs):
                    wincorr_batch = pearson.compute_corr_matrix(
                        df_tensor[i:i + win, cols_start:cols_end].to(device),
                        df_tensor[i:i + win].to(device),
                    )
                    sum_wincorrs_batch += wincorr_batch
                mean_wincorrs_batch = sum_wincorrs_batch / len_wincorrs
                for i in range(len_wincorrs):
                    wincorr_batch = pearson.compute_corr_matrix(
                        df_tensor[i:i + win, cols_start:cols_end].to(device),
                        df_tensor[i:i + win].to(device),
                    )
                    sum_wincorrs_centered_batch += torch.square(wincorr_batch - mean_wincorrs_batch)
                rcorr_stds_batch = torch.sqrt(sum_wincorrs_centered_batch / len_wincorrs)
                rcorr_means_batch = torch.abs(mean_wincorrs_batch)
                # compute and collect batch boolean matrices
                rcorr_isstrong_batch = (
                    (rcorr_stds_batch < float(self.cfg.strong_corr_std_bound))
                    * (rcorr_means_batch > float(self.cfg.strong_corr_mean_bound))
                )  # AND
                rcorr_isstrong_sparses.append(rcorr_isstrong_batch.to_sparse())
                if win == sorted(self.cfg.strong_corr_windows)[-1]:
                    rcorr_isnan_batch = torch.isnan(rcorr_stds_batch) + torch.isnan(rcorr_means_batch)  # OR
                    rcorr_isnan_sparses.append(rcorr_isnan_batch.to_sparse())
            # evaluation
            corr_cols_index = pearson.compute_true_columns_optimal_set_sparse(rcorr_isstrong_sparses)
            corr_cols.extend(cols[corr_cols_index.cpu().numpy()])
        # evaluation (continue)
        nan_cols_index = pearson.compute_true_columns_optimal_set_sparse(rcorr_isnan_sparses)
        corr_cols.extend(cols[nan_cols_index.cpu().numpy()])
        # return
        return corr_cols

    def _get_corr_cols_pandas(self, df: DataFrame) -> Iterable:
        "For applying either to the entire table or to the `Calculator.out`"
        # init
        pearson = PearsonNumpy()
        corr_cols = []
        rcorr_isnan = np.array([], dtype=bool)
        for win in sorted(self.cfg.strong_corr_windows):
            # compose columns list
            cols = []
            for col in df.columns:
                if not (col in corr_cols) and not (col in self.original_cols) and not (col in self.datetime_cols):
                    cols.append(col)
            cols = Index(cols)
            cols_iter = tqdm(range(len(cols))) if len(cols) > len(self.cfg.rolling_windows) else range(len(cols))
            # init
            win = int(win)
            rcorr_stds = np.zeros((len(cols), len(cols)), dtype=self.cfg.float_dtype)
            rcorr_means = np.zeros_like(rcorr_stds)
            # compute correlation matrix
            if self.multiprocessing:
                for i in cols_iter:
                    with Pool(int(self.cfg.multiprocessing_num_workers)) as pool:
                        for rcorr_std, rcorr_mean, j_ in pool.map(
                            self._pandas_rcorr_func,
                            [
                                (
                                    df.loc[df.index[self.sample_index], cols[i]],
                                    df.loc[df.index[self.sample_index], cols[j]],
                                    win,
                                    j,
                                )
                                for j in range(i + 1, len(cols))
                            ],
                        ):
                            rcorr_stds[i, j_] = rcorr_std
                            rcorr_means[i, j_] = rcorr_mean
            else:
                for i in cols_iter:
                    for j in range(i + 1, len(cols)):
                        rcorr_std, rcorr_mean, j_ = self._pandas_rcorr_func((
                            df.loc[df.index[self.sample_index], cols[i]],
                            df.loc[df.index[self.sample_index], cols[j]],
                            win,
                            j,
                        ))
                        rcorr_stds[i, j_] = rcorr_std
                        rcorr_means[i, j_] = rcorr_mean
            rcorr_stds += np.transpose(rcorr_stds)
            rcorr_means += np.transpose(rcorr_means)
            # compute boolean matrices
            rcorr_isstrong = (
                (rcorr_stds < float(self.cfg.strong_corr_std_bound))
                * (rcorr_means > float(self.cfg.strong_corr_mean_bound))
            )  # AND
            if win == sorted(self.cfg.strong_corr_windows)[-1]:
                rcorr_isnan = np.isnan(rcorr_stds) + np.isnan(rcorr_means)  # OR
            # evaluation
            corr_cols_index = pearson.compute_true_columns_optimal_set(rcorr_isstrong)
            corr_cols.extend(cols[corr_cols_index])
        # evaluation (continue)
        nan_cols_index = pearson.compute_true_columns_optimal_set(rcorr_isnan)
        corr_cols.extend(cols[nan_cols_index])
        # return
        return corr_cols

    def _pandas_rcorr_func(self, input_tuple: Tuple[Series, Series, int, int]) -> Tuple[float, float, int]:
        # input tuple extraction
        series_col1 = input_tuple[0]
        series_col2 = input_tuple[1]
        win = input_tuple[2]
        col2_index = input_tuple[3]
        # computation
        rcorr = series_col1.rolling(win).corr(series_col2)
        rcorr_std = float(rcorr.std(ddof=0))
        rcorr_mean = float(np.abs(rcorr.mean()))
        # return
        return rcorr_std, rcorr_mean, col2_index

    def _print_corr_cols(self, corr_cols: Iterable[str]) -> None:
        if len(corr_cols) > 0:
            print('\tThe following columns are strongly correlated ({}):\n{}'.format(
                len(corr_cols), ', '.join(corr_cols)
            ))
            print(f'\tThe total number of strongly correlated columns: {len(corr_cols)}')

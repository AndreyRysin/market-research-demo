import json
import pickle
from multiprocessing import Pool
from os import makedirs
from os.path import join
from typing import Tuple

import numpy as np
from dslib.base_modules.utils import Timestamp
from easydict import EasyDict as edict
from numpy import ndarray


class InferenceMeanStdCalculator:
    def __init__(self, config: edict, num_dfs: int, num_workers: int = None) -> None:
        self.num_workers = config.multiprocessing_num_workers if num_workers is None else num_workers
        self.dtype = config.float_dtype
        self.root_path = config.paths.inference_metadata
        self.num_dfs = num_dfs
        self.init_df_is_passed_mean = False
        self.init_df_is_passed_var = False

    def compute_df_means(self, arr: ndarray) -> None:
        if not self.init_df_is_passed_mean:
            self._check_arr_shape(arr)
            self.df_sums = np.zeros((self.num_dfs, arr.shape[1]), dtype=float)
            self.df_lengths = np.zeros(self.num_dfs, dtype=int)
            self.chunksize = int(np.ceil(arr.shape[1] / self.num_workers))
            self.init_df_is_passed_mean = True
            self.df_means_counter = 0
        # means on columns - for each columns chunk separately
        if self.num_workers == 1:
            self.df_sums[self.df_means_counter] = np.sum(arr, axis=0)
        else:
            with Pool(self.num_workers) as pool:
                for i, chunk_sums in pool.map(
                    compute_sum,
                    [
                        (i, arr[:, i: i + self.chunksize])
                        for i in range(0, arr.shape[1], self.chunksize)
                    ],
                ):
                    self.df_sums[self.df_means_counter, i: i + self.chunksize] = chunk_sums
        self.df_lengths[self.df_means_counter] = arr.shape[0]
        self.df_means_counter += 1

    def compute_df_vars(self, arr: ndarray) -> None:
        if not self.init_df_is_passed_var:
            self.df_vars = np.zeros_like(self.df_sums)
            self.init_df_is_passed_var = True
            self.df_vars_counter = 0
            # total means and length on all the dataframes
            if self.df_means_counter == self.num_dfs:
                self.total_length = np.sum(self.df_lengths)
                self.total_means = np.array(np.sum(self.df_sums, axis=0) / self.total_length, dtype=self.dtype)
            else:
                raise RuntimeError('Number of processed dataframes and number provided to init are not the same')
        # variances on columns - for each columns chunk separately
        if self.num_workers == 1:
            self.df_vars[self.df_vars_counter] = np.sum(np.square(arr - self.total_means), axis=0)
        else:
            with Pool(self.num_workers) as pool:
                for i, chunk_vars in pool.map(
                    compute_var,
                    [
                        (i, arr[:, i: i + self.chunksize], self.total_means[i: i + self.chunksize])
                        for i in range(0, arr.shape[1], self.chunksize)
                    ],
                ):
                    self.df_vars[self.df_vars_counter, i: i + self.chunksize] = chunk_vars
        self.df_vars_counter += 1
        # total stds on all the dataframes
        if self.df_vars_counter == self.num_dfs:
            self.total_stds = np.array(np.sqrt(np.sum(self.df_vars, axis=0) / self.total_length), dtype=self.dtype)

    def save_meanstd(self, ticker: str, prefix: str) -> None:
        """
        Saves meanstd as a tuple of the mean and std arrays and a dict of the
        mean and std lists.

        The tuple is saved as a binary file (pickle); the dict is saved as a
        text file (json).

        The tuple:
            (means, stds)

        The dict:
            {'means': means, 'stds': stds}

        `prefix` can be used, for example, for specifying the data origin:
        'true' if the data is the ground truth one, or 'pred' if the data is
        predicted with a model.
        """
        # check
        if not hasattr(self, 'total_means'):
            raise RuntimeError('Compute means and vars first')
        if not hasattr(self, 'total_stds'):
            raise RuntimeError('Compute vars first')
        # init
        ts = Timestamp().filename_timestamp
        makedirs(join(self.root_path, ts), exist_ok=True)
        # save tuple (pickle)
        path_to_pkl = join(self.root_path, ts, f'{ts}_{ticker}_{prefix}_meanstd_tuple.pkl')
        meanstd_tuple = (self.total_means, self.total_stds)
        with open(path_to_pkl, 'wb') as f:
            pickle.dump(meanstd_tuple, f)
        # save dict (json)
        path_to_json = join(self.root_path, ts, f'{ts}_{ticker}_{prefix}_meanstd_dict.json')
        meanstd_dict = {'means': self.total_means.tolist(), 'stds': self.total_stds.tolist()}
        with open(path_to_json, 'w') as f:
            json.dump(meanstd_dict, f)

    @property
    def means(self) -> ndarray:
        return self.total_means

    @property
    def stds(self) -> ndarray:
        return self.total_stds

    def _check_arr_shape(self, arr: ndarray) -> None:
        if not arr.ndim == 2:
            raise RuntimeError(f'Input array is {arr.ndim}D but 2D is expected')


def compute_sum(input_tuple: Tuple[int, ndarray]) -> Tuple[int, ndarray]:
    i = input_tuple[0]
    arr = input_tuple[1]
    return i, np.sum(arr, axis=0)


def compute_var(input_tuple: Tuple[int, ndarray, ndarray]) -> Tuple[int, ndarray]:
    i = input_tuple[0]
    arr = input_tuple[1]
    means = input_tuple[2]
    return i, np.sum(np.square(arr - means), axis=0)

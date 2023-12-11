from typing import Iterable

import torch
from torch import BoolTensor, FloatTensor, IntTensor, LongTensor, Tensor


class PearsonTorch:
    """
    PyTorch implementation for computing the Pearson correlation coefficient.

    Supports computations on both CPU and CUDA. It is recommended to compare
    the speed on both devices in order to choose the optimal device according
    to a task.
    """
    def __init__(self) -> None:
        self.device = torch.device('cuda')

    def set_device(self, device: str) -> None:
        """
        Sets the device ('cuda' or 'cpu') to compute on.

        If the method is not called, the default device is 'cuda'.
        """
        if device in ('cuda', 'cpu'):
            self.device = torch.device(device)
        else:
            raise ValueError('`device` must be "cuda" or "cpu"')

    def compute_corr_matrix(
        self, x: FloatTensor, y: FloatTensor | None = None
    ) -> FloatTensor:
        """
        Computes the Pearson correlation coefficients for all possible pairs of
        columns on two matrices or on one matrix (itself).

        1. If both the `x` and `y` matrices are passed, it is computed the
        coefficients between the corresponding columns of two tables.
        The resulting matrix's shape is (x.shape[1], y.shape[1]).

        2. If only the `x` matrix is passed, it is computed the coefficients
        between self columns. In this case, the resulting matrix's main
        diagonal consists strictly of ones.
        The resulting matrix's shape is (x.shape[1], x.shape[1]).
        """
        xv = x - torch.mean(x, dim=0)
        xvss = torch.sum(xv * xv, dim=0)
        yv = xv if y is None else y - torch.mean(y, dim=0)
        yvss = xvss if y is None else torch.sum(yv * yv, dim=0)
        result = torch.matmul(xv.T, yv) / torch.sqrt(torch.outer(xvss, yvss))
        result = torch.clip(result, -1.0, 1.0)
        return result

    def compute_true_columns_optimal_set(self, boolean_matrix: BoolTensor) -> IntTensor:
        """
        Takes a square boolean matrix of size (num_columns, num_columns) as
        input. This matrix is a derivative from the correlation coefficients
        matrix obtained with `mat_pearson_corr`.

        How to obtain the boolean matrix for input (prerequisites):
        The correlation coefficients are evaluated priorly according to some
        condition. Those of them that meet the condition (e.g. correlation is
        too high) turn into True, otherwise - False.

        Rows and columns of the boolean matrix correspond to the proper
        original table's columns whose correlation is being evaluated.

                a b c d e
            a [[0 1 0 1 0]    0 -> False
            b  [1 0 0 1 1]    1 -> True
            c  [0 0 0 0 0]
            d  [1 1 0 0 0]
            e  [0 1 0 0 0]]

        The goal is to choose optimal (minimal) set of the columns (their
        indices) whose values are True (or 1).

        In the example, the columns to return are [`a`, `b`]. Others have no
        True's in their crossings.
        """
        # check the shape
        if not boolean_matrix.shape[0] == boolean_matrix.shape[1]:
            raise RuntimeError('The input matrix must be square')
        # init
        binary_matrix = boolean_matrix.type(torch.int8)
        binary_matrix *= (
            torch.eye(boolean_matrix.shape[0], dtype=torch.int8) * -1 + 1
        ).to(self.device)
        counter = 0
        is_true_cols_index = []
        sum_binary_matrix = torch.sum(binary_matrix, dim=0)
        # columns selection loop
        while not torch.all(sum_binary_matrix == 0):
            col_index = int(torch.argmax(sum_binary_matrix).item())
            binary_matrix[col_index, :] = 0
            binary_matrix[:, col_index] = 0
            is_true_cols_index.append(col_index)
            sum_binary_matrix = torch.sum(binary_matrix, dim=0)
            counter += 1
            if counter >= len(binary_matrix):
                raise RuntimeError('Correlation evaluation error')
        # return
        values = IntTensor(sorted(is_true_cols_index)).to(self.device)
        return values

    def compute_true_columns_optimal_set_sparse(
        self, boolean_sparse_matrices: Iterable[Tensor]
    ) -> IntTensor:
        """
        Implementation for sparse boolean matrices.
        """
        # check the shape
        shape_0 = 0
        for boolean_sparse_matrix in boolean_sparse_matrices:
            shape_0 += boolean_sparse_matrix.size()[0]
            shape_1 = boolean_sparse_matrix.size()[1]
        if not shape_0 == shape_1:
            raise RuntimeError('The input matrix must be square')
        # init
        counter = 0
        is_true_cols_index = []
        sum_binary_matrix = self._sum_binary_matrix(boolean_sparse_matrices, LongTensor(is_true_cols_index))
        # columns selection loop
        while not torch.all(sum_binary_matrix == 0):
            col_index = int(torch.argmax(sum_binary_matrix).item())
            is_true_cols_index.append(col_index)
            sum_binary_matrix = self._sum_binary_matrix(boolean_sparse_matrices, LongTensor(is_true_cols_index))
            counter += 1
            if counter >= shape_0:
                raise RuntimeError('Correlation evaluation error')
        # return
        values = IntTensor(sorted(is_true_cols_index)).to(self.device)
        return values

    def _sum_binary_matrix(
        self, boolean_sparse_matrices: Iterable[Tensor], index_to_assign_zero: LongTensor
    ) -> IntTensor:
        batch_size = boolean_sparse_matrices[0].shape[0]
        total_sum = torch.zeros(
            boolean_sparse_matrices[0].shape[1], dtype=torch.int
        ).to(self.device)
        index_to_assign_zero = index_to_assign_zero.to(self.device)
        for b, boolean_sparse_matrix in enumerate(boolean_sparse_matrices):
            cols_start = b * batch_size
            cols_end = (b + 1) * batch_size
            boolean_matrix = boolean_sparse_matrix.to_dense()
            binary_matrix = boolean_matrix.type(torch.int8)
            diag = torch.zeros_like(binary_matrix)
            diag[:, cols_start:cols_end] += torch.eye(
                diag.shape[0], dtype=torch.int8
            ).to(self.device)
            diag = diag * -1 + 1
            binary_matrix *= diag.to(self.device)
            idx = torch.full([binary_matrix.shape[1]], False, dtype=torch.bool).to(self.device)
            idx[index_to_assign_zero] = True
            binary_matrix[idx[cols_start:cols_end], :] = 0
            binary_matrix[:, index_to_assign_zero] = 0
            total_sum += torch.sum(binary_matrix, dim=0)
        return total_sum

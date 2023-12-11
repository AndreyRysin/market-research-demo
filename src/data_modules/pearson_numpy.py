import numpy as np
from numpy import ndarray


class PearsonNumpy:
    """
    NumPy implementation for computing the Pearson correlation coefficient.
    """
    def __init__(self) -> None:
        pass

    def compute_corr_matrix(
        self, x: ndarray, y: ndarray | None = None
    ) -> ndarray:
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
        xv = x - np.mean(x, axis=0)
        xvss = np.sum(xv * xv, axis=0)
        yv = xv if y is None else y - np.mean(y, axis=0)
        yvss = xvss if y is None else np.sum(yv * yv, axis=0)
        result = np.matmul(np.transpose(xv), yv) / np.sqrt(np.outer(xvss, yvss))
        result = np.asarray(np.clip(result, -1.0, 1.0))
        return result

    def compute_true_columns_optimal_set(self, boolean_matrix: ndarray) -> ndarray:
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
        binary_matrix = boolean_matrix.astype(int)
        np.fill_diagonal(binary_matrix, 0)
        counter = 0
        is_true_cols_index = []
        # columns selection loop
        while not np.all(np.sum(binary_matrix, axis=0) == 0):
            col_index = np.argmax(np.sum(binary_matrix, axis=0))
            binary_matrix[col_index, :] = 0
            binary_matrix[:, col_index] = 0
            is_true_cols_index.append(col_index)
            counter += 1
            if counter >= len(binary_matrix):
                raise RuntimeError('Correlation evaluation error')
        values = np.array(sorted(is_true_cols_index), dtype=int)
        return values

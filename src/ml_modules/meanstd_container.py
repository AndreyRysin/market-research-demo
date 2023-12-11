from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

from numpy import ndarray


@dataclass
class MeanStdContainer:
    """
    Contains column-wise means and column-wise standard deviations (stds).

    Class variables:
        dt_means, dt_stds, ft_means, ft_stds, tg_means, tg_stds

    Namespace description:
        'dt' - datetime features
        'ft' - features
        'tg' - targets
    """
    dt_means: ndarray | None
    dt_stds: ndarray | None
    ft_means: ndarray | None
    ft_stds: ndarray | None
    tg_means: ndarray | None
    tg_stds: ndarray | None

    @property
    def meanstd_tuple(self) -> Tuple[ndarray | None, ...]:
        """
        Return:
            dt_means, dt_stds, ft_means, ft_stds, tg_means, tg_stds
        """
        return self.dt_means, self.dt_stds, self.ft_means, self.ft_stds, self.tg_means, self.tg_stds

    @property
    def meanstd_dict(self) -> Dict[str, Iterable[float] | None]:
        """
        Return:
            {
                'dt_means': dt_means, 'dt_stds': dt_stds,
                'ft_means': ft_means, 'ft_stds': ft_stds,
                'tg_means': tg_means, 'tg_stds': tg_stds,
            }

        All the arrays are converted to lists.
        """
        return {
            'dt_means': self._tolist(self.dt_means),
            'dt_stds': self._tolist(self.dt_stds),
            'ft_means': self._tolist(self.ft_means),
            'ft_stds': self._tolist(self.ft_stds),
            'tg_means': self._tolist(self.tg_means),
            'tg_stds': self._tolist(self.tg_stds),
        }

    def _tolist(self, arr: ndarray | None) -> Iterable[float] | None:
        out = arr.tolist() if isinstance(arr, ndarray) else arr
        return out

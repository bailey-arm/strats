from collections.abc import Mapping

import pandas as pd

from .base import Alpha
from .data import DataCol


class OvernightZScore(Alpha):
    """Overnight-return z-score, lagged one day, clipped and rescaled to [-1, 1].

    The overnight return is z-scored with a ``period``-span EWM, shifted so the
    signal is tradable on the next open, clipped at ``±threshold`` and divided
    by ``threshold`` so the output lies in ``[-1, 1]``.
    """

    def __init__(self, period: int = 300, threshold: float = 2.0):
        self.period = period
        self.threshold = threshold

    def data_needed(self) -> frozenset[DataCol]:
        return frozenset({DataCol.OVN_RETURN})

    def calc(self, data: Mapping[DataCol, pd.DataFrame]) -> pd.DataFrame:
        ovn = data[DataCol.OVN_RETURN]
        ewm = ovn.ewm(span=self.period, min_periods=1)
        z = (ovn - ewm.mean()) / ewm.std()
        return z.shift(1).clip(-self.threshold, self.threshold).fillna(0) / self.threshold

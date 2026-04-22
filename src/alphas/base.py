from abc import ABC, abstractmethod
from collections.abc import Mapping

import pandas as pd

from .data import DataCol


class Alpha(ABC):
    """Base class for an alpha signal.

    Subclasses declare the data they need via :meth:`data_needed` and implement
    the signal computation in :meth:`calc`. Call the instance like a function
    to run: the base class checks the data package contains every required
    column, filters it down to just those columns, and hands it to ``calc``.
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def data_needed(self) -> frozenset[DataCol]:
        """Return the set of :class:`DataCol` this alpha reads."""

    @abstractmethod
    def calc(self, data: Mapping[DataCol, pd.DataFrame]) -> pd.DataFrame:
        """Compute the signal from ``data`` (pre-filtered to declared columns)."""

    def __call__(self, data: Mapping[DataCol, pd.DataFrame]) -> pd.DataFrame:
        needed = self.data_needed()
        missing = needed - data.keys()
        if missing:
            raise KeyError(
                f"{self.name} missing required data columns: "
                f"{sorted(c.value for c in missing)}"
            )
        filtered = {col: data[col] for col in needed}
        return self.calc(filtered)

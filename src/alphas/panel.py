from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import DataCol


_PIVOT_COLS: frozenset[DataCol] = frozenset({
    DataCol.OPEN,
    DataCol.HIGH,
    DataCol.LOW,
    DataCol.CLOSE,
    DataCol.VOLUME,
})


@dataclass(frozen=True)
class _Derived:
    deps: frozenset[DataCol]
    fn: Callable[[Mapping[DataCol, pd.DataFrame]], pd.DataFrame]


def _clean(s: pd.DataFrame) -> pd.DataFrame:
    return s.replace([np.inf, -np.inf], np.nan).fillna(0)


def _day_return(d: Mapping[DataCol, pd.DataFrame]) -> pd.DataFrame:
    return _clean(d[DataCol.CLOSE] / d[DataCol.OPEN] - 1)


def _ovn_return(d: Mapping[DataCol, pd.DataFrame]) -> pd.DataFrame:
    return _clean(d[DataCol.OPEN] / d[DataCol.CLOSE].shift(1) - 1)


_DERIVED: Mapping[DataCol, _Derived] = {
    DataCol.DAY_RETURN: _Derived(frozenset({DataCol.OPEN, DataCol.CLOSE}), _day_return),
    DataCol.OVN_RETURN: _Derived(frozenset({DataCol.OPEN, DataCol.CLOSE}), _ovn_return),
}


def build_data_package(
    df: pd.DataFrame,
    cols: Iterable[DataCol],
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    ffill_limit: int = 2,
) -> dict[DataCol, pd.DataFrame]:
    """Assemble a ``{DataCol: wide DataFrame}`` package from a long panel.

    Raw OHLCV columns are pivoted and forward-filled up to ``ffill_limit`` days;
    derived columns (returns) are computed from their dependencies.
    """
    requested = set(cols)
    unknown = requested - _PIVOT_COLS - _DERIVED.keys()
    if unknown:
        raise ValueError(f"Unsupported DataCol(s): {sorted(c.value for c in unknown)}")

    raw_needed = {c for c in requested if c in _PIVOT_COLS}
    for c in requested:
        if c in _DERIVED:
            raw_needed |= _DERIVED[c].deps

    indexed = df.set_index([date_col, ticker_col])
    raw: dict[DataCol, pd.DataFrame] = {
        c: indexed[c.value].unstack(ticker_col).ffill(limit=ffill_limit).fillna(0)
        for c in raw_needed
    }

    return {
        c: raw[c] if c in raw else _DERIVED[c].fn(raw)
        for c in requested
    }

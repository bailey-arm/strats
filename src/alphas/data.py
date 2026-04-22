from enum import StrEnum


class DataCol(StrEnum):
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"
    DAY_RETURN = "day_return"
    OVN_RETURN = "ovn_return"

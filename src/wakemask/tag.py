"""Container for canonical tags and generated variables."""
from enum import StrEnum, unique


# PYWAKE ALIASES
@unique
class PWDim(StrEnum):
    """Canonical pywake's objects dimension names."""

    TIME = 'time'
    WD = 'wd'
    WS = 'ws'
    WT = 'wt'
    TYPE = 'type'


@unique
class PWTag(StrEnum):
    """Canonical pywake's objects variables names."""

    # SimRes variables
    CT = 'CT'
    P = 'P'
    POWER = 'Power'
    TI = 'TI'
    TI_EFF = 'TI_eff'
    WD = 'WD'
    WS = 'WS'
    WS_EFF = 'WS_eff'
    X = 'x'
    Y = 'y'

    # Operating state
    # NOTE: This may be moved to another class without notice
    OP_STATE = 'op_state'

"""A bunch of utility method for masked array simulation."""
import logging
from copy import deepcopy
from typing import get_args

import pandas as pd

from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import (
    PowerCtFunctionList,
    PowerCtTabular,
)

# PACKAGE IMPORTS
from wakemask.config import PwUnit
from wakemask.tag import PWDim, PWTag


# LOGGER
logger = logging.getLogger(__name__)


# UTILS
def offline_powerct() -> PowerCtTabular:
    """
    Return a power ct function object corresponding to offline conditions.
    """
    return PowerCtTabular(ws=[0., 0.], power=[0., 0.], power_unit='w', ct=[0., 0.])


def with_op_state(wturb: WindTurbine) -> WindTurbine:
    """
    Given a PyWake's wind turbine object, make a copy and replace it's power and \
    ct functions by a PowerCtFunctionList object keyed on "op_state" and \
    including and "offline" power curve.

    Notes
    -----
    Does nothing if wturb.powerCtFunction is already a PowerCtFunctionList.

    Parameters
    ----------
    wturb: WindTurbine
        Input wind turbine.

    Returns
    -------
        WindTurbine
    """
    wturb = deepcopy(wturb)
    if isinstance(wturb.powerCtFunction, PowerCtFunctionList):
        logger.warning(
            'Power and ct function is already an instance of "PowerCtFunctionList" '
            'return a copy of the original object without modification.'
        )
        return wturb
    wturb.powerCtFunction = PowerCtFunctionList(
        key=PWTag.OP_STATE,
        powerCtFunction_lst=[
            offline_powerct(),
            wturb.powerCtFunction
        ]
    )
    return wturb


def to_pct(
    df: pd.DataFrame,
    pw_unit: PwUnit = 'kW',
) -> PowerCtTabular:
    """
    Given a dataframe containing power- and CT-curves data, \
    create the corresponding powerCtFunction object.

    Parameters
    ----------
    df: DataFrame
        Input dataframe.
    pw_unit: 'W' | 'kW' | 'MW' |'GW'
        Power curve unit.

    Returns
    -------
        PowerCtTabular

    Raises
    ------
        ValueError
    """
    # Check
    allowed = get_args(PwUnit)
    if pw_unit not in allowed:
        raise ValueError(
            f'Invalid power units: {pw_unit}. Allowed units are {allowed}.'
            'Please check your inputs.'
        )
    if (missing := {PWDim.WS, PWTag.POWER, PWTag.CT}) - set(df.columns):
        raise ValueError(
            'The input power and ct curve dataframe is missing requested columns. '
            f'Missing columns: {missing}. DataFrame columns: {df.columns}. '
            'Please check your inputs.'
        )
    out = PowerCtTabular(
        ws=df[PWDim.WS].to_numpy(),
        power=df[PWTag.POWER].to_numpy(),
        ct=df[PWTag.CT].to_numpy(),
        power_unit='kW',  # TODO: Make some tests
    )
    return out

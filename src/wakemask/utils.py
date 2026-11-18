"""A bunch of utility method for masked array simulation."""
import logging
from copy import deepcopy

from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import (
    PowerCtFunctionList,
    PowerCtTabular,
)

# PACKAGE IMPORTS
from wakemask.tag import PWTag


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

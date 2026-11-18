"""Mask array model for wind farm simulation with offline turbines."""
import abc
import logging
from typing import Any, Iterable

import numpy as np

from py_wake.wind_farm_models.engineering_models import EngineeringWindFarmModel
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtFunctionList, PowerCtTabular

import xarray as xr

# PACKAGE IMPORTS
from wakemask.tag import PWDim, PWTag


# LOGGER
logger = logging.getLogger(__name__)


# ERROR HANDLER
class NotFittedError(ValueError, AttributeError):
    """
    Custom error class for non-fitted object, inspired from sklearn own's \
    class.

    See: https://scikit-learn.org/stable/modules/generated/sklearn.exceptions.NotFittedError.html)
    """  # noqa: E501
    pass


# UTILS
def as_op_state(
    op_state: Iterable | None,
    n_turb: int,
    stamps: Iterable | None = None,
) -> np.ndarray:
    """
    Given the turbine operating states, validate against the other \
    inputs.

    Parameters
    ----------
    op_states: Iterable | None
        Input operating states.
    n_turbs: int
        Number of turbine in the simulation. This is use to verify the \
        size of the op_state object.
    stamps: Iterable | None = None
        Timestamps of the timeseries simulation. This is use to verify the \
        size of the op_state object.

    Returns
    -------
        np.ndarray
    """
    # wt = np.arange(n_turb)

    if op_state is None:
        return np.ones(n_turb, bool)
        # return xr.DataArray(np.ones(n_turb, bool), dims='wt', coords={'wt': wt})

    arr = np.asarray(op_state)
    if arr.dtype != bool:
        if not np.issubdtype(arr.dtype, np.number):
            msg = 'Operating states must be bool of {0, 1} numeric. ' \
                  f'Received: {arr.dtype}. Please check your inputs.'
            logger.error(msg)
            raise TypeError(msg)
        if not ((arr == 0) | (arr == 1)).all():
            # NOTE: Also rejects NaN
            msg = 'Operating states must be binary ({0, 1} or bool). Partial' \
                  'operation (e.g, curtailement) are not supported.'
            logger.error(msg)
            raise ValueError(msg)
        arr = arr.astype(bool)

    if arr.ndim == 1:
        # NOTE: Valid in both modes
        if arr.shape != (n_turb,):
            msg = f'Operating states must have shape ({n_turb},), got {arr.shape}. ' \
                  'Please check your inputs.'
            logger.error(msg)
            raise ValueError(msg)
        return arr
        # return xr.DataArray(arr, dims='wt', coords={'wt': wt})

    if arr.ndim == 2:
        # NOTE: Valid only in timeseries mode.
        if stamps is None:
            msg = 'Grid mode takes a single operating state of shape (n_turb,).' \
                  'Please check your inputs.'
            logger.error(msg)
            raise ValueError(msg)
        n_time = len(stamps)
        if arr.shape != (n_time, n_turb):
            hint = ' It looks transposed.' if arr.shape == (n_turb, n_time) else ''
            msg = f'Operating states must have shape (n_time, n_turb) = ' \
                  f'({n_time}, {n_turb}), got {arr.shape}.{hint}'
            logger.error(msg)
            raise ValueError(msg)
        return arr
        # return xr.DataArray(arr, dims=('time', 'wt'), coords={'time': stamps, 'wt': wt})

    msg = f'Operating states must be 1-D or 2-D, got {arr.ndim}-D' \
          'Please check your inputs.'
    logger.error(msg)
    raise ValueError(msg)


# INTERFACE
class MaskedArrayModel(abc.ABC):
    """Interface class for wind farm simulation on masked array."""

    def __init__(self, wfarm: EngineeringWindFarmModel) -> None:
        """Create class instance."""
        # Prevent un-tested behavior
        _wturb = wfarm.windTurbines
        if not isinstance(_wturb, WindTurbine):
            msg = 'Wakemask (alpha version) only works for single wind turbine. ' \
                  'Simulation for multiple type wind turbines not yet ' \
                  'implemented.'
            logger.error(msg)
            raise NotImplementedError(msg)

        # Test structure of the power and Ct curve
        _pct_fun = _wturb.powerCtFunction
        if not isinstance(_pct_fun, PowerCtFunctionList):
            msg = 'Masked array simulation require to provide power ct functions ' \
                  'for offline conditions. Received a plain "powerCtFunction". ' \
                  'Use the `with_op_state` from wakemask.utils to create a ' \
                  'to a powerCtFunctionList with the "offline" power curve.'
            logger.error(msg)
            raise ValueError(msg)

        test = all([
            isinstance(pct, PowerCtTabular)
            for pct in _pct_fun.windTurbineFunction_lst
        ])
        if not test:
            msg = 'Wakemask (alpha version) only works with one dimensional ' \
                  'power curves. All power ct function must "PowerCtTabular".'
            logger.error(msg)
            raise NotImplementedError(msg)

        if _pct_fun.key != PWTag.OP_STATE:
            msg = f'Turbine power CT function must have the the key "{PWTag.OP_STATE}". ' \
                  f'Received a powerCtFunctionList with key {_pct_fun.key}. ' \
                  'Use the `with_op_state` from wakemask.utils to create a ' \
                  'to a powerCtFunctionList with the "offline" power curve.'
            logger.error(msg)
            raise ValueError(msg)

        self.wfarm = wfarm

        # Internal state
        self._is_fitted: bool = False
        self.base_res: xr.Dataset | None = None
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._ws: np.ndarray | None = None
        self._wd: np.ndarray | None = None

    # Properties
    @property
    def name(self) -> str:
        """Return class name."""
        return type(self).__name__

    @property
    def nturb(self) -> int:
        """Return the number of turbine in the stored layout."""
        return len(self._x)

    # Utils
    def _check_fitted(self) -> None:
        """
        Check that the model is fitted and raise a `NotFittedError` \
        if otherwise.

        Raises
        ------
            NotFittedError
        """
        if not self._is_fitted:
            msg = 'Model is not fitted. This likely happen for calling the ' \
                  '"predict" method before "fit". Call "fit" first and retry.'
            logger.error(msg)
            raise NotFittedError(msg)

    def _as_conditions(
        self,
        ws: float | Iterable | None = None,
        wd: float | Iterable | None = None,
        time: bool | Iterable | None = False,
    ) -> tuple[np.ndarray | xr.DataArray, np.ndarray | xr.DataArray, None | Iterable]:
        """
        Given the inflow condition for predict, verify that they have the \
        correct shape.

        Parameters
        ----------
        wd: float | Iterable | None
            Wind direction(s). Uses the full grid used for fitting if None is passed.
        ws: float | Iterable | None
            Wind speed(s). Uses the full grid used for fitting if None is passed.
        time: bool | Iterable | None
            If False (default), the simulation will be computed for the full wd x ws matrix
            If True, the wd and ws will be considered as a time series of flow conditions with time stamp 0,1,..,n
            If array_like: same as True, but the time array is used as flow case time stamp.

        Returns
        -------
            tuple[np.ndarray | xr.DataArray, np.ndarray | xr.DataArray]
        """  # noqa: E501
        # For fixed grid
        if time is False or time is None:
            ws = self._ws if ws is None else np.asarray(ws, float)
            wd = self._wd if wd is None else np.asarray(wd, float)
            return ws, wd % 360, None  # TODO: Decide if we want to take the modulo here.

        # For time simulation
        if ws is None or wd is None:
            # NOTE: np.asarray(None, float) returns NaNs. Unser what the
            # behavior with 'nearest' in sel (probably an error) but it's
            # better to fail loudly here.
            msg = 'At least one wind speed or wind direciton is None. ' \
                  'Inflow condition must be provided for time simulation.' \
                  'Please check your inputs.'
            logger.error(msg)
            raise ValueError(msg)
        ws, wd = np.asarray(ws, float), np.asarray(wd, float) % 360
        if ws.shape != wd.shape or ws.ndim != 1:
            msg = 'For time simulation, wind speed and wind direction ' \
                  'must de one-dimensional array of the same length.' \
                  f'Input shapes: ws.shape = {ws.shape}, wd.shape = {wd.shape}. ' \
                  'Please check your inputs.'
            logger.error(msg)
            raise ValueError(msg)
        stamps = np.arange(ws.size) if time is True else np.asarray(time)
        ds = xr.Dataset(
            {PWDim.WS: (PWDim.TIME, ws), PWDim.WD: (PWDim.TIME, wd)},
            coords={PWDim.TIME: (PWDim.TIME, stamps)}
        )
        return ds[PWDim.WS], ds[PWDim.WD], stamps

    # Interface
    def fit(
        self,
        x: Iterable,
        y: Iterable,
        ws: float | Iterable,
        wd: float | Iterable,
        ti: float = 0.1,
        **sim_kw: Any,
    ) -> 'MaskedArrayModel':
        """
        Given the turbine locations and wind conditions, \
        fit the model internal state.

        Parameters
        ----------
        x : array_like
            Wind turbine x positions
        y : array_like
            Wind turbine y positions
        wd : int or array_like
            Wind direction(s)
        ws : int, float or array_like
            Wind speed(s)
        ti: float
            Turbulence intensity.
        **kwargs: Any
            Other argument to be passed to the wfarm.__call__ method. \
            Refer to https://topfarm.pages.windenergy.dtu.dk/PyWake/notebooks/EngineeringWindFarmModels.html \
            for more details.
        """  # noqa: E501
        # Validate inputs
        if 'time' in sim_kw:
            msg = f'{self.name}: Time simulation are not supported yet.'
            logger.error(msg)
            raise NotImplementedError(msg)
        if not isinstance(ti, float):
            msg = f'{self.name}: TI must a single real value.'
            logger.error(msg)
            raise ValueError(msg)

        # Prepare inputs
        # TODO: Define behavior for time sim
        _, _, stamps = self._as_conditions(ws, wd, sim_kw.get(PWDim.TIME))
        op_state = as_op_state(sim_kw.get(PWTag.OP_STATE), len(x), stamps)
        sim_kw = {**sim_kw, PWTag.OP_STATE: op_state}  # NOTE: Don't mutate the original

        # Run the fit method
        self._fit(x, y, ws, wd, ti, **sim_kw)
        self._is_fitted = True
        return self

    def predict(
        self,
        op_state: Iterable | None = None,
        ws: float | Iterable | None = None,
        wd: float | Iterable | None = None,
        time: bool | Iterable = False,
        **kwargs: Any
    ) -> xr.DataArray:
        """
        Given the turbine operating states and wind conditions, predict the \
        farm power output.

        Parameters
        ----------
        op_state: Iterable | None
            Turbine operating states. Assumes that all turbines are in operation \
            if None is passed.
        wd: float | Iterable | None
            Wind direction(s). Uses the full grid used for fitting if None is passed.
        ws: float | Iterable | None
            Wind speed(s). Uses the full grid used for fitting if None is passed.
        time: bool | Iterable | None
            If False (default), the simulation will be computed for the full wd x ws matrix
            If True, the wd and ws will be considered as a time series of flow conditions with time stamp 0,1,..,n
            If array_like: same as True, but the time array is used as flow case time stamp.

        Returns
        -------
            xr.DataArray
        """  # noqa: E501
        # Verify is fitted
        self._check_fitted()
        # Validate inputs
        if kwargs:
            msg = f'{self.name}: Simulation options must be passed to fit().'
            logger.error(msg)
            raise TypeError(msg)
        # TODO: Allow for time sim
        if time is not False and time is not None:
            msg = f'{self.name}: Time simulation are not supported yet.'
            logger.error(msg)
            raise NotImplementedError(msg)

        ws, wd, stamps = self._as_conditions(ws, wd, time)
        op_state = as_op_state(op_state, self.nturb, stamps)
        return self._predict(ws, wd, op_state)

    # Abstract methods
    # To be implement by childs
    @abc.abstractmethod
    def _fit(
        self,
        x: Iterable,
        y: Iterable,
        ws: float | Iterable,
        wd: float | Iterable,
        ti: float = 0.1,
        **sim_kw: Any,
    ) -> None:
        """
        Given the turbine locations and wind conditions, run the internal fit.

        Parameters
        ----------
        x : array_like
            Wind turbine x positions
        y : array_like
            Wind turbine y positions
        wd : int or array_like
            Wind direction(s)
        ws : int, float or array_like
            Wind speed(s)
        ti: float
            Turbulence intensity.
        **kwargs: Any
            Other argument to be passed to the wfarm.__call__ method. \
            Refer to https://topfarm.pages.windenergy.dtu.dk/PyWake/notebooks/EngineeringWindFarmModels.html \
            for more details.
        """  # noqa: E501
        raise NotImplementedError('This is an abstract class!!!')

    @abc.abstractmethod
    def _predict(
        self,
        ws: np.ndarray,
        wd: np.ndarray,
        op_state: np.ndarray
    ) -> xr.DataArray:
        """
        Given the turbine operating states and wind conditions, predict the \
        farm power output and return.

        Parameters
        ----------
        op_state: Iterable | None
            Turbine operating states. Assumes that all turbines are in operation \
            if None is passed.
        wd: float | Iterable | None
            Wind direction(s). Uses the full grid used for fitting if None is passed.
        ws: float | Iterable | None
            Wind speed(s). Uses the full grid used for fitting if None is passed.

        Returns
        -------
            np.ndarray
        """
        raise NotImplementedError('This is an abstract class!!!')


# MODELS
class BruteForceMask(MaskedArrayModel):
    """
    A masked array model that apply estimate the wind farm power output \
    from a full farm simulation with offline turbine (brute force approach).

    Parameters
    ----------
    wfarm: EngineeringWindFarmModel
        Instance of PyWake's engineering model to run farm flow simulation.

    Attributes
    ----------
    wfarm: EngineeringWindFarmModel
        Instance of PyWake's engineering model to run farm flow simulation.
    """

    # Init
    def __init__(self, wfarm: EngineeringWindFarmModel) -> None:
        """Create class instance."""
        super().__init__(wfarm)

        # Additional internal w.r.t to abstract
        self._ti: float | None = None
        self._sim_kw: dict[str, Any] | None = None

    # Method
    def _fit(
        self,
        x: Iterable,
        y: Iterable,
        ws: float | Iterable,
        wd: float | Iterable,
        ti: float = 0.1,
        **sim_kw: Any,
    ) -> None:
        """
        Given the turbine locations and wind conditions, run the internal fit.

        Parameters
        ----------
        x : array_like
            Wind turbine x positions
        y : array_like
            Wind turbine y positions
        wd : int or array_like
            Wind direction(s)
        ws : int, float or array_like
            Wind speed(s)
        ti: float
            Turbulence intensity.
        **kwargs: Any
            Other argument to be passed to the wfarm.__call__ method. \
            Refer to https://topfarm.pages.windenergy.dtu.dk/PyWake/notebooks/EngineeringWindFarmModels.html \
            for more details.
        """  # noqa: E501
        # Brute force is greedy. Store the simulation here, run in predict.
        self._ti = ti
        self._sim_kw = {k: v for k, v in sim_kw.items() if k != PWTag.OP_STATE}
        self._x, self._y = x, y
        self._ws, self._wd = ws, wd
        self._is_fitted = True  # NOTE: That's an easy fit, isn't it? :)

    def _predict(
        self,
        ws: np.ndarray,
        wd: np.ndarray,
        op_state: np.ndarray
    ) -> xr.DataArray:
        """
        Given the turbine operating states and wind conditions, predict the \
        farm power output and return.

        Parameters
        ----------
        op_state: Iterable | None
            Turbine operating states. Assumes that all turbines are in operation \
            if None is passed.
        wd: float | Iterable | None
            Wind direction(s). Uses the full grid used for fitting if None is passed.
        ws: float | Iterable | None
            Wind speed(s). Uses the full grid used for fitting if None is passed.

        Returns
        -------
            xr.DataArray
        """
        res = self.wfarm(
            self._x, self._y,
            ws=ws, wd=wd, TI=self._ti,
            **{PWTag.OP_STATE: op_state},
            **self._sim_kw
        )
        return res[PWTag.POWER].sum(PWDim.WT)


class StaticMask(MaskedArrayModel):
    """
    A masked array model that apply estimate the wind farm power output \
    of the masked array using a static mask.



    Parameters
    ----------
    wfarm: EngineeringWindFarmModel
        Instance of PyWake's engineering model to run farm flow simulation.

    Attributes
    ----------
    wfarm: EngineeringWindFarmModel
        Instance of PyWake's engineering model to run farm flow simulation.
    """

    # Method
    def _fit(
        self,
        x: Iterable,
        y: Iterable,
        ws: float | Iterable,
        wd: float | Iterable,
        ti: float = 0.1,
        **sim_kw: Any,
    ) -> None:
        """
        Given the turbine locations and wind conditions, run the internal fit.

        Parameters
        ----------
        x : array_like
            Wind turbine x positions
        y : array_like
            Wind turbine y positions
        wd : int or array_like
            Wind direction(s)
        ws : int, float or array_like
            Wind speed(s)
        ti: float
            Turbulence intensity.
        **kwargs: Any
            Other argument to be passed to the wfarm.__call__ method. \
            Refer to https://topfarm.pages.windenergy.dtu.dk/PyWake/notebooks/EngineeringWindFarmModels.html \
            for more details.
        """  # noqa: E501
        if PWDim.TIME in sim_kw:
            msg = f'{self.name}: Fit must be call on a fixed grid of wind conditions.'
            logger.error(msg)
            raise ValueError(msg)
        if not all(sim_kw.get(PWTag.OP_STATE)):
            msg = f'{self.name}: Fit requires that are turbines are online. If some turbines' \
                  'are permanently offline, remove them from the simulation.'
            logger.error(msg)
            raise ValueError(msg)
        sim_kw = {**sim_kw, **{PWDim.WS: ws, PWDim.WD: wd, PWTag.TI: ti}}
        sim_res = self.wfarm(x, y, **sim_kw)

        # Assign internal
        # NOTE: Drop the annoying SimulationResult type
        self.base_res = xr.Dataset({**sim_res})
        self._x, self._y = x, y
        self._ws, self._wd = ws, wd
        self._is_fitted = True

    def _predict(
        self,
        ws: np.ndarray,
        wd: np.ndarray,
        op_state: np.ndarray
    ) -> xr.DataArray:
        """
        Given the turbine operating states and wind conditions, predict the \
        farm power output and return.

        Parameters
        ----------
        op_state: Iterable | None
            Turbine operating states. Assumes that all turbines are in operation \
            if None is passed.
        wd: float | Iterable | None
            Wind direction(s). Uses the full grid used for fitting if None is passed.
        ws: float | Iterable | None
            Wind speed(s). Uses the full grid used for fitting if None is passed.

        Returns
        -------
            xr.DataArray
        """
        p = (
            self.base_res
            [PWTag.POWER]
            .sum(PWDim.WT)
            .sel({
                PWDim.WS: ws,
                PWDim.WD: wd,
            })
        )
        return (p * op_state.mean()).rename(PWTag.POWER.value)


class LeaveOneOutMask(MaskedArrayModel):
    """
    A masked array model that apply estimate the wind farm power output \
    applying wind speed corrected derived from leave-one-out simulation.



    Parameters
    ----------
    wfarm: EngineeringWindFarmModel
        Instance of PyWake's engineering model to run farm flow simulation.

    Attributes
    ----------
    wfarm: EngineeringWindFarmModel
        Instance of PyWake's engineering model to run farm flow simulation.
    """

    # Init
    def __init__(self, wfarm: EngineeringWindFarmModel) -> None:
        """Create class instance."""
        super().__init__(wfarm)

        # Additional internal w.r.t to abstract
        self._ws0_ilk: np.ndarray | None = None
        self._dws_ijlk: np.ndarray | None = None

    # Method
    def _fit(
        self,
        x: Iterable,
        y: Iterable,
        ws: float | Iterable,
        wd: float | Iterable,
        ti: float = 0.1,
        **sim_kw: Any,
    ) -> None:
        """
        Given the turbine locations and wind conditions, run the internal fit.

        Parameters
        ----------
        x : array_like
            Wind turbine x positions
        y : array_like
            Wind turbine y positions
        wd : int or array_like
            Wind direction(s)
        ws : int, float or array_like
            Wind speed(s)
        ti: float
            Turbulence intensity.
        **kwargs: Any
            Other argument to be passed to the wfarm.__call__ method. \
            Refer to https://topfarm.pages.windenergy.dtu.dk/PyWake/notebooks/EngineeringWindFarmModels.html \
            for more details.
        """  # noqa: E501
        if PWDim.TIME in sim_kw:
            msg = f'{self.name}: Fit must be called on a fixed grid of wind conditions.'
            logger.error(msg)
            raise ValueError(msg)
        # Run the simulation
        if not all(sim_kw.get(PWTag.OP_STATE)):
            msg = f'{self.name}: Fit requires that are turbines are online. If some turbines' \
                  'are permanently offline, remove them from the simulation.'
            logger.error(msg)
            raise ValueError(msg)

        # Baseline simulation (no offline)
        logger.info(f'{self.name}: Run baseline simulation without offline turbine.')
        sim_kw = {**sim_kw, **{PWDim.WS: ws, PWDim.WD: wd, PWTag.TI: ti}}
        dim_order = (PWDim.WT, PWDim.WD, PWDim.WS)  # NOTE: So that we do not rely on default
        sim_res = self.wfarm(x, y, **sim_kw).transpose(*dim_order)
        ws0_ilk = sim_res[PWTag.WS_EFF].data

        # LOO simulation
        logger.info(f'{self.name}: Run LOO simulations')
        nwt = len(x)
        masks = (np.ones((nwt, nwt)) - np.eye(nwt)).astype(bool)
        dws_ijlk = np.full((nwt, nwt, len(wd), len(ws)), np.nan)

        for wt_j in range(nwt):
            logger.info(
                f'{self.name}: Run simulation with masked on for turbine '
                f'{wt_j + 1:3d} (of {nwt:3d})'
            )
            res_k = (
                self.wfarm(x, y, **{**sim_kw, PWTag.OP_STATE: masks[:, wt_j]})
                .transpose(*dim_order)
            )
            dws_ijlk[:, wt_j, :, :] = res_k[PWTag.WS_EFF].data - ws0_ilk

        # Assign internal
        # NOTE: Drop the annoying SimulationResult type
        self.base_res = xr.Dataset({**sim_res})
        self._x, self._y = x, y
        self._ws, self._wd = ws, wd
        self._ws0_ilk = ws0_ilk
        self._dws_ijlk = dws_ijlk
        self._is_fitted = True

    def _predict(
        self,
        ws: np.ndarray,
        wd: np.ndarray,
        op_state: np.ndarray
    ) -> xr.DataArray:
        """
        Given the turbine operating states and wind conditions, predict the \
        farm power output and return.

        Parameters
        ----------
        op_state: Iterable | None
            Turbine operating states. Assumes that all turbines are in operation \
            if None is passed.
        wd: float | Iterable | None
            Wind direction(s). Uses the full grid used for fitting if None is passed.
        ws: float | Iterable | None
            Wind speed(s). Uses the full grid used for fitting if None is passed.

        Returns
        -------
            xr.DataArray
        """
        # NOTE: Are things to be tested ?
        # NOTE: Probably for time-simulation
        dw_ilk = self._dws_ijlk[:, ~op_state, :, :].sum(axis=1)
        ws_ilk = self._ws0_ilk + dw_ilk
        np.clip(ws_ilk, 0, self._ws[None, None, :], out=ws_ilk)
        pw_ilk = (
            self.wfarm.windTurbines
            .power(ws=ws_ilk, **{PWTag.OP_STATE: op_state})
            .sum(axis=0)
        )

        is_time = isinstance(ws, xr.DataArray)
        out = xr.DataArray(
            pw_ilk,
            name=PWTag.POWER.value,
            dims=[PWDim.TIME] if is_time else [PWDim.WD, PWDim.WS],
            coords={
                PWDim.WS: ws if is_time else (PWDim.WS, ws),
                PWDim.WD: wd if is_time else (PWDim.WD, wd),
                **({PWDim.TIME: ws[PWDim.TIME]} if is_time else {})
            }
        )
        return out.sel({PWDim.WS: ws, PWDim.WD: wd})

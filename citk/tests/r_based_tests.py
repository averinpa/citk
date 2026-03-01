from typing import List, Optional

import numpy as np
from causallearn.utils.cit import NO_SPECIFIED_PARAMETERS_MSG, register_ci_test

from .base import CITKTest


def _load_rcit_package():
    """
    Lazy-load rpy2 + RCIT package and raise a clear actionable error if missing.
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.packages import importr
    except ModuleNotFoundError as exc:
        raise ImportError(
            "R-based CI tests require optional dependency 'rpy2'. "
            "Install with: pip install 'citk[r]' (or uv sync --extra r)."
        ) from exc

    numpy2ri.activate()
    try:
        rcit_pkg = importr("RCIT")
    except Exception as exc:
        raise ImportError(
            "R package 'RCIT' is required for RCoT/RCIT tests. "
            "Install in R from GitHub: ericstrobl/RCIT."
        ) from exc

    return ro, rcit_pkg


def _to_r_vector(ro, arr: np.ndarray):
    return ro.FloatVector(np.asarray(arr, dtype=float).ravel())


def _to_r_matrix(ro, arr: np.ndarray):
    arr = np.asarray(arr, dtype=float)
    return ro.r.matrix(ro.FloatVector(arr.ravel()), nrow=arr.shape[0], ncol=arr.shape[1])


def _extract_p_value(result) -> float:
    try:
        return float(result.rx2("p.value")[0])
    except Exception as exc:
        raise RuntimeError("Could not extract 'p.value' from RCIT result.") from exc


class _RCITBase(CITKTest):
    supported_dtypes = {"continuous"}
    method_name = ""
    rcit_func_name = ""

    def __init__(self, data: np.ndarray, **kwargs):
        super().__init__(data, **kwargs)
        self.check_cache_method_consistent(self.method_name, NO_SPECIFIED_PARAMETERS_MSG)

    def _compute(self, X: int, Y: int, condition_set: Optional[List[int]] = None, **kwargs) -> float:
        ro, rcit_pkg = _load_rcit_package()

        x = _to_r_vector(ro, self.data[:, X])
        y = _to_r_vector(ro, self.data[:, Y])
        if condition_set:
            z = _to_r_matrix(ro, self.data[:, condition_set])
            result = getattr(rcit_pkg, self.rcit_func_name)(x, y, z)
        else:
            result = getattr(rcit_pkg, self.rcit_func_name)(x, y)

        return _extract_p_value(result)


class RCoT(_RCITBase):
    method_name = "rcot"
    rcit_func_name = "RCoT"


class RCIT(_RCITBase):
    method_name = "rcit"
    rcit_func_name = "RCIT"


register_ci_test("rcot", RCoT)
register_ci_test("rcit", RCIT)

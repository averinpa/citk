from typing import List, Optional

import numpy as np
import pandas as pd
from causallearn.utils.cit import (
    Chisq_or_Gsq,
    NO_SPECIFIED_PARAMETERS_MSG,
    register_ci_test,
)

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


def _load_bnlearn_package():
    try:
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Hartemink CI test requires optional dependency 'rpy2'. "
            "Install with: pip install 'citk[r]' (or uv sync --extra r)."
        ) from exc

    pandas2ri.activate()
    try:
        bnlearn_pkg = importr("bnlearn")
    except Exception as exc:
        raise ImportError(
            "R package 'bnlearn' is required for Hartemink discretization. "
            "Install from CRAN in your R environment."
        ) from exc
    return pandas2ri, bnlearn_pkg


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


class RKCIT(_RCITBase):
    method_name = "kci"
    rcit_func_name = "KCIT"

    def __init__(self, data: np.ndarray, **kwargs):
        if data.shape[0] > 2000:
            raise ValueError(
                "R KCIT wrapper is capped at n=2000 samples for stability/performance."
            )
        super().__init__(data, **kwargs)


class HarteminkChiSq(CITKTest):
    supported_dtypes = {"continuous", "discrete"}

    def __init__(self, data: np.ndarray, **kwargs):
        self.breaks = kwargs.get("breaks", 4)
        self.ibreaks = kwargs.get("ibreaks", 10)
        discretized = self._hartemink_discretize(data)
        super().__init__(discretized, **kwargs)
        params = f"breaks={self.breaks},ibreaks={self.ibreaks}"
        self.check_cache_method_consistent("hartemink_chisq", params)
        self.test_instance = Chisq_or_Gsq(self.data, method_name="chisq", **kwargs)

    def _hartemink_discretize(self, data: np.ndarray) -> np.ndarray:
        pandas2ri, bnlearn_pkg = _load_bnlearn_package()
        frame = pd.DataFrame(data, columns=[f"v{i}" for i in range(data.shape[1])])
        r_frame = pandas2ri.py2rpy(frame)
        r_disc = bnlearn_pkg.discretize(
            r_frame,
            method="hartemink",
            breaks=self.breaks,
            ibreaks=self.ibreaks,
        )
        disc_df = pandas2ri.rpy2py(r_disc)
        if not isinstance(disc_df, pd.DataFrame):
            disc_df = pd.DataFrame(disc_df)
        out = np.zeros((len(disc_df), disc_df.shape[1]), dtype=int)
        for j, col in enumerate(disc_df.columns):
            out[:, j] = pd.Categorical(disc_df[col]).codes
        return out

    def _compute(self, X: int, Y: int, condition_set: Optional[List[int]] = None, **kwargs) -> float:
        return float(self.test_instance(X, Y, condition_set))


register_ci_test("rcot", RCoT)
register_ci_test("rcit", RCIT)
register_ci_test("kci", RKCIT)
register_ci_test("hartemink_chisq", HarteminkChiSq)

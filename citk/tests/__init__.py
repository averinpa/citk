from .simple_tests import FisherZ, GSq, ChiSq, Spearman
from .statistical_model_tests import Regression, Logit, Poisson
from .extended_tests import DiscChiSq, DiscGSq, DummyFisherZ, GCMLinear, GCMRF, WGCMRF

try:
    from .ml_based_tests import KCI, RandomForest, DML, CRIT, EDML
except ModuleNotFoundError:
    KCI = RandomForest = DML = CRIT = EDML = None

try:
    from .r_based_tests import RCoT, RCIT
except ModuleNotFoundError:
    RCoT = RCIT = None

__all__ = [
    "FisherZ", "GSq", "ChiSq", "Spearman",
    "Regression", "Logit", "Poisson",
    "DiscChiSq", "DiscGSq", "DummyFisherZ",
    "GCMLinear", "GCMRF", "WGCMRF",
]

if KCI is not None:
    __all__.extend(["KCI", "RandomForest", "DML", "CRIT", "EDML"])

if RCoT is not None:
    __all__.extend(["RCoT", "RCIT"])

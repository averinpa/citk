from .simple_tests import FisherZ, GSq, ChiSq, Spearman
from .extended_tests import DiscChiSq, DiscGSq, DummyFisherZ, GCMLinear, GCMRF, WGCMRF
from .tigramite_based_tests import CMIknn, CMIknnMixed, RegressionCI

try:
    from .ml_based_tests import KCI, RandomForest, DML, CRIT, EDML
except ModuleNotFoundError:
    KCI = RandomForest = DML = CRIT = EDML = None

try:
    from .r_based_tests import RCoT, RCIT, HarteminkChiSq
except ModuleNotFoundError:
    RCoT = RCIT = HarteminkChiSq = None

__all__ = [
    "FisherZ", "GSq", "ChiSq", "Spearman",
    "DiscChiSq", "DiscGSq", "DummyFisherZ",
    "GCMLinear", "GCMRF", "WGCMRF",
    "CMIknn", "CMIknnMixed", "RegressionCI",
]

if KCI is not None:
    __all__.extend(["KCI", "RandomForest", "DML", "CRIT", "EDML"])

if RCoT is not None:
    __all__.extend(["RCoT", "RCIT", "HarteminkChiSq"])

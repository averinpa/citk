import codecs
import json
import os
from causallearn.utils.cit import CIT_Base

class CITKTest(CIT_Base):
    """
    Abstract base class for all conditional independence tests in the citk package.
    It standardizes the interface to be compatible with causal-learn and
    implements a robust file-based caching mechanism.
    """
    def __init__(self, data, **kwargs):
        """
        Parameters
        ----------
        data : np.ndarray
            The dataset from which to run the test.
        """
        # The parent __init__ from causallearn's CIT_Base handles cache loading.
        # We just need to ensure kwargs (containing cache_path) are passed up.
        super().__init__(data, **kwargs)

    def save_cache(self):
        """
        Explicitly saves the p-value cache to the file path provided during
        initialization. This is more reliable than relying on the garbage
        collector with __del__.
        """
        if hasattr(self, 'cache_path') and self.cache_path is not None:
            try:
                # The pvalue_cache is initialized in the parent CIT_Base class
                if hasattr(self, 'pvalue_cache'):
                    with codecs.open(self.cache_path, 'w') as fout:
                        fout.write(json.dumps(self.pvalue_cache, indent=2))
            except Exception as e:
                print(f"Error saving cache for {self.__class__.__name__}: {e}")

    def __call__(self, X, Y, condition_set=None, **kwargs):
        """
        Executes the conditional independence test.
        Subclasses must implement this method.

        Parameters
        ----------
        X : int
            The index of the first variable.
        Y : int
            The index of the second variable.
        condition_set : list[int], optional
            A list of indices for the conditioning set. Can be empty.

        Returns
        -------
        p_value : float
            The p-value of the test.
        """
        raise NotImplementedError("Subclasses must implement the __call__ method.") 
from pysindy import SR3
import numpy as np

class my_SR3(SR3):
    # Modified SR3 so it prints out all losses intead of some for each max_iter/10

    def _objective(self, x, y, q, coef_full, coef_sparse, trimming_array=None):
        """Objective function"""
        R2 = (y - np.dot(x, coef_full)) ** 2
        D2 = (coef_full - coef_sparse) ** 2
        if self.use_trimming:
            assert trimming_array is not None
            R2 *= trimming_array.reshape(x.shape[0], 1)
        regularization = self.reg(coef_full, self.reg_weight_lam)
        if self.verbose:
            row = [
                q,
                np.sum(R2),
                np.sum(D2) / self.relax_coeff_nu,
                regularization,
                np.sum(R2) + np.sum(D2) + regularization,
            ]
            print(
                "{0:10d} ... {1:10.4e} ... {2:10.4e} ... {3:10.4e}"
                " ... {4:10.4e}".format(*row)
            )
        return (
            0.5 * np.sum(R2)
            + 0.5 * regularization
            + 0.5 * np.sum(D2) / self.relax_coeff_nu
        )

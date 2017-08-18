import numpy as np
from scipy import sparse
import pandas as pd


class Signal(object):
    """Signal processing tools
    """
    @staticmethod
    def baseline_als(y, lamda, p, niter):
        """Asymmetric Least Squares Smoothing for Baseline fitting
        :param y: data
        :param lamda: smoothness
        :param p: assymetry
        :param niter: number of iteration
        :return:
        """
        L = len(y)
        D = sparse.csc_matrix(np.diff(np.eye(L), 2))
        w = np.ones(L)
        for i in xrange(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lamda * D.dot(D.transpose())
            z = sparse.linalg.spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        return pd.Series(z)

    @staticmethod
    def baseline_fitting(data, lamda, p, niter=10):
        """Apply baseline fitting
        """
        z = Signal.baseline_als(data, lamda, p, niter)
        z = z - z[0]
        output = data - z
        return pd.Series(output)

    @staticmethod
    def smoothing(data, niter=1):
        """Simple three-point smoothing
        :param data:
        :param niter:
        :return:
        """
        def osf(data):
            """Smoothing
            """
            output = list()
            for i, p in enumerate(list(data)):
                b = data[i]
                if i == 0:
                    a = b
                    c = data[i + 1]
                elif i == len(data) - 1:
                    a = data[i - 1]
                    c = b
                else:
                    a = data[i - 1]
                    c = data[i + 1]
                output.append(0.15 * min([a, b, c]) + 0.70 * np.median([a, b, c]) + 0.15 * max([a, b, c]))
            return output

        for i in range(niter):
            data = osf(data)
        return pd.Series(data)


# import statsmodels.api as sm
# class Postproc(object):
#     """ Post processing methods
#     """
#     @staticmethod
#     def statistic_threshold(path, p=0.05):
#         """This function works for excel files of the extracted timecourse using PyNIT
#         :param path:
#         :param p:
#         :return:
#         """
#         df = pd.read_excel(path)
#         rho = df.corr()
#         pval = rho.copy()
#         for j in range(df.shape[1]):
#             for k in range(df.shape[1]):
#                 try:
#                     df_ols = sm.OLS(df.iloc[:, j], df.iloc[:, k])
#                     pval.iloc[j, k] = df_ols.fit().pvalues[0]
#                 except ValueError:
#                     pval.iloc[j, k] = None
#         qval = sm.stats.fdrcorrection(pval.values.flatten())[1].reshape(pval.shape)
#         fisher = np.arctanh(rho)
#         # fisher = np.tril(fisher, -1)
#         fisher[qval > p] = 0
#         # fisher[fisher == 0] = np.nan
#         return pd.DataFrame(fisher, columns=rho.index, index=rho.index)
#
#     @staticmethod
#     def construct_groupaverage(panel, significant_rate=0.5):
#         """Construct groupaverage based on the significant rate across subjects
#         :param panel:
#         :param significant_rate:
#         :return:
#         """
#         zpanel = panel.copy()
#         zpanel = zpanel.fillna(0)
#         zpanel.values[zpanel.values != 0] = 1
#         grpavr = zpanel.mean(axis=0)
#         grpavr[grpavr < significant_rate] = 0
#         return panel.mean(axis=0)[grpavr != 0]
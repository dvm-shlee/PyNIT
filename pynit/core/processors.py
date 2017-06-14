from shutil import rmtree
from scipy import sparse
import statsmodels.api as sm
import os
import pandas as pd
from methods import np, read_table, objects
import copy
import methods


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


class Postproc(object):
    """ Post processing methods
    """
    @staticmethod
    def statistic_threshold(path, p=0.05):
        """This function works for excel files of the extracted timecourse using PyNIT

        :param path:
        :param p:
        :return:
        """
        df = pd.read_excel(path)
        rho = df.corr()
        pval = rho.copy()
        for j in range(df.shape[1]):
            for k in range(df.shape[1]):
                try:
                    df_ols = sm.OLS(df.iloc[:, j], df.iloc[:, k])
                    pval.iloc[j, k] = df_ols.fit().pvalues[0]
                except ValueError:
                    pval.iloc[j, k] = None
        qval = sm.stats.fdrcorrection(pval.values.flatten())[1].reshape(pval.shape)
        fisher = np.arctanh(rho)
        # fisher = np.tril(fisher, -1)
        fisher[qval > p] = 0
        # fisher[fisher == 0] = np.nan
        return pd.DataFrame(fisher, columns=rho.index, index=rho.index)

    @staticmethod
    def construct_groupaverage(panel, significant_rate=0.5):
        """Construct groupaverage based on the significant rate across subjects

        :param panel:
        :param significant_rate:
        :return:
        """
        zpanel = panel.copy()
        zpanel = zpanel.fillna(0)
        zpanel.values[zpanel.values != 0] = 1
        grpavr = zpanel.mean(axis=0)
        grpavr[grpavr < significant_rate] = 0
        return panel.mean(axis=0)[grpavr != 0]





class TempFile(object):
    """This class is designed to make Template Object can be utilized on Processing steps
    Due to the major processing steps using command line tools(AFNI, ANTs so on..), the numpy
    object cannot be used on these tools.

    Using this class, loaded ImageObj now has temporary files on the location at './.tmp' or './atlas_tmp'
    """
    def __init__(self, obj, filename='image_cache', atlas=False, flip=False, merge=False, bilateral=False):
        """Initiate instance

        :param obj:         ImageObj
        :param filename:    Temporary filename
        :param atlas:       True if the input is atlas data
        :param flip:        True if you want to flip
        :param merge:       True if you want to merge flipped ImageObj
        """
        # If given input object is atlas
        if atlas:
            self._image = None
            # Copy object to protect the intervention between object
            self._atlas = copy.copy(obj)
            if flip:
                self._atlas.extract('./.atlas_tmp', contra=True)
            if merge:
                self._atlas.extract('./.atlas_tmp', merge=True)
            if bilateral:
                self._atlas.extract('./.atlas_tmp')
                obj.extract('./.atlas_tmp', contra=True)
            else:
                self._atlas.extract('./.atlas_tmp')
            self._listdir = [ f for f in os.listdir('./.atlas_tmp') if '.nii' in f ]
            atlas = objects.Atlas('./.atlas_tmp')
            methods.mkdir('./.tmp')
            self._path = os.path.join('./.tmp', "{}.nii".format(filename))
            self._fname = filename
            atlas.save_as(os.path.join('./.tmp', filename), quiet=True)
            self._label = [roi for roi, color in atlas.label.values()][1:]
        else:
            # Copy object to protect the intervention between object
            self._image = copy.copy(obj)
            if flip:
                self._image.flip(invertx=True)
            if merge:
                self._image._dataobj += self._image._dataobj[::-1,]
            self._fname = filename
            methods.mkdir('./.tmp')
            self._image.save_as(os.path.join('./.tmp', filename), quiet=True)
            self._atlas = None
            self._label = None
            self._path = os.path.join('./.tmp', "{}.nii".format(filename))

    @property
    def path(self):
        return self._path

    @property
    def label(self):
        return self._label

    def __getitem__(self, idx):
        if self._image:
            raise IndexError
        else:
            if self._atlas:
                return os.path.abspath(os.path.join('.atlas_tmp', self._listdir[idx]))
            else:
                return None

    def __repr__(self):
        if self._image:
            return os.path.abspath(os.path.join('.tmp', "{}.nii".format(self._fname)))
        else:
            if self._atlas:
                output = []
                for i, roi in enumerate(self._listdir):
                    output.append('{:>3}  {:>100}'.format(i, os.path.abspath(os.path.join('.atlas_tmp', roi))))
                return str('\n'.join(output))

    def close(self):
        if self._image:
            os.remove(os.path.join('.tmp', "{}.nii".format(self._fname)))
        if self._atlas:
            rmtree('.atlas_tmp', ignore_errors=True)
            os.remove(os.path.join('.tmp', "{}.nii".format(self._fname)))
            os.remove(os.path.join('.tmp', "{}.label".format(self._fname)))
        self._atlas = None
        self._image = None
        self._path = None

import numpy as np
from pathlib import Path

from scipy.signal import savgol_filter

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score



def get_basis_and_projector(waveforms, n_components=1, n_iter=20):
    """
    Returns the basis vector A, subspace projector and svd of the waveforms.
    
    Remark: although in the single pulse case, A and the projector are simply the transpose of 
    each other, they are still assigned to two different variables, as their relationship is not as 
    straighforward for the multi-pulse case. The WaveformRegressor can thus handle both cases the 
    same way.
    """
    
    """ (i) Perform SVD"""
    svd = TruncatedSVD(n_components=25, n_iter=n_iter)
    svd.fit(waveforms)
    
    """ (ii) Construct projector """
#     The projector is defined as the pseudo inverse of the basis vectors A:
#     projector = np.linalg.pinv(A)
#     However, if the matrix A is orthonormal (which it is here!),then the pseudoinverse becomes:
    projector = svd.components_[:n_components]
    A = projector.transpose()
    return A, projector, svd


def construct_2PulseProjector(singlePulseBasis, delay=None, sampling=.125, method='pinv', **kwargs):
    """
    Gives the projector onto the subspace mapped by the chosen single-pulse SVD components for a two-pulse waveform
    Inputs:
        singlePulseBasis: single pulse basis A
        delay: delay between the two pulses
        nCoeff: number of single pulse basis vectors to take
        method: 'pinv', 'QR', 'Ridge'
    Returns:
        Basis matrix A and projector function
        The projector is given by:
            P=A.dot(projector).dot)data)
        The coefficients projector onto the subspace A are:
            coeffs=projector.dot(data)
        
        Note: if method 'Ridge' is used, then a RidgeRegressor object is returned instead of a projector (matrix).
        The function fitPulse will take care of handling this difference.
    """
    
    if delay is None:
        raise ValueError('Delay is None, give it a value!')
        
    
    """ (i) build the basis matrix """
    A0 = singlePulseBasis
    A1 = A0
    A2 = np.roll(A0,int(delay/sampling),axis=0)
    A = np.append(A1,A2,axis=1)
    
    """ (ii) Construct the projector """
    if method=='pinv':
        projector = np.linalg.pinv(A)
        return A, projector
    elif method=='QR':
        Q, R = np.linalg.qr(A)
        projector = np.transpose(np.linalg.inv(A.transpose().dot(Q))).dot(Q)
        return A, projector
    elif method=='Ridge':
        if 'alpha' in kwargs:
            alpha = kwargs.pop('alpha')
        else:
            alpha=0
        projector = Ridge(alpha=alpha, fit_intercept=False) # is not a projector per say
        return A, projector
    else:
        raise NameError('Method not implemented')
        

def construct_waveformRegressor(X_ref, n_components=1, mode='single', **kwargs):
    A, projector, svd = get_basis_and_projector(X_ref, n_components=n_components)
    if mode=='double':
        A, projector = construct_2PulseProjector(A, **kwargs)
    return WaveformRegressor(A=A, projector=projector, mode=mode)



class WaveformRegressor(BaseEstimator, RegressorMixin):
    """ Regressor compatible with sk-learn package """
    def __init__(self, A=None, projector=None, mode='single'):
        """
        A: Basis vectors of the subspace in matrix form (column vectors)
        projector: projector on the subspace A
        
        Construct basis A and projector using the function 'get_basis_projector' or 'construct_2PulseProjector'
        """
        self.A = A.T
        self.projector = projector.T # transpose to allow fit of many data at once
        self.mode = mode
    
    
    def fit(self, X, y=None):
        """ y=None for sklearn compatibility reason """
        # Check validity of input X:
#         if X.shape[0] != self.A.shape[0]:
#             self.coeffs_ = np.zeros(self.A.shape[1])
#             return self
        
        if isinstance(self.projector, Ridge):
            ridge = self.projector.fit(self.A, X)
            coeffs = ridge.coef_
        else:
#             coeffs = self.projector.dot(X.T)
            coeffs = X.dot(self.projector)
        
        self.coeffs_ = coeffs
        return self
    
    
    def reconstruct(self):
        try:
            getattr(self,"coeffs_")
        except AttributeError:
            raise RuntimeError("You must fit the waveform before reconstructing it!")
            
#         reconstructed = self.A.dot(self.coeffs_)
        reconstructed = self.coeffs_.dot(self.A)
        return reconstructed
    
    
    def predict(self, X=None):
        """ Just for sklearn compatibility reason """
        return self.reconstruct()
    
    
    def score(self, X):
        """ Returns the r2 score of the projected waveform """
        return r2_score(X, self.reconstruct())
        
    
    def fit_reconstruct(self, X, return_score=False):
        self.fit(X)
        if return_score:
            return self.reconstruct(), self.score(X)
        return self.reconstruct()
    
    
    def get_pulse_intensity(self, X):
        """
        Inputs:
            - wavefor X
        Ouputs:
            - norm(coeff)
            - max of reconstructed waveforms
        """
        self.fit(X)
        if self.mode=='single':
            p = np.linalg.norm(self.coeffs_)
            p_max = self.coeffs_.dot(self.A)
            p_max = p_max.max()
            return p, p_max
        elif self.mode=='double':
            nCoeff = int(self.coeffs_.shape[0]/2)
            coeffs_p1 = self.coeffs_[:nCoeff]
            coeffs_p2 = self.coeffs_[nCoeff:]
            p1 = np.linalg.norm(coeffs_p1)
            p2 = np.linalg.norm(coeffs_p2)
#             p1_max = self.A[:,:nCoeff].dot(coeffs_p1)
            p1_max = coeffs_p1.dot(self.A[:nCoeff,:])
#             p2_max = self.A[:,nCoeff:].dot(coeffs_p2)
            p2_max = coeffs_p2.dot(self.A[nCoeff:,:])
            p1_max = p1_max.max()
            p2_max = p2_max.max()
            return p1, p2, p1_max, p2_max
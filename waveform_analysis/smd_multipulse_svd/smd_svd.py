from pathlib import Path
import h5py as h5
import glob
import os
import numpy as np

import svd_waveform_processing as proc
import smalldata_tools.DetObject as dobj
from smalldata_tools.DetObject import DetObjectFunc


class svdFit(DetObjectFunc):
    """ Performs fit of waveforms using singular value decomposition. The main utility is to get the 
    intensity of multiple pulse in a single waveform, from a basis of single-pulse reference waveforms.
    Check the file make_waveform_basis.py for information on how to create the basis.
    See svd_waveform_processing.py for the underlying processing algorithm.
    """
    def __init__(self, **kwargs):
        """ Set delays and hyperparameters for the regressor
        Args:
            name (str): DetObjectName, default: svdFit
            n_pulse (int): number of pulses to fit. Default: 1
            delay (list or array): delay between each pulse
            sampling (float): sampling of the waveform. Default: 1
            basis_file (str or Path object): if not given will take the latest one in the detector calib folder
            mode (str): 'max', 'norm' or 'both', method to calculate the pulse amplitudes. Default: 'max'
            return_reconstructed (bool): return reconstructed waveforms or not
        """
        self._name = kwargs.get('name','svdFit')
        super(svdFit, self).__init__(**kwargs)
        self.n_pulse = kwargs.get('n_pulse',1)
        self.delay = kwargs.get('delay',[0])
        if isinstance(self.delay, int) or isinstance(self.delay, float):
            self.delay = [self.delay]
        self.sampling = kwargs.get('sampling', 1)
        self.basis_file = kwargs.get('basis_file', None)
        self._mode = kwargs.get('mode', 'max')
        self._return_reconstructed = kwargs.get('return_reconstructed', False)
            
    
    def setFromDet(self, det):
        """ Load basis, projector and other useful variables from calib h5 file """
        super(svdFit, self).setFromDet(det)
        calibDir = det.det.env.calibDir()
        if self.basis_file is None:
            try:
                # Automatically find the latest waveform basis file in calibDir
                calib_files = glob.glob('./wave_basis_'+det.det.alias+'*.h5') # dir for local test
#                 calib_files = glob.glob(calibDir+'/wave_basis_'+det.det.alias+'*.h5')
                self.basis_file = max(calib_files, key=os.path.getctime)
            except:
                pass
        if not Path(self.basis_file).is_file():
            print("No basis file found. Return 0")
            return 0
        else:
            print('{}: basis file found at {}'.format(self._name, self.basis_file))
        
        with h5.File(self.basis_file,'r') as f:
            A = f['A'][()]
            self.roi = f['roi'][()]
            if self.roi is None:
                self.roi=[0, 1e6]
            self.bkg_idx = f['background_index'][()]
                
        A, proj = proc.multiPulseProjector(
                A,
                n_pulse=self.n_pulse,
                delay=self.delay,
                sampling=self.sampling
            )
        self.regressor = proc.WaveformRegressor(A=A, projector=proj, n_pulse=self.n_pulse)
    
    
    def process(self, waveform):
        """
        Fit waveform and output dictionary with coefficient, intensity and score
        """
        if waveform.shape==(1,):
            waveform = waveform[None,:]
        if self.bkg_idx is not None:
            waveform = waveform - np.mean(waveform[:,:self.bkg_idx], axis=1)
        waveform = waveform[:,self.roi[0]:self.roi[1]]
        intensities = self.regressor.get_pulse_intensity(waveform, mode=self._mode)
        score = self.regressor.score(waveform)
        output = {
            'intensities': intensities,
            'score': score,
            'coefficients': self.regressor.coeffs_
        }
        if self._return_reconstructed:
            output['reconstructed'] = self.regressor.reconstruct()
        return output
    
    
    def process_with_alignment(self, waveform):
        """ To implement
        """
        return None
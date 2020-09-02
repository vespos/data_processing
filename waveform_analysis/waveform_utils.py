import numpy as np


def removeBackground(signal, med_window=[0,50]):
    """ Set the baseline of the waveform to 0 be removing a median
    Args:
        signal: input waveform
        med_window: [idx1, idx2] range over which the median is taken
    """
    if signal.ndim ==1:
        return signal - np.median(signal[med_window[0]:mede_window[1]])
    elif signal.ndim==2:
        med = np.median(signal[:,med_window[0]:med_window[1]], axis=1)  
        return signal - med.reshape(med.shape[0],1)
    
    
def alignWaveforms(waveforms):
    return
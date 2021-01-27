import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit



width_percentile = (15,80)


def pol1(x, intercept, m):
    return (intercept + x*m)


def gauss(x, mean, sigma, height=1.):
    if sigma==0:
        return 1e32
    return (height* np.exp(-0.5 * ((x-mean)/sigma)**2))


def lorentzian(x, x0, gam, a=1):
    return a * gam**2 / ( gam**2 + ( x - x0 )**2)


def moments(x,y=None):
    """
    Args:
        x: values
        y: weights
    """
    if y is None:
        y = np.ones_like(x)
    mean = np.sum(x*y)/np.sum(y)
    variance = np.sum( (x-mean)**2*y )/y.sum()
    sigma = np.sqrt(variance)
    skew = np.sum( (x-mean)**3*y ) /sigma**3
    return mean, sigma, skew


def get_residuals(var1, var2, fit, relative_residual=False):
    if relative_residual:
        residuals = (var2-fit)/var2
    else:
        residuals = (var2-fit)
    return residuals


def get_med_width(x):
    med = np.median(x)
    width = np.percentile(x, width_percentile[1])-np.percentile(x, width_percentile[0])
    thisresDim = (-3*width, 3*width)
    filt = np.logical_and(x>thisresDim[0], x<thisresDim[1])
    width = np.percentile(x[filt], width_percentile[1])-np.percentile(x[filt], width_percentile[0])
    return med, width


def linear_fit(var1, var2, percentile=[1,99]):
    # initial parameters for pol1 & fit
    var1_low = np.nanpercentile(var1, percentile[0])
    var1_high = np.nanpercentile(var1, percentile[1])
    d_var1 = var1_high - var1_low
    
    var2_low_vals = var2[np.argwhere( ((var1_low-d_var1*0.05) < var1) & (var1 < (var1_low+d_var1*0.05)) )]
    var2_high_vals = var2[np.argwhere( ((var1_high-d_var1*0.05) > var1) & (var1 < (var1_high+d_var1*0.05)) )]
    var2_low = var2_low_vals.mean()
    var2_high = var2_high_vals.mean()
    d_var2 = var2_high_vals.mean()-var2_low_vals.mean()
    
    par_est_pol1 = (var1_low-d_var2/d_var1*var2_low), d_var2/d_var1 # offset and slope guesses
    p1_params, p1_cov = curve_fit(pol1, var1, var2, par_est_pol1)
    
    return p1_params, p1_cov


def residual_fit(residuals, nBins=50, fct='gauss'):
    if fct=='gauss':
        fun = gauss
    elif fct=='lorentz':
        fun = lorentzian
    else:
        print('functions not implemented')
        return
    # estimate gauss parameters
    resMed, resWidth = get_med_width(residuals)
#     resMed, resWidth, skew = moments(residuals) # with moments
    thisresDim = (-3*resWidth, 3*resWidth)
    
    # residual histogram
    resHis = np.histogram(residuals, np.linspace(thisresDim[0], thisresDim[1], nBins))
    
    xVals = 0.5*(resHis[1][1:]+resHis[1][:-1])
    yVals = resHis[0]
    param_estimate = resMed, resWidth, max(yVals)
    
    try:
        fit_params, fit_cov = curve_fit(fun, xVals, resHis[0], param_estimate)
    except:
        fit_params, fit_cov = ([np.nan,np.nan,np.nan], np.nan)
    return fit_params, fit_cov
    

def correlation_fit(var1, var2, percentile=[1,99], relative_residual=False, nBins=50, fct='gauss'):
    """
    Linear fit to var1, var2 and gaussian fit on the var2-fit residuals.
    """
    # linear fit
    p1_params, p1_cov = linear_fit(var1, var2, percentile=percentile)
    fit_p1 = pol1(var1, p1_params[0], p1_params[1] )
    
    # gaussian fit on residuals
    residuals = get_residuals(var1, var2, fit_p1, relative_residual=relative_residual)
    fit_params, fit_cov = residual_fit(residuals, nBins, fct=fct)
    
    # summary dict
    out = {
        'correlation_offset': p1_params[0],
        'correlation_slope': p1_params[1],
        'residual_center': fit_params[0],
        'residual_width': fit_params[1],
        'residual_amp': fit_params[2],
        'fct': fct,
        'relative_residual': relative_residual
    }
    return out
    
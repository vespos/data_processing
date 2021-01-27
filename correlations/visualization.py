import numpy as np
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
# import holoviews as hv
# hv.extension('bokeh')

import correlations as corr



def plot_lin_fit(var1, var2, fit_res, labels=[None,None], ax=None, use_hist=False):
    if ax is None:
        fig, ax = plt.subplots()
    
    offset = fit_res['correlation_offset']
    slope = fit_res['correlation_slope']
    plot_var1 = np.linspace(np.nanmin(var1), np.nanmax(var1), 5)
    fit_p1 = corr.pol1(plot_var1, offset, slope)
    
    fit_label = '{}: {:.5f}*({}) + {:.5f}'.format(labels[1], slope, \
                                              labels[0], offset)
    
    if use_hist==False:
        ax.plot(var1, var2, '.', markersize=0.7, color='orange')
    else:
        bins = [np.linspace(np.percentile(var1,1), 1.2*np.percentile(var1,99),50), 
                np.linspace(np.percentile(var2,0.1), 1.2*np.percentile(var2,99),50)]
        ax.hist2d(var1, var2, bins=bins, cmap='magma')
    
    ax.plot(plot_var1, fit_p1, '-', color='purple', linewidth=1, label=fit_label) # plot fit
    ax.legend(loc='upper left')
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.grid(alpha=0.3)
    return ax


def plot_residual_fit(var1, var2, fit_res, ax=None, nBins=50):
    if ax is None:
        fig, ax = plt.subplots()
    
    offset = fit_res['correlation_offset']
    slope = fit_res['correlation_slope']
    center = fit_res['residual_center']
    width = fit_res['residual_width']
    amp = fit_res['residual_amp']
    if fit_res['fct']=='gauss':
        fun = corr.gauss
    elif fit_res['fct']=='lorentz':
        fun = corr.lorentzian
    else:
        return
    
    fit_p1 = corr.pol1(var1, offset, slope)
    
    residuals = corr.get_residuals(var1, var2, fit_p1, relative_residual=fit_res['relative_residual'])
    resMed, resWidth = corr.get_med_width(residuals)
    thisresDim = (-3*resWidth, 3*resWidth)
    resHis = np.histogram(residuals, np.linspace(thisresDim[0], thisresDim[1], nBins))
    xfit = np.linspace(thisresDim[0],thisresDim[1],100)
    yfit = fun(xfit, center, width, amp)
    
    ax.plot(0.5*(resHis[1][:-1]+resHis[1][1:]), resHis[0], 'o', color='orange')
    ax.plot(xfit, yfit, '-', color='purple', linewidth=1) # plot fit
    ax.set_xlabel('Residuals')
    ax.set_ylabel('#')
    ax.grid(alpha=0.3)
    return ax
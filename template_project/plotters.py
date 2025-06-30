import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

def plot_corr(corr_dict, plot_dict, fig=None, ax=None):
    """
    Plot the correlation between lags and correlations with dynamic plot configuration.
    
    Parameters:
        corr_dict (dict): Dictionary containing 'lags' and 'correlations' keys and 'labels'
                          'lags' represents the lags, 'correlations' represents the correlation values and 'labels' represents the labels for each set of lags and correlations.
        plot_dict (dict): Dictionary to specify plot parameters such as:
                          'title','ylim', 'xlim', 'xlabel', 'ylabel', 'max_lag', 'colors', 'yticks',etc.
    
    Returns:
        fig, ax: The matplotlib figure and axes objects for further customization.

    Example: 
    corr_dict = {'lags': [range(-24*3, 24*3+1)]	, 'correlations': [np.random.uniform(-1,1,145)]	}
    plot_dict = {
        'max_lag': 24*3,
        'xlabel': 'Time shift [days], ocean lags atmosphere',
        'ylabel': r'Correlation with w$_{rms}$',
        'ylim': (-1,1),
        'color': 'r',
        'title': 'Correlation vs Time Lag'
    }
    # Generate the plot
    fig, ax = plot_corr(corr_dict, plot_dict)
    plt.show()
    """

    
    # Set up figure and axis
    if fig is None or ax is None:
        fig, ax = plt.subplots(1,1, figsize=(11, 6))
    elif fig is not None and ax is None:
        ax = fig.add_subplot(111)

    # Extract lags and correlation values
    lag_sets            = corr_dict.get('lags', [])
    correlation_sets    = corr_dict.get('correlations', [])
    print(lag_sets, correlation_sets)
    
    # Extract labels and colors (use default if not provided)
    labels = corr_dict.get('labels', [f'Series {i+1}' for i in range(len(lag_sets))])
    colors = plot_dict.get('colors', ['r'] * len(lag_sets))  # Default to red if no colors provided
    marker = plot_dict.get('marker', '.')  # Default to circle marker if not provided
    
    # Plot each set of lags and correlations
    for i, (lags, correlations) in enumerate(zip(lag_sets, correlation_sets)):
        ax.plot(lags, correlations, color=colors[i], label=labels[i], marker=marker)
        ax.plot(lags, correlations, color=colors[i])
    ax.legend()

    # Draw horizontal and vertical reference lines at zero
    ax.axhline(0, color='k', linestyle='--')
    ax.axvline(0, color='k', linestyle='--')

    # Set axis labels (can be customized in plot_dict)
    xlabel = plot_dict.get('xlabel', 'Time shift [days], ocean lags atmosphere')
    ylabel = plot_dict.get('ylabel', r'Correlation with w$_{rms}$')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set custom x-ticks at every 24 hours and appropriate labels
    max_lag = plot_dict.get('max_lag', 24*3)        # Default to 3 days if not provided
    x_ticks = list(range(-max_lag, max_lag+1, 24))  # Generate ticks at every 24 hours
    x_labels = [f'{i//24}' if i == 24 else f'{i//24}' for i in x_ticks]
    ax.set_xticks(x_ticks, x_labels)

    # Set custom y-ticks and limits
    ax.set_yticks(plot_dict.get('yticks', [0,0.5]))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.set_ylim(plot_dict.get('ylim', [-0.2,0.8]))
    # Set custom x-limits
    ax.set_xlim(plot_dict.get('xlim',[-24,max_lag]))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5*24)) # Minor ticks at every 12 hours
    # Set optional title
    title = plot_dict.get('title', '')
    ax.set_title(title)
    return fig, ax


def plot_lag_correlation_steps(intermediates, ds, fontsize=15, mld_var='MLD', wprime_rms=None, series_2=None, ds_hp=None):
    """Plot steps of lag correlation 

    Parameters:
    ---------
    intermediates:      dict, optional
                        dictionary with intermediate results for debugging, contains:
                        - 'wprime_inner': wprime values calculated in the inner rolling window
                        - 'is_nan': boolean mask of NaN values in the vertical velocity variable
                        - 'wprime_in_mld': wprime values within the mixed layer
                        - 'is_below': boolean mask of data below the mixed layer depth
                        - 'time_leaves_mld': times when the glider leaves the mixed layer
                        - 'time_enters_mld': times when the glider enters the mixed layer
                        - 'gap_mask': boolean mask of gaps exceeding an hour in the data within the mixed layer
                        - 'wprime_outer': wprime values calculated in the outer rolling window
                        - 'wprime_interp_seg': wprime values interpolated onto resample_time
    ds :                xarray.Dataset
                        Input dataset with glider track data containing 'depth', 'time', 'w' and mld_var
    fontsize:           int
                        Fontsize of the labels
    mld_var:            str or list
                        plot variable in ds with this label(s) in the first row
    w_prime_rms:        xarray.Dataset
                        plot the values of this dataset over time in the 4th row (e.g. w_prime_rms if we do not consider gaps)
    series_2:           pandas Series
                        plot the values of this dataset vs its index (should be time) in the 4th row (e.g. output of an alternative gapping procedure)
    
    ds_hp:              xarray.Dataset
                        plot the mean over depth values of this dataset 

    """
    wprime_inner     = intermediates['wprime_inner']
    wprime_in_mld    = intermediates['wprime_in_mld']
    wprime2          = intermediates['wprime_outer']
    wprime_interp    = intermediates['wprime_interp_seg']
    gap_mask         = intermediates['gap_mask']
    is_below        = intermediates['is_below']
    is_nan2         = intermediates['is_nan']

    import matplotlib.pyplot as plt
    fig, (ax,ax2,ax3,ax4) = plt.subplots(4,1,figsize=(25, 18), sharex=True)

    # 1) Pressure vs depth of the glider, showing the masking of the MLD and the gaps
    ax.plot(ds.time[~is_nan2], ds.depth[~is_nan2], '.',label='Full dive', color='grey', alpha=0.5) #all depth
    ax.plot(ds.time[~is_nan2][is_below], ds.depth[~is_nan2][is_below], '.',label='dive shallower than MLD', color='k', alpha=0.5) #only depth shallower than MLD
    ax.plot(ds.time[~is_nan2][is_below][~gap_mask], ds.depth[~is_nan2][is_below][~gap_mask], '.', color='green', alpha=0.5, label='dive shallower than MLD & within ') #only depth shallower than MLD and not within 1h distance from gap
    if len(mld_var) > 1:
        for mld_vars in mld_var:
            ax.plot(ds.time, np.abs(ds[mld_vars]), label=mld_vars, alpha=0.5)
    else:
         ax.plot(ds.time, np.abs(ds[mld_var]), label=mld_var, color='blue', alpha=0.5)
    ax.invert_yaxis()
    ax.set_ylabel('Depth (m)', fontsize=fontsize)
    ax.legend(loc='upper right', fontsize=fontsize)

    # 2 ) w vs depth of the glider, showing the masking of the MLD and the gaps
    ax2.plot(ds.time[~is_nan2], ds.w[~is_nan2], '.', label='w', color='grey', alpha=0.5)
    ax2.plot(ds.time[~is_nan2][is_below], ds.w[~is_nan2][is_below], '.', label='w in MLD', color='black', alpha=0.5)
    ax2.plot(ds.time[~is_nan2][is_below][~gap_mask], ds.w[~is_nan2][is_below][~gap_mask], '.', label='w in MLD no gap', color='green', alpha=0.5)
    ax2.set_ylabel('w (m/s)', fontsize=fontsize)

    # 3) w' vs time, showing the masking of the MLD and the gaps
    ax3.plot(wprime_inner.time, wprime_inner, '.',label='w\' inner', color='grey', alpha=0.5)
    ax3.plot(wprime_in_mld.time, wprime_in_mld, '.',label='w\' inner in MLD', color='black', alpha=1)
    ax3.plot(wprime_in_mld.time[~gap_mask], wprime_in_mld[~gap_mask], '.',label='w\' inner in MLD no gap', color='green', alpha=1)
    ax3.set_ylabel('(w - <w>)(m/s)', fontsize=fontsize)

    # 4) w'rms vs time, showing interpolated values
    ax4.plot(wprime2.time, wprime2, '.', label='w\' outer', color='green', alpha=1, lw=1, zorder=9)
    ax4.plot(wprime_interp.time, wprime_interp.w_prime.values, 'D', label='w\' hourly interp', color='tab:olive', alpha=1, lw=1, zorder=10)
    if wprime_rms is not None:
        ax4.plot(wprime_rms.time, wprime_rms, '.', label='w\' without mask', color='grey', alpha=1, lw=1, zorder=1) #calculating w_prime from the beginning without any gaps (w-rolling1hw)rolling6h
    if ds_hp is not None:
        ax4.plot(ds_hp.mid_times, ds_hp.mean(dim='depth').w_rms, label='w Highpass 70m, full depth avg', color='blue', alpha=1, lw=3)
    # ax4.plot(w_hp_sel.mid_times, w_hp_sel.w_rms_MLD_avg, label='w Highpass 70m, MLD avg', color='orange', alpha=1, lw=3)
    if series_2 is not None:
        ax4.plot(series_2.index, series_2, 'X', label='w\' hourly interp - mask before interpolation', color='red', alpha=1, lw=1, zorder=2)
    ax4.legend(loc='upper right', fontsize=fontsize)
    ax4.set_ylabel(r'sqrt(<(w - <w>)$^2$>) (m/s)', fontsize=fontsize)

    return fig, (ax, ax2, ax3, ax4)


def plot_series_timelag(series_rms, series_heat, series_wind):

    fig, ax = plt.subplots(3,1, figsize=(20, 6), sharex=True)
    ax[0].plot(series_rms, color='grey', marker='.')
    ax[0].set_ylabel(r'$w_{rms}$'+'\n [m/s]')
    #ax[0].set_ylim(0,0.02)

    ax[1].plot(series_heat, 'r.-')
    ax[1].set_ylabel('Heat flux \n [W/m$^2$]')
    ax[1].set_ylim(0,1000)

    ax[2].plot(series_wind, 'k.-')
    ax[2].set_ylabel('Wind speed \n [m/s]')
    ax[2].set_ylim(0,35)

    #ax[2].set_xticks(pd.date_range(start=start_time, end=end_time, freq='7D'))
    ax[2].set_xlabel('Time')

    fig.align_ylabels(ax)
    return fig, ax

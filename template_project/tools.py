import numpy as np 
from scipy.stats import pearsonr, norm, probplot, jarque_bera, shapiro, kstest
import statsmodels.api as sm
import pandas as pd
import xarray as xr

def isel_time_range(data, start_time, end_time, time_dim=None):
    """ Selects a time range from an xarray Dataset or DataArray by selecting indices (uses integers instead of labels).
    Parameters:
    -----------
    data : xarray.Dataset or xarray.DataArray
        The input data containing a time dimension.
    start_time : str or pd.Timestamp
        The start time of the range to select.
    end_time : str or pd.Timestamp
        The end time of the range to select.
    time_dim : str, optional
        The name of the time dimension. If not provided, it will be automatically determined as the first data dimension.
    
    Returns:
    -----------
    xarray.Dataset or xarray.DataArray
        The data subsetted to the specified time range.
    """
    if time_dim is None:
        # Automatically determine the time dimension if not provided
        time_dim = list(data.dims)[0]
    return data.isel({time_dim : (data[time_dim] >= start_time) & (data[time_dim] <= end_time)}, drop=True)

def sel_time_range(data, start_time, end_time):
    """ Selects a time range from an xarray Dataset or DataArray by selecting index labels (uses labels instead of integergs & might be slower).
    Parameters:
    -----------
    data : xarray.Dataset or xarray.DataArray
        The input data containing a time dimension.
    start_time : str or pd.Timestamp
        The start time of the range to select.
    end_time : str or pd.Timestamp
        The end time of the range to select.
    Returns:
    -----------
    xarray.Dataset or xarray.DataArray
        The data subsetted to the specified time range.
    """
    return data.sel(time=slice(start_time, end_time), drop=True)

def filter_variables(data, variables_to_keep):
    """ Filters the variables in an xarray Dataset or DataArray to keep only the specified variables.
    Parameters:
    -----------
    data : xarray.Dataset or xarray.DataArray
        The input data containing multiple variables.
    variables_to_keep : list of str
        The list of variable names to keep in the data.
    Returns:
    -----------
    xarray.Dataset or xarray.DataArray
        The data with only the specified variables retained.
    """
    return data.drop_vars(set(data.data_vars) - set(variables_to_keep)).reset_coords(names=['longitude', 'latitude', 'depth'])

def convert_time(data, time_var='time'):
    """ Converts the time coordinate in an xarray Dataset or DataArray to a datetime64 format and adds a numeric time variable.
    Parameters:
    -----------
    data : xarray.Dataset or xarray.DataArray
        The input data containing a time dimension.
    time_var : str, optional
        The name of the time variable in the data. Default is 'time'.

    Returns:
    -----------
    xarray.Dataset or xarray.DataArray
        The data with the time coordinate converted to datetime64 and a numeric time variable added.
    """
    data[time_var]     = pd.to_datetime(data[time_var].values)
    data[time_var+'_var'] = (time_var, data[time_var].values.astype('int64')) #save time as numeric value
    return data

def compute_squared_velocity(data, var='w'):
    """ Computes the square of the specified velocity variable in an xarray Dataset or DataArray.
    Parameters:
    -----------
    data : xarray.Dataset or xarray.DataArray
        The input data containing the velocity variable.
    var : str, optional
        The name of the velocity variable to compute the square of. Default is 'w'.
    Returns:
    -----------
    xarray.Dataset or xarray.DataArray
        The data with an additional variable containing the square of the specified velocity variable.
    """
    data[var+'_2'] = ('time', data[var].values**2)
    return data

def filter_by_depth(data, depth_min):
    """ Filters the data in an xarray Dataset or DataArray to keep only the data below a specified depth minimum.
    To ensure that the depth is treated correctly, the absolute values of the depth are used.
    Parameters:
    -----------
    data : xarray.Dataset or xarray.DataArray
        The input data containing a depth dimension.
    depth_min : float
        The minimum depth to retain data below this threshold.
    Returns:
    -----------
    xarray.Dataset or xarray.DataArray
        The data subsetted to include only the data below the specified depth minimum.
    """
    #return data.where(np.abs(data.depth) >= np.abs(depth_min), drop=True) #filter out data above the specified depth minimum (taking absolute values only)
    time_dim = list(data.dims)[0]
    return data.isel({time_dim : np.abs(data.depth) >= np.abs(depth_min)}, drop=True)

def preprocess_data(config):
    """
    Preprocesses the input dataset by selecting a time range, filtering by depth, 
    and preparing data for interpolation and further analysis.

    Parameters:
    ----------- 
    config: dict
        Configuration dictionary containing preprocessing parameters:
        - 'start_time' (str or pd.Timestamp): Start time for the time selection.
        - 'end_time' (str or pd.Timestamp): End time for the time selection.
        - 'depth_min' (float): Minimum depth to retain data below a threshold.
        - 'data_in' (xarray.Dataset): Input dataset.

    Returns:
    -----------
    xarray.Dataset
        Preprocessed dataset ready for further analysis.
    """
    start_time          = config["start_time"]
    end_time            = config["end_time"]
    depth_min           = config["depth_min"]
    data                = config["data_in"]
    var                 = config.get("var", 'w') #default variable to compute squared velocity
    variables_to_keep   = config.get("variables_to_keep", ['w', 'longitude', 'latitude', 'depth', 'MLD', 'SA', 'CT', 'pressure', 'salinity','profile_index', var]) #default variables to keep
    
    data = sel_time_range(data, start_time, end_time)
    data = filter_variables(data, variables_to_keep)
    data = convert_time(data)
    print('filter depths shallower than:', depth_min, 'dbar')
    #data = compute_squared_velocity(data, var=var)
    data = filter_by_depth(data, depth_min)
    
    return data

### Get the mid times for all prfile indexes
def get_mid_profile_times(L1, profile_index_var='profile_index', time_var='time'):
    """
    Get the mid times for all profile indexes in a dataset.
    
    Parameters:
    L1 (xarray.Dataset): The dataset containing the profiles.
    L1398_manual (xarray.Dataset): The dataset containing the manual profile data.
    
    Returns:
    numpy.ndarray: An array of mid times for each profile index.
    """
    # Get unique profile indexes
    profile_indexs = np.unique(L1[profile_index_var].values) #PROFILE_NUMBER.values)
    mid_times = np.full(profile_indexs.size, np.datetime64('NaT'), dtype='datetime64[ns]')
    mid_profs = np.full(profile_indexs.size, np.nan)  # Store profile indexes for which mid times are found

    # Loop and store mid times
    for i, prof_ind in enumerate(profile_indexs):
        #print(f"Processing profile {i+1}/{len(profile_indexs)}: {prof_ind}")
        try:
            mid_times[i] = find_time_mid_profile(prof_ind, L1, profile_index_var=profile_index_var, time_var='time')
            mid_profs[i] = prof_ind
        except Exception as e:
            print(f"Skipping profile {prof_ind} due to error: {e}")
            continue
    return mid_times, mid_profs


def find_time_mid_profile(profile_index, ds, profile_index_var='profile_index', time_var='time', depth_var='depth', depth_min=20):
    # Get the dimension name over which profile_index varies (e.g., "N_MEASUREMENTS")
    dim_name = ds[profile_index_var].dims[0]

    # Find the index positions where the profile_index matches
    #matching_indices = np.where(ds[profile_index_var].values == profile_index)[0]

    #if len(matching_indices) == 0:
    #    raise ValueError(f"No entries found for profile_index {profile_index}")

    # Select using isel
    profile_ds = ds.isel({dim_name: ds[profile_index_var] == profile_index})

    #get rid of time the glider spend at the surface
    if depth_var in profile_ds:
        profile_ds = profile_ds.isel({dim_name: profile_ds[depth_var] > depth_min})  # Select only measurements deeper than depth_min
    else:
        print(f"Warning: {depth_var} not found in dataset. Skipping depth filtering.")

    times = profile_ds[time_var].values
    if times.size == 0:
        raise ValueError(f"No time data found for profile_index {profile_index}")

    # Calculate midpoint in time
    mid_time = (times.astype('int64').mean())
    return mid_time.astype('datetime64[ns]')


def interpolate_onto_track(ds, config):
    """
    Interpolates wind and heat flux data onto the glider track.

    Parameters:
    -----------
    ds : xarray.Dataset 
        Input dataset with glider track data containing 'latitude', 'longitude', and 'time'.
    config : dict
        Configuration dictionary containing interpolation parameters:
        - 'interp_method' (str): Interpolation method for interpolating atmospheric data on track.
        - 'data_wind' (xarray.Dataset): Wind data for interpolation containing 'u10' and 'v10' (hourly).
        - 'data_heat' (xarray.Dataset): Heat flux data for interpolation containing 'slhf', 'sshf', 'ssr', and 'str' (hourly).
        - 'start_time' (str or pd.Timestamp): Start time for the time selection.
        - 'end_time' (str or pd.Timestamp): End time for the time selection.
        - 'freq' (str): Frequency to interpolate the atmospheric data onto. Should not be lower than 1h for ERA5 reanalysis data.

    Returns:
    -----------
    resample_time   : pd.DatetimeIndex
        Regular time array with hourly resolution.
    series_win      : pd.Series 
        Series of wind speed values interpolated onto the glider track.
    series_heat     :   pd.Series 
        Series of heat flux values interpolated onto the glider track.
    Optional:
    if config['add_ext_interp'] is True:
        lon_points_interp (xarray.DataArray): Interpolated longitude values at resample time.
        lat_points_interp (xarray.DataArray): Interpolated latitude values at resample time.
        time_points_interp (xarray.DataArray): Interpolated time values at resample time.
    """
    # Extract values from the config dictionary
    interp_method                 = config['interp_method']
    data_wind                     = config['data_wind']
    data_heat                     = config['data_heat']

    # Create a regular time array with a given frequency between start and end time
    resample_time                 = pd.date_range(start=config['start_time'], end=config['end_time'], freq=config['freq']) # Create a regular time array with hourly resolution
    
    # interpolate data onto resample time linearly 
    # maybe it would make sense to use dropna with a subset so dropna(dim=time, subset=longitude,latitude), but maybe it is not necessary here
    ds_interp                     = ds.interp(time=resample_time, method=interp_method, kwargs={'fill_value': np.nan})#.dropna(dim='time')  #interpolate to the same time as wind data

    # Pointwise Interpolation onto the glider positions
    # to trigger the pointwise indexing, they need to create DataArray objects with the same dimension name, and then use them to index the DataArray. (https://tutorial.xarray.dev/intermediate/indexing/advanced-indexing.html)
    lat_points_interp             = xr.DataArray(ds_interp.latitude.values , dims="points")  # glider latitude at resample time
    lon_points_interp             = xr.DataArray(ds_interp.longitude.values, dims="points") # glider longitude at resample time
    time_points_interp            = xr.DataArray(resample_time             , dims="points")              # resample time as xarray object
    wind_points_interp            = data_wind.interp(longitude=lon_points_interp, latitude=lat_points_interp, time=time_points_interp, method=interp_method, kwargs={'fill_value': np.nan})
    heat_points_interp            = data_heat.interp(longitude=lon_points_interp, latitude=lat_points_interp, time=time_points_interp, method=interp_method, kwargs={'fill_value': np.nan})
    all_fluxes                    = -(heat_points_interp['slhf']   + heat_points_interp['sshf'] + heat_points_interp['ssr'] + heat_points_interp['str'])/3600 #J/m2 to W/m2 needs to be divided by 3600 as it is on a hourly resolution

    # Create pandas Series for wind and heat fluxes
    series_wind                   = pd.Series(np.sqrt(wind_points_interp['u10']**2 + wind_points_interp['v10']**2), index=time_points_interp.values)
    series_heat                   = pd.Series(all_fluxes.values, index=all_fluxes.time) 
    
    if config['add_ext_interp']:
        return resample_time, series_wind, series_heat, lon_points_interp, lat_points_interp, time_points_interp
    else:
        return resample_time, series_wind, series_heat

###### Example usage ######
# config = {
#     "data_in"         : data_ML_nonan,
#     "data_wind"       : data_wind,
#     "data_heat"       : data_heat,
#     "start_time"      : pd.to_datetime("2022-03-01T00:00:00"),
#     "end_time"        : pd.to_datetime("2022-03-31T05:00:00"),
#     "interp_method"   : "linear",
#     "depth_min"       : 20,  # Drop upper 50m (wave effects, etc.)
#     "rolling_window"  : "1D",
#     "rolling_min_periods": 1,  # Minimum number of observations within the window
#     "max_lag"         : 24 * 3,
#     "freq"            : 'H',
# }
# ds_reduced                                = preprocess_data(config)
# resample_time, series_wind, series_heat   = interpolate_onto_track(ds_reduced, config)


def get_w_timeseries(resample_time, ds_reduced, config, return_mask=False, vary='w'):
    """Function to calculate the square root of the rolling mean of w^2 (w_rms) for a given dataset.
    Parameters:
    ------------
    resample_time:  
        time array with regular time steps
    ds_reduced:     xarray dataset with the data
    config:         dictionary with the configuration parameters
    return_mask:    boolean to return the mask of gaps exceeding an hour
    vary:           string, 
        variable name to calculate the rolling mean of (default is 'w', vertical velocity)
    """
    rolling_window      = config['rolling_window']
    rolling_min_periods = config['rolling_min_periods']
    interp_method       = config['interp_method']

    # Mask out nans in w and calculate w^2
    w_nonan                                    = ds_reduced[vary][~np.isnan(ds_reduced[vary])]**2
    # rolling over w^2 (for <w^2> part)
    w_nonan_roll                               = w_nonan.to_dataframe().rolling(rolling_window, center=True, min_periods=rolling_min_periods).median()
    # construct new array, otherwise NaNs are reput
    ds_w_nonan_roll                            = w_nonan_roll.to_xarray() #same as : xr.DataArray(w_nonan_roll.w, coords=[w_nonan.time], dims=["time"])
    ds_w_nonan_roll['w_roll_2']                = (('time'), w_nonan_roll[vary].values)

    # Interpolation 
    ds_w_roll_interp                           = ds_w_nonan_roll.interp(time=resample_time, method=interp_method, kwargs={'fill_value': np.nan})#.dropna(dim='time')  #interpolate to the same time as wind data
    
    # Find gaps above an hour and drop them
    is_gap_mask                                = get_gaps(resample_time, ds_w_nonan_roll.time) 
    ds_w_roll_interp_clean                     = ds_w_roll_interp.w_roll_2[~is_gap_mask] 

    # Calculate square root of the rolling mean of w^2
    w_data                                     = np.sqrt(ds_w_roll_interp_clean.values)
    series_w                                   = pd.Series(w_data, index=ds_w_roll_interp_clean.time)

    # Reindex the series to fill gaps with NaNs (at moment necressary for correlation calculation)
    series_w_reindex                           = series_w.reindex(resample_time, fill_value=np.nan) 
    if return_mask:
        return series_w_reindex, is_gap_mask
    else:
        return series_w_reindex
    
def calculate_wprime(ds_reduced, config, return_mask=False):
    """Function to calculate the square root of the rolling mean of wprime^2 (wprime_rms) for a given dataset.
    wprime_rms = sqrt(<wprime^2>)=sqrt(<(w-<w>)^2>)

    Parameters:
    ------------
    ds_reduced:     xr.Dataset
                    vertical velocity variable 'w' (default) or other variable specified in config as "var"
    config:         dict
                    Configuration parameters for rolling windows and minimum periods
    return_mask:    array, optional
                    boolean mask of nan values in the vertical velocity variable

    Returns:
    ------------
    ds_w_nonan_roll: xr.DataArray
                    dimensions: time, var name: 'w_prime_rms'
                    dataset with the square root of the rolling mean of wprime^2 (wprime_rms)
    """
    rolling_window_inner      = config['rolling_window_inner']
    rolling_min_periods_inner = config['rolling_min_periods_inner']
    rolling_window_outer      = config['rolling_window_outer']
    rolling_min_periods_outer = config['rolling_min_periods_outer']
    
    var                 = config.get("var", 'w') #default variable of vertical velocity
    # Mask out nans in w
    is_nan                                     = np.isnan(ds_reduced[var])   
    w_nonan                                    = ds_reduced[var][~is_nan]
    # Inner rolling window to define w'
    w_nonan_roll                               = w_nonan.to_dataframe().rolling(rolling_window_inner, min_periods=rolling_min_periods_inner, center=True).mean() #rolling_window
    # Outer rolling window to define the rms of w'
    w_prime_rms                                = np.sqrt(((w_nonan - w_nonan_roll[var])**2).to_dataframe().rolling(rolling_window_outer, center=True, min_periods=rolling_min_periods_outer).mean()[var].values)
    # construct new array, otherwise NaNs are reput (only nans now are if there are not enough data points for inner/outer rolling mean as specified by min_periods)
    ds_w_nonan_roll                            = xr.DataArray(w_prime_rms, coords=[w_nonan.time], dims=["time"], name='w_prime_rms')

    if return_mask:
        return ds_w_nonan_roll, is_nan
    else:
        return ds_w_nonan_roll
    
def calculate_wprime_innerwindow(ds_reduced, config, return_mask=False):
    """Function to calculate wprime^2 (w_prime_squared) for a given dataset.
    wprime_squared = (w-<w>)^2 = w'^2

    Parameters:
    ------------
    ds_reduced:     xr.Dataset
                    vertical velocity variable 'w' (default) or other variable specified in config as "var"
    config:         dict
                    Configuration parameters for rolling windows and minimum periods
                    contains: 'rolling_window_inner', 'rolling_min_periods_inner', 'var'
    return_mask:    array, optional
                    boolean mask of nan values in the vertical velocity variable
    
    Returns:
    ------------
    ds_w_nonan_roll: xr.DataArray
                    dimensions: time, var name: 'w_prime_squared'
                    dataset with the square of wprime^2 (w_prime_squared)
    """
    rolling_window_inner      = config['rolling_window_inner']
    rolling_min_periods_inner = config['rolling_min_periods_inner']
    
    var                 = config.get("var", 'w') #default variable of vertical velocity
    # Mask out nans in w
    is_nan                                     = np.isnan(ds_reduced[var])   
    w_nonan                                    = ds_reduced[var][~is_nan]
    # Inner rolling window to define <w>
    w_nonan_roll                               = w_nonan.to_dataframe().rolling(rolling_window_inner, min_periods=rolling_min_periods_inner, center=True).mean() #rolling_window
    # Square of w'2 = (w-<w>)^2 as array
    w_prime_2                                = ((w_nonan - w_nonan_roll[var])**2).to_dataframe()[var].values
    # construct new array, otherwise NaNs are reput (only nans now are if there are not enough data points for inner/outer rolling mean as specified by min_periods)
    ds_w_nonan_roll                            = xr.DataArray(w_prime_2, coords=[w_nonan.time], dims=["time"], name='w_prime_squared')

    if return_mask:
        return ds_w_nonan_roll, is_nan
    else:
        return ds_w_nonan_roll

def calculate_wprime_outerwindow(wprime2, config):
    """Function to calculate the square root of the rolling mean of wprime^2 (wprime_rms) for a given dataset.
    wprime_rms = sqrt(<wprime^2>)=sqrt(<(w-<w>)^2>)
    Parameters:
    ------------
    wprime2:       xr.DataArray
                    wprime^2 values, dimensions: time, var name: 'w_prime_squared'
    config:        dict
                    Configuration parameters for rolling windows and minimum periods
                    contains: 'rolling_window_outer', 'rolling_min_periods_outer'
    Returns:
    ------------
    ds_w_nonan_roll: xr.DataArray
                    dimensions: time, var name: 'w_prime_rms'
                    dataset with the square root of the rolling mean of wprime^2 (wprime_rms)
    """
    rolling_window_outer      = config['rolling_window_outer']
    rolling_min_periods_outer = config['rolling_min_periods_outer']

    # Outer rolling window to define the rms of w'
    w_prime_rms                                = np.sqrt((wprime2).to_dataframe().rolling(rolling_window_outer, center=True, min_periods=rolling_min_periods_outer).mean()['w_prime_squared'].values)
    # construct new array, otherwise NaNs are reput (only nans now are if there are not enough data points for inner/outer rolling mean as specified by min_periods)
    ds_w_nonan_roll                            = xr.DataArray(w_prime_rms, coords=[wprime2.time], dims=["time"], name='w_prime_rms')

    return ds_w_nonan_roll



    
def get_wprime_timeseries_update_3(resample_time, ds, config, 
                                   return_mask=True, debug=True, name='w_prime', is_down=1, max_gap_duration=np.timedelta64(1, "h"), mld_var='MLD', fill_value_mld=None):
    """Function to calculate the vertical velocity variance and interpolate to resample time excluding any data from outside the mixed layer (ML).
    wprime_rms = sqrt(<wprime^2>)=sqrt(<(w-<w>)^2>), w: vertical velocity
    We follow these steps:
    1) Calculate rolling_inner(w) and substract from w: (w-<w>)
    2) Mask out data below the MLD
    3) Identify time gaps and mask the data if
        the glider leaves the ML for a time similar to the inner rolling window, 
        then also mask surrounding half inner rolling window of data before and after leaving the ML
    4) Do the outer rolling window on the wprime: sqrt(<(w-<w>)^2>)
    5) Interpolate segmentwise on the time array resample_time

    Parameters:
    ------------
    resample_time:  DatetimeIndex
                    time array with regular time steps
    ds:     xarray.Dataset
                    dataset with the data, containing 'w' variable (default) or other variable specified in config as "var"
    config:         dict
                    Configuration parameters for rolling windows and minimum periods
                    contains: 'interp_method', 'rolling_window_inner', 'rolling_window_outer', 'rolling_min_periods_inner', 'rolling_min_periods_outer', 'var'
    return_mask:    boolean
                    whether to return a mask of gaps for the data within the mixed layer which have no influence from data outside the MLD via the inner rolling window
                    shape is different from ds as it is only for the data within the MLD and without nans (to align: ds[var][~is_nan][is_below][~gap_mask])
    debug:          boolean 
                    whether to return intermediate results for debugging
    name:           str
                    name of the variable to be returned (default is 'w_prime')
    is_down:        int
                    direction index of the glider when it is going down (default is 1, meaning down)
    max_gap_duration: np.timedelta64
                    maximum duration of the glider to be outside the ML to be considered a gap(default is inner rolling window duration)
    mld_var:        str
                    name of the mixed layer depth variable in the dataset (default is 'MLD')
    fill_value_mld: float or None
                    value to fill NaN values in the mixed layer depth variable (default is None, meaning no filling)
    
    Returns:
    ------------
    series_w_prime: pd.Series
                    Series of wprime values interpolated onto resample_time, dimensions: time, name: 'w_prime'
    gap_mask:       np.ndarray, optional
                    boolean mask of gaps exceeding an hour in the data within the mixed layer, shape is different from ds as it is only for the data within the MLD and without nans
    intermediates:  dict, optional
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
    """
    intermediates = {} if debug else None

    interp_method = config['interp_method']
    max_gap_duration = config['rolling_window_inner'] if max_gap_duration is None else max_gap_duration  # Use the inner rolling window duration if not specified

    # 0) Calculate wprime: (w-<w>)2
    wprime_inner, is_nan = calculate_wprime_innerwindow(ds, config, return_mask=return_mask)
    if debug:
        intermediates['wprime_inner'] = wprime_inner
        intermediates['is_nan'] = is_nan

    # 1) Mask out data below the MLD
    is_below = _mask_below_mld(ds, is_nan, mld_var=mld_var, fill_value=fill_value_mld)
    wprime_in_mld = wprime_inner[is_below]
    if debug:
        intermediates['wprime_in_mld'] = wprime_in_mld
        intermediates['is_below'] = is_below

    # 2) Identify time gaps and mask surrounding data
    direction_in_mld = ds['profile_direction'][~is_nan][is_below] # get only data in the ML and witout nans
    time_leaves_mld, time_enters_mld, gap_mask = _get_gap_mask(wprime_in_mld, direction_in_mld, is_down=is_down, max_gap_duration=max_gap_duration)
    if debug:
        intermediates['time_leaves_mld'] = time_leaves_mld
        intermediates['time_enters_mld'] = time_enters_mld
        intermediates['gap_mask'] = gap_mask

    # 3) Do the outer rolling window on the wprime
    wprime2 = calculate_wprime_outerwindow(wprime_in_mld[~gap_mask], config)
    if debug:
        intermediates['wprime_outer'] = wprime2

    # 4) Interpolate segments between gaps
    wprime_interp_seg = _interpolate_segments(
        wprime2, time_enters_mld, time_leaves_mld, resample_time, interp_method, name=name
    )
    if debug:
        intermediates['wprime_interp_seg'] = wprime_interp_seg

    # 5) Convert to pandas Series
    series_w_prime = wprime_interp_seg.to_pandas()[name]
    
    if return_mask and debug:
        return series_w_prime, gap_mask, intermediates
    elif return_mask:
        return series_w_prime, gap_mask
    elif debug:
        return series_w_prime, intermediates
    else:
        return series_w_prime
    


    
def get_wprime_timeseries_update_2(resample_time, ds, config, 
                                   return_mask=True, debug=True, name='w_prime', is_down=1, max_gap_duration=np.timedelta64(1, "h"), mld_var='MLD', fill_value_mld=None):
    """Function to calculate the square root of the rolling mean of wprime^2 (wprime_rms) for a given dataset.
    this version might contain more data from outside the mixed layer depth (MLD) than get_wprime_timeseries_update_3 as it does not mask out the data below the MLD before calculating the rolling mean.

    wprime_rms = sqrt(<wprime^2>)=sqrt(<(w-<w>)^2>), w: vertical velocity
    We follow these steps:
    1) Calculate sqrt(<(w-<w>)^2>)
    2) Mask out data below the MLD
    3) Identify time gaps and mask the data if
        the glider leaves the ML for a time similar to the inner rolling window, 
        then also mask surrounding half inner rolling window of data before and after leaving the ML
    4) Interpolate segmentwise on the time array resample_time

    Parameters:
    ------------
    resample_time:  DatetimeIndex
                    time array with regular time steps
    ds:     xarray.Dataset
                    dataset with the data, containing 'w' variable (default) or other variable specified in config as "var"
    config:         dict
                    Configuration parameters for rolling windows and minimum periods
                    contains: 'interp_method', 'rolling_window_inner', 'rolling_window_outer', 'rolling_min_periods_inner', 'rolling_min_periods_outer', 'var'
    return_mask:    boolean
                    whether to return a mask of gaps for the data within the mixed layer which have no influence from data outside the MLD via the inner rolling window
                    in this version, there is still data below the ML via the outer rolling window, so the mask is not as strict as in get_wprime_timeseries_update_3
                    shape is different from ds as it is only for the data within the MLD and without nans (to align: ds[var][~is_nan][is_below][~gap_mask])
    debug:          boolean 
                    whether to return intermediate results for debugging
    name:           str
                    name of the variable to be returned (default is 'w_prime')
    is_down:        int
                    direction index of the glider when it is going down (default is 1, meaning down)
    max_gap_duration: np.timedelta64
                    maximum duration of the glider to be outside the ML to be considered a gap (default is inner rolling window duration)
    mld_var:        str
                    name of the mixed layer depth variable in the dataset (default is 'MLD')
    fill_value_mld: float or None
                    value to fill NaN values in the mixed layer depth variable (default is None, meaning no filling)
    
    Returns:
    ------------
    series_w_prime: pd.Series
                    Series of wprime values interpolated onto resample_time, dimensions: time, name: 'w_prime'
    gap_mask:       np.ndarray, optional
                    boolean mask of gaps exceeding an hour in the data within the mixed layer, shape is different from ds as it is only for the data within the MLD and without nans
    intermediates:  dict, optional
                    dictionary with intermediate results for debugging, contains:
                    - 'wprime_rms': wprime values calculated for both the inner and outer rolling windows
                    - 'is_nan': boolean mask of NaN values in the vertical velocity variable
                    - 'wprime_in_mld': wprime values within the mixed layer
                    - 'is_below': boolean mask of data below the mixed layer depth
                    - 'time_leaves_mld': times when the glider leaves the mixed layer
                    - 'time_enters_mld': times when the glider enters the mixed layer
                    - 'gap_mask': boolean mask of gaps exceeding an hour in the data within the mixed layer
                    - 'wprime_interp_seg': wprime values interpolated onto resample_time
    """
    intermediates = {} if debug else None
    
    interp_method       = config['interp_method']
    max_gap_duration    = config['rolling_window_inner'] if max_gap_duration is None else max_gap_duration  # Use the inner rolling window duration if not specified

    # 0) Calculate wprime_rms: sqrt(<(w-<w>)^2>)
    w_rms, is_nan = calculate_wprime(ds, config, return_mask=return_mask)
    if debug:
        intermediates['w_prime_rms'] = w_rms
        intermediates['is_nan'] = is_nan

    # 1) Mask out data below the MLD
    is_below = _mask_below_mld(ds, is_nan, mld_var=mld_var, fill_value=fill_value_mld)  # Mask out data below the MLD
    wprime_in_mld = w_rms[is_below]
    direction_in_mld = ds['profile_direction'][~is_nan][is_below]  # Save direction of the glider in the MLD excluding nans
    if debug:
        intermediates['wprime_in_mld'] = wprime_in_mld
        intermediates['is_below'] = is_below

    # 2) Identify time gaps and mask surrounding data
    time_leaves_mld, time_enters_mld, gap_mask = _get_gap_mask(wprime_in_mld, direction_in_mld, is_down=is_down, max_gap_duration=max_gap_duration)
    if debug:
        intermediates['time_leaves_mld'] = time_leaves_mld
        intermediates['time_enters_mld'] = time_enters_mld
        intermediates['gap_mask'] = gap_mask

    # 3) Interpolate segments between gaps
    wprime_interp_seg = _interpolate_segments(
        wprime_in_mld[~gap_mask], time_enters_mld, time_leaves_mld, resample_time, interp_method, name=name
    )
    if debug:
        intermediates['wprime_interp_seg'] = wprime_interp_seg

    # 4) Convert to pandas Series
    series_w_prime = wprime_interp_seg.to_pandas()[name]

    if return_mask and debug:
        return series_w_prime, gap_mask, intermediates
    elif return_mask:
        return series_w_prime, gap_mask
    elif debug:
        return series_w_prime, intermediates
    else:
        return series_w_prime
    

def _mask_below_mld(ds, is_nan, mld_var='MLD', fill_value=None):
    """Mask when depth is deeper than the mixed layer depth (MLD).
    Taking absolute values of depth and MLD to ensure correct comparison.

    Parameters:
    ------------
    ds: xarray.Dataset
        Dataset containing the mixed layer depth variable.
    is_nan: np.ndarray
        Boolean array indicating NaN values (e.g. in the vertical velocity variable).
    mld_var: str
        Name of the mixed layer depth variable in the dataset (default is 'MLD').
    fill_value: float or None
        Value to fill NaN values in the mixed layer depth variable (default is None, meaning no filling).
    Returns:
    ------------
    is_below: np.ndarray
        Boolean mask indicating where the depth is below the mixed layer depth (MLD).
        True if depth is below MLD, False otherwise.
    """
    MLD = ds[mld_var][~is_nan]
    if fill_value is not None and np.isnan(MLD).any():
        print(f"Warning: {mld_var} contains NaN values, replacing them with {fill_value} m")
        MLD = MLD.fillna(fill_value)  # or use .where(~np.isnan(MLD), 1000)
    is_below = abs(ds.depth[~is_nan]) < abs(MLD)
    return is_below


def _get_gap_mask(wprime_in_mld, direction_in_mld=None, is_down=1, max_gap_duration=np.timedelta64(1, "h")):
    """Identify the gaps when the glider leaves the mixed layer depth (MLD) and mask out half of the gap duration before and after the gap.
    This function assumes that the glider is leaving the MLD when the time difference between two consecutive measurements exceeds the max_gap_duration and 
    if direction is provided, it checks if the direction is 'down' (is_down=1).
    
    Parameters:
    ------------
        wprime_in_mld:      xarray.DataArray
                            DataArray containing wprime values masked within the mixed layer depth (MLD). dimensions should include 'time'.
        direction_in_mld :  xarray.DataArray, optional
                            DataArray containing the direction of the glider in the MLD. If provided, it should have the same time dimension as wprime_in_mld.
        is_down :           int, optional
                            Value indicating when the glider is diving downwards. Default is 1 (for slocum).
    
    Returns:
    ------------
        time_leaves_mld :   np.ndarray
                            Array of times when the glider leaves the MLD.
        time_enters_mld :   np.ndarray
                            Array of times when the glider enters the MLD again.
        gap_mask :          np.ndarray
                            Boolean mask indicating where gaps exist in the data, with half of the gap duration masked before and after the gap.
    """
    # Settings
    half_gap_duration = max_gap_duration.astype('timedelta64[s]') / 2 # Cast into seconds before dividing to avoid truncation errors

    # Calculate the time difference between consecutive measurements in the mld
    time_diff = wprime_in_mld.time.diff('time')
    index_leaves_mld = time_diff > max_gap_duration

    if direction_in_mld is not None:
        # If direction is provided, ensure that we only consider the glider leaving the MLD if the direction is 'down'
        direction = direction_in_mld.isel(time=slice(None, -1)).values # get directions with same shape like time_diff
        index_leaves_mld = index_leaves_mld & (direction == is_down)  # Only keep indices where the direction of the glider is 'down'

    # Shift the index to get the next time step when the glider enters the MLD again
    index_enters_mld = index_leaves_mld.shift(time=1, fill_value=False)

    # Get the times when the glider leaves and enters the MLD
    time_leaves_mld = wprime_in_mld.time.isel(time=slice(None, -1)).values[index_leaves_mld] #using slice(None, -1) to align with time_diff
    time_enters_mld = wprime_in_mld.time.isel(time=slice(None, -1)).values[index_enters_mld]

    # Mask out half of the gap duration before and after the gap
    time_leaves_mld_before = time_leaves_mld - half_gap_duration
    time_enters_mld_after = time_enters_mld + half_gap_duration

    # Create a mask for the time range where the glider is within half of the gap duration before and after being outside of the MLD
    gap_mask = np.logical_or.reduce([
        (wprime_in_mld.time <= after) & (wprime_in_mld.time >= before)
        for before, after in zip(time_leaves_mld_before, time_enters_mld_after)
    ])

    return time_leaves_mld, time_enters_mld, gap_mask




def _interpolate_segments(wprime_in_mld, time_enters_mld, time_leaves_mld, resample_time, interp_method, name='w_prime'):
    """ This function takes the wprime_in_mld data and interpolates it onto the resample_time array, 
    segmenting the data based on when the glider enters and leaves the mixed layer depth (MLD).

    Parameters:
    ------------
        wprime_in_mld:      xarray.DataArray
                            DataArray containing wprime values masked within the mixed layer depth (MLD). dimensions should include 'time'.
        time_enters_mld:    np.ndarray
                            Array of times when the glider enters the MLD.
        time_leaves_mld:    np.ndarray
                            Array of times when the glider leaves the MLD.
        resample_time:      pd.DatetimeIndex
                            Time array with regular time steps to interpolate onto.
        interp_method:      str
                            Interpolation method to use (e.g., 'linear', 'nearest').
        name:               str, optional
                            Name of the variable to be returned (default is 'w_prime').
    Returns:
    ------------
        wprime_interp_seg:  xarray.Dataset
                            Dataset containing the interpolated wprime values on resample_time. 
                            Dimensions are 'time' and variable name is specified by 'name'.
    """
    wprime_interp_seg = xr.Dataset(
        coords={'time': resample_time},
        data_vars={name: ('time', np.nan * np.ones(len(resample_time)))}
    )

    for i in range(len(time_enters_mld) - 1):
        seg = wprime_in_mld.isel(time=(
            (wprime_in_mld.time >= time_enters_mld[i]) &
            (wprime_in_mld.time <= time_leaves_mld[i + 1])
        ), drop=True)

        next_full_hour = (time_enters_mld[i] + pd.Timedelta(hours=0)).ceil('H')
        last_full_hour = (time_leaves_mld[i + 1] + pd.Timedelta(hours=0)).floor('H')
        full_hours = pd.date_range(start=next_full_hour, end=last_full_hour, freq='H')

        if len(full_hours) == 0 or seg.time.size == 0:
            continue

        seg_interp = seg.interp(time=full_hours, method=interp_method, kwargs={'fill_value': np.nan})
        wprime_interp_seg[name].loc[seg_interp.time] = seg_interp

    return wprime_interp_seg


def get_heat_wind_timeseries(resample_time, ds_reduced, config):
    """Function to calculate the wind and heat fluxes above a glider track.

    Parameters:
    ------------
    resample_time:  DatetimeIndex
                    time array with regular time steps
    ds_reduced:     xarray.Dataset
                    dataset with the data, containing 'longitude' and 'latitude' variables
    config:         dict
                    Configuration parameters for interpolation and data selection
                    contains: 'interp_method', 'data_wind', 'data_heat'

    Returns:
    ------------
    series_wind:    pd.Series
        Series of wind speed values interpolated onto resample_time, dimensions: time
    series_heat:    pd.Series
        Series of heat flux values interpolated onto resample_time, dimensions: time
    """
    interp_method = config['interp_method']
    data_wind     = config['data_wind']
    data_heat     = config['data_heat']

    # Interpolate data onto resample time
    ds_interp                                                   = ds_reduced.interp(time=resample_time, method=interp_method, kwargs={'fill_value': np.nan})
    glider_lon_interp, glider_lat_interp                        = ds_interp.longitude.values, ds_interp.latitude.values 

    # Pointwise Interpolation onto the glider positions
    # to trigger the pointwise indexing, they need to create DataArray objects with the same dimension name, and then use them to index the DataArray. (https://tutorial.xarray.dev/intermediate/indexing/advanced-indexing.html)
    lat_points_interp             = xr.DataArray(glider_lat_interp, dims="points") 
    lon_points_interp             = xr.DataArray(glider_lon_interp, dims="points")
    time_points_interp            = xr.DataArray(resample_time    , dims="points")

    wind_points_interp            = data_wind.interp(longitude=lon_points_interp, latitude=lat_points_interp, time=time_points_interp, method=interp_method, kwargs={'fill_value': np.nan})
    heat_points_interp            = data_heat.interp(longitude=lon_points_interp, latitude=lat_points_interp, time=time_points_interp, method=interp_method, kwargs={'fill_value': np.nan})
    all_fluxes                    = -(heat_points_interp['slhf']   + heat_points_interp['sshf'] + heat_points_interp['ssr'] + heat_points_interp['str'])/3600 #J/m2 to W/m2 needs to be divided by 3600 as it is on a hourly resolution
    
    # Create pandas Series for wind and heat fluxes
    series_wind                   = pd.Series(np.sqrt(wind_points_interp['u10']**2 + wind_points_interp['v10']**2), index=time_points_interp.values)
    series_heat                   = pd.Series(all_fluxes.values, index=all_fluxes.time) 

    return series_wind, series_heat

#### Example usage ####
# series_w_reindex                    = get_w_timeseries(resample_time, ds_reduced, config)
# series_wind, series_heat    = get_heat_wind_timeseries(resample_time, ds_reduced, config)


def interpolate_vars_onto_track(ds, config, time_dim = 'time'):
    """
    Interpolates wind and heat flux data onto the glider track.

    Parameters:
    ------------
        ds :     xarray.Dataset
                 Input dataset with glider track data containing 'latitude', 'longitude', and 'time'.
        config : dict
                Configuration dictionary containing interpolation parameters:
                - 'interp_method' (str): Interpolation method for interpolating atmospheric data on track.
                - 'data' (xarray.Dataset): Wind data for interpolation containing variables to interpolate (hourly).
                - 'vars_to_interp' (list): List of variables to interpolate onto the glider track.
                - 'start_time' (str or pd.Timestamp): Start time for the time selection.
                - 'end_time' (str or pd.Timestamp): End time for the time selection.
                - 'freq' (str): Frequency to interpolate the atmospheric data onto. Should match the atmosph. data frequency.
                - 'add_ext_interp' (bool): Whether to return additional interpolation information like the glider lon,lat,time points used (default is False).

    Returns:
    ------------
        resample_time : pd.DatetimeIndex
                        Regular time array with the specified frequency.
        series_vars : pd.Series
                      Interpolated variables on the glider track, indexed by time.
        lon_points_interp : xarray.DataArray, optional
                            Interpolated longitude points of the glider track.
        lat_points_interp : xarray.DataArray, optional
                            Interpolated latitude points of the glider track.
        time_points_interp : xarray.DataArray, optional
                             Interpolated time points of the glider track.
    """
    # Extract values from the config dictionary
    interp_method                 = config['interp_method']
    data                          = config['data']
    vars_to_interp                = config['vars_to_interp']

    # Create a regular time array with a given frequency between start and end time
    resample_time                 = pd.date_range(start=config['start_time'], end=config['end_time'], freq=config['freq']) # Create a regular time array with hourly resolution
    
    # interpolate data onto resample time linearly 
    # maybe it would make sense to use dropna with a subset so dropna(dim=time, subset=longitude,latitude), but maybe it is not necessary here
    ds_interp                     = ds.interp(time=resample_time, method=interp_method, kwargs={'fill_value': np.nan})#.dropna(dim='time')  #interpolate to the same time as wind data

    # Pointwise Interpolation onto the glider positions
    # to trigger the pointwise indexing, they need to create DataArray objects with the same dimension name, and then use them to index the DataArray. (https://tutorial.xarray.dev/intermediate/indexing/advanced-indexing.html)
    lat_points_interp             = xr.DataArray(ds_interp.latitude.values , dims="points")  # glider latitude at resample time
    lon_points_interp             = xr.DataArray(ds_interp.longitude.values, dims="points") # glider longitude at resample time
    time_points_interp            = xr.DataArray(resample_time             , dims="points")              # resample time as xarray object
    data_points_interp            = data.interp(longitude=lon_points_interp, latitude=lat_points_interp, **{time_dim: time_points_interp}, method=interp_method, kwargs={'fill_value': np.nan})
    #print(data_points_interp[vars_to_interp])

    # Create pandas Series for wind and heat fluxes
    if len(vars_to_interp) == 1:
        series_vars                   = pd.Series(data_points_interp[vars_to_interp[0]], index=time_points_interp.values)
    else:
        series_vars                   = data_points_interp[vars_to_interp].to_dataframe()#.set_index(time_dim)
    
    if config['add_ext_interp']:
        return resample_time, series_vars, lon_points_interp, lat_points_interp, time_points_interp
    else:
        return resample_time, series_vars



def get_wprime_timeseries(resample_time, ds_reduced, config, return_mask=False):
    """Function to calculate the square root of the rolling mean of wprime^2 (wprime_rms) for a given dataset.
    wprime_rms = sqrt(<wprime^2>)=sqrt(<(w-<w>)^2>)
    Parameters:
    ------------
    resample_time:  DatetimeIndex
                    time array with regular time steps
    ds_reduced:     xarray dataset
                    containing the vertical velocity variable 'w' (default) or other variable specified in config as "vary"
    config:         dict
                    configuration parameters including
                    vary:           string,
                                    variable name to calculate the rolling mean of (default is 'w', vertical velocity)
                    interp_method:  string,
                                    interpolation method to use (default is 'linear')
                    inner_rolling_window: string,
                                    rolling window for the inner rolling mean (default is '1h')
                    outer_rolling_window: string,
                                    rolling window for the outer rolling mean (default is '1h')
                    inner_rolling_min_periods: int,
                                    minimum number of observations within the inner rolling window (default is 1)
                    outer_rolling_min_periods: int,
                                    minimum number of observations within the outer rolling window (default is 1)

    return_mask:    bool,
                    if True, returns the mask of gaps exceeding an hour

    Returns:
    ------------
    series_w_prime: pd.Series
                    Series of wprime_rms values interpolated onto the resample_time
    is_gap_mask:    np.ndarray, optional
                    boolean mask of gaps exceeding an hour, if return_mask is True
    """
    ds_w_nonan_roll     = calculate_wprime(ds_reduced, config)
    interp_method       = config['interp_method']
    #xr.DataArray((w_nonan - w_nonan_roll.w)**2, coords=[w_nonan.time], dims=["time"])#w_nonan_roll.to_xarray() 
    #ds_w_nonan_roll['w_roll_2']                = (('time'), w_nonan_roll[vary].values)

    # Interpolation 
    ds_w_roll_interp                           = ds_w_nonan_roll.interp(time=resample_time, method=interp_method, kwargs={'fill_value': np.nan})#.dropna(dim='time')  #interpolate to the same time as wind data

    # Find gaps above an hour and drop them
    is_gap_mask                                = get_expanded_gaps(resample_time, ds_w_nonan_roll.time) 
    ds_w_roll_interp_clean                     = ds_w_roll_interp[~is_gap_mask] 

    # Calculate square root of the rolling mean of w^2
    w_data                                     = (ds_w_roll_interp_clean.values)
    series_w                                   = pd.Series(w_data, index=ds_w_roll_interp_clean.time)

    # Reindex the series to fill gaps with NaNs (at moment necressary for correlation calculation)
    series_w_prime_reindex                     = series_w.reindex(resample_time, fill_value=np.nan) 
    if return_mask:
        return series_w_prime_reindex, is_gap_mask
    else:
        return series_w_prime_reindex


def get_gaps(resample_time, data_time, dt_max = np.timedelta64(1, "h")):
    ''' Function to find gaps in the data that are larger than dt_max
    Parameters:
    ------------
    resample_time:  DatetimeIndex
                    time array with regular time steps
    data_time:      time array of the data with irregular time steps and gaps
    dt_max:         maximum time difference above, beyond which we consider it as a gap

    Returns:
    ------------
    is_gap_dt:      np.ndarray
                    boolean array indicating gaps larger than dt_max, with True for gaps larger than dt_max, has same length like resample_time
    '''
    # Find the indices in `data_time` that correspond to each time point in `resample_time`.
    # Use 'right' to find the smallest index in `data_time` where the time is greater than `resample_time`.
    x_index = np.searchsorted(data_time, resample_time, side='right')
    x_index = np.clip(x_index, 0, len(data_time)-1)

    # Identify the indices of the closest earlier and later times in `data_time` for each `resample_time` point.
    prev_idx = x_index - 1
    next_idx = x_index

    # Retrieve the actual timestamps in 'data_time' corresponding to the identified indices.
    prev_time = data_time[prev_idx]
    next_time = data_time[next_idx]

    # Determine which of the two times (previous or next) is closer to the `resample_time` point.
    nearest_times = np.where(
        np.abs(resample_time.values - prev_time.values) <= np.abs(resample_time.values - next_time.values),
        prev_time.values,
        next_time.values)
    
    # Calculate the time difference (dt) between `resample_time` and the nearest time in `data_time`.
    dt = nearest_times - resample_time.values

    # Identify gaps: Check if the absolute value of the time difference exceeds `dt_max`.
    # Convert the difference to seconds for comparison.
    is_gap_dt = abs(dt.astype('timedelta64[s]')) > dt_max

    # Return the boolean array indicating the presence of gaps.
    return is_gap_dt

def get_expanded_gaps(resample_time, data_time, dt_max = np.timedelta64(1, "h")):
    ''' Function to find gaps in the data that is influenced by data from the gap by widening the gaps by half of the dt_max
    Parameters:
    ------------
    resample_time:  DatetimeIndex
                    time array with regular time steps
    data_time:      time array of the data with irregular time steps and gaps
    dt_max:         maximum time difference above, beyond which we consider it as a gap

    Returns:
    ------------
    is_gap_dt:      np.ndarray
                    boolean array indicating gaps larger than dt_max, with True for gaps larger than dt_max, has same length like resample_time
                    also excluding values within dt_max/2 from the gap
    '''
    # Find the indices in `data_time` that correspond to each time point in `resample_time`.
    # Use 'right' to find the smallest index in `data_time` where the time is greater than `resample_time`.
    x_index = np.searchsorted(data_time, resample_time, side='right')
    x_index = np.clip(x_index, 0, len(data_time)-1)

    # Identify the indices of the closest earlier and later times in `data_time` for each `resample_time` point.
    prev_idx = x_index - 1
    next_idx = x_index

    # Retrieve the actual timestamps in 'data_time' corresponding to the identified indices.
    prev_time = data_time[prev_idx]
    next_time = data_time[next_idx]

    # Determine which of the two times (previous or next) is closer to the `resample_time` point.
    nearest_times = np.where(
        np.abs(resample_time.values - prev_time.values) <= np.abs(resample_time.values - next_time.values),
        prev_time.values,# - half_dt,
        next_time.values,# + half_dt)
    )
    
    # Modify nearest time based on its position relative to resample_time
    half_dt = (dt_max.astype('timedelta64[s]')/2) # get half of the gap_time (need to convert to seconds, because timedelta keeps the units and rounds to the nearest int value)
    nearest_times = np.where(
        nearest_times < resample_time.values,  # If before, subtract 30 min
        nearest_times - half_dt,
        nearest_times + half_dt                 # If after, add 30 min
    )
    # Calculate the time difference (dt) between `resample_time` and the nearest time in `data_time`.
    dt = nearest_times - resample_time.values

    # Identify gaps: Check if the absolute value of the time difference exceeds `dt_max`.
    # Convert the difference to seconds for comparison.
    is_gap_dt = abs(dt.astype('timedelta64[s]')) > dt_max

    # Return the boolean array indicating the presence of gaps.
    return is_gap_dt
    

def lag_correlation(x, y, max_lag):
    """Calculate the lag correlation between two time series.
    Parameters
    ----------
    x:      pd.Series
            first time series
    y:      pd.Series
            second time series
    max_lag: int
            maximum lag to consider
    Returns
    -------
    lags:   range
            range of lags from -max_lag to max_lag
    corrs:  list
            list of correlation coefficients for each lag
    """
    # function to calculate lag correlation between two time series
    lags  = range(-max_lag, max_lag + 1)
    corrs = [x.corr(y.shift(lag)) for lag in lags]
    return lags, corrs

def Z(r):
    """Fisher Z-Transformation
    Parameters
    ----------
    r:      int or array
            correlation coefficient
    returns:array
            Fisher Z-transformed value of r.
    """
    r = np.asarray(r)  # Ensure r is a numpy array for element-wise operations
    return 0.5*(np.log(r+1) - np.log(1-r))

def inverse_Z(Z):
    """Inverse Fisher Z-transformation
    Parameters
    ----------
    Z:      int or array
            Fisher Z-transformed value
    Returns
    -------
            array
            Inverse Fisher Z-transformed value of Z.
    """
    Z = np.asarray(Z)  # Ensure Z is a numpy array for element-wise operations
    return (np.exp(2 * Z) - 1) / (np.exp(2 * Z) + 1)

def sigma(N):
    """Standart deviation of the Fisher Z-transformed correlation coefficient
    Parameters
    ----------
    N:      int
            number of samples
    Returns:
            float
            standart deviation of the Fisher Z-transformed correlation coefficient
    """
    return 1/(N-3)**(0.5)

def t_value(r):
    """Calculate the t-value from the correlation coefficient r.
    Parameters
    ----------
    r:      int or array
            correlation coefficient
    Returns:
    -------
            array
            t-value corresponding to the correlation coefficient r.
    """
    return r*np.sqrt((len(r)-2)/(1-r**2))

# from Cristina (= 1.96 for 0.05), but should be the same as Z(alpha/2) and it is not
Z_crit = norm.ppf(1-0.05/2) # percent point function, inverse of the cumulative distribution function
#### why is it different from Z(1-(0.05/2)) ??? 

def N_eff_Tint(series):
    """effective number of independent samples from integral timescale, 
    see https://andrewcharlesjones.github.io/journal/21-effective-sample-size.html#:~:text=The%20effective%20sample%20size%20is%20a%20metric%20that,to%20the%20correlation%20and%20redundancy%20between%20the%20samples.
    
    Parameters
    ----------
    series: pd.Series
            time series data for which to calculate the effective sample size
    Returns
    -------
    N_eff:  float
            effective number of independent samples calculated from the integral timescale
    """
    n_lag = len(series)-1
    autocorr = sm.tsa.acf(series, nlags=n_lag)
    T_int = 1 + 2*np.sum(autocorr)
    N_eff = len(series)/T_int
    return N_eff

def confid_interval(r, N_eff, Zcrit=1.96):
    """Confidence interval of the correlation coefficient.

    Parameters
    ----------
    r:      int or array
            correlation coefficient
    alpha:  float
            significance level
    N_eff:  int
            effective number of independent samples
    Returns:
            array
            lower bound, upper bound
    """
    return [inverse_Z(Z(r) - Z_crit*sigma(N_eff)), inverse_Z(Z(r) + Z_crit*sigma(N_eff))]

def test_normal_dist(series1):
    """Test if the distribution of the values is normally distributed.

    Parameters
    ----------

    series1:    array
                first array of values
    series2:    array
                second array of values
    Returns:
                tuple
                test results of the normality test
    """
    # Testing if distribution of values is normally distributed (needed for pearson)
    print('p-value below 0.05 would indicate non-normality')
    print(f'jarqueb: {jarque_bera(series1)}')         # see https://www.statology.org/jarque-bera-test-python/, p-value below 0.05 would indicate skewness
    print(f'shapiro: {shapiro(series1)}')             # see https://www.statology.org/shapiro-wilk-test-python/, p-value below 0.05 would indicate non-normality
    print(f'kstest: {kstest(series1, 'norm')}')       # see https://www.statology.org/kolmogorov-smirnov-test-python/, p-value below 0.05 would indicate non-normality
    return jarque_bera(series1), shapiro(series1),kstest(series1, 'norm')



def generate_signal_w_noise(length=24*10, noise_scale1=0.2, noise_scale2=0.2, magnitude2=1.5):
    """Generate two time series with a sinusoidal signal and different random noise for each.
    Parameters:
    
    length: int, the length of the time series in hours
    noise_scale1: float, the scale of the random noise to add to the signal1
    noise_scale2: float, the scale of the random noise to add to the signal2
    magnitude2: float, the magnitude of the second signal relative to the first
    
    Returns: 
        pd.Series, pd.Series, the two time series with the second shifted by 12 hours

    Example:
    series1, series2    = generate_signal_w_noise()                     # Generate a sinus signal with random noise
    lags_exp, corrs_exp = ag.lag_correlation(series2, series1, max_lag) # Calculate the lag correlation
    """
    import pandas as pd
    # Set the random seed for reproducibility
    np.random.seed(42)
    # Generate a time index for 10 days at 1-hour intervals
    time_index = pd.date_range(start="2024-10-01", periods=length, freq="H")                                             # maximum time lag in hours
    # Create an underlying signal, e.g., a sine wave with some periodic peaks
    underlying_signal = np.sin(np.linspace(0, 10*np.pi, len(time_index)))  # Sine wave signal

    # Add random noise to the signal for the first series
    noise1  = np.random.normal(loc=0, scale=noise_scale1, size=len(time_index)) # increase scale for more noise
    series1 = pd.Series(underlying_signal + noise1, index=time_index)

    # Create the second series by shifting the underlying signal and adding different random noise
    shift_hours = 12  # Shift by 24 hours (1 day)
    noise2  = np.random.normal(loc=0, scale=noise_scale2, size=len(time_index))
    series2 = magnitude2*pd.Series(np.roll(underlying_signal, shift_hours) + noise2, index=time_index)

    return series1, series2


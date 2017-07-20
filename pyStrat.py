import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import statistics
import statsmodels.api as sm





########################################
########## PLOTTING FUNCTIONS ##########
########################################





def read_data(csv_string):
    """
    Imports data from a .csv.

    Args:
        - csv_string (string): path to data .csv

    Returns:
        - data (dataframe): properly formatted data

    Prints:
        - total stratigraphic thickness of the section

    Notes:
        The .csv must follow the form of the template (adapted from the Matstrat
        template):
        - lines 1-4 can be filled with arbitrary relevant information - the code
          will start reading in data at line 5
        - Lithofacies:
            - line 5 MUST contain at least two headers
            - one of these headers MUST be named 'THICKNESS' - other columns may
              be named whatever the user desires
        - Samples:
            - optional
    """
    # the data
    data = pd.read_csv(csv_string, header=4)

    # get rid of blank columns
    cols = []
    for c in data.columns:
        if 'Unnamed' not in c:
            cols.append(c)
    data=data[cols]

    # print total stratigraphic thickness
    print('total stratigraphic thickness = ' + str(np.sum(data['THICKNESS'])) + ' m')

    # return the dataframe
    return data





def read_formatting(csv_string):
    """
    Imports formatting from a .csv.

    Args:
        - csv_string (string): path to formatting .csv

    Returns:
        - formatting (dataframe): properly formatted formatting

    Notes:
        The .csv must follow the form of the template:
        - columns 1-4 are used to set the colour of the boxes:
            - columns 1-3 must be called 'r', 'g', and 'b' (for red, green, and
              blue)
                - values in columns 1-3 must be between 0-255
            - the header of column 4 must match one of the headers used in the
              data .csv, and all values in the data must be a subset of the
              values in this column
        - columns 6-7 are used to set the width of the boxes:
            - column 6 must be called 'width'
            - the header of column 7 must match one of the headers used in the
              data .csv, and all values in the data must be a subset of the
              values in this column
        - column 5 should be left blank for readability
    """
    # the formatting
    formatting = pd.read_csv(csv_string)

    # convert rgb to 0-1 scale for plotting
    for i in range(len(formatting.index)):
        formatting.loc[i,'r'] = formatting.loc[i,'r'] / 255
        formatting.loc[i,'g'] = formatting.loc[i,'g'] / 255
        formatting.loc[i,'b'] = formatting.loc[i,'b'] / 255

    # get the colour and width headers being used
    colour_header = formatting.columns[3]
    width_header = formatting.columns[6]

    # return the dataframe
    return formatting





def integrity_check(data, formatting):
    """
    Check that values in the data are a subset of values in the formatting.

    Args:
        - data (dataframe): properly formatted data
        - formatting (dataframe): properly formatted formatting

    Raises:
        Prints result of integrity check. If fail, also prints the item which
        fails the check.
    """
    # get the colour and width headers being used
    colour_header = formatting.columns[3]
    width_header = formatting.columns[6]

    # if width_header and colour_header are the same, pandas appends a '.1'
    if width_header.endswith('.1'):
        width_header = width_header[:-2]

    all_check = True

    # loop over values in the data
    for i in range(len(data.index)):
        colour_check = False
        width_check = False

        # only check if the value is not nan
        if pd.notnull(data['THICKNESS'][i]):

            # check to see if specified value is in the formatting table
            for j in range(len(formatting.index)):
                if data[colour_header][i] == formatting[colour_header][j]:
                    colour_check = True
                if data[width_header][i] == formatting[width_header][j]:
                    width_check = True

            # print warning if the check fails
            if colour_check == False:
                print('Colour check failed for item ' + str(data[colour_header][i]) + '.')
                all_check = False
            if width_check == False:
                print('Width check failed for item ' + str(data[width_header][i]) + '.')
                all_check = False

    # print an all clear statement if the check passes
    if all_check:
        print('Colour and width check passed.')





def plot_legend(formatting):
    """
    Plot all items in the formatting.

    Args:
        - formatting (dataframe): properly formatted formatting

    Returns:
        - fig (figure): figure handle
        - ax (axis): axis handle
    """
    # get the colour and width headers being used
    colour_header = formatting.columns[3]
    width_header = formatting.columns[6]

    # if width_header and colour_header are the same, pandas appends a '.1'
    if width_header.endswith('.1'):
        width_header = width_header[:-2]

    # if the colour and width headers are the same...
    if colour_header == width_header:

        # sort the formatting
        formatting = formatting.copy()
        formatting.sort_values('width', inplace=True)
        formatting.reset_index(inplace=True, drop=True)

        # initiate fig and ax
        fig, ax = plt.subplots()

        # initiate counting of the stratigraphic height
        strat_height = 0.0

        # loop over each item
        for i in range(len(formatting.index)):
            this_colour = [formatting['r'][i], formatting['g'][i], formatting['b'][i]]
            this_width = formatting['width'][i]

            # create the rectangle - with thickness of 1
            ax.add_patch(patches.Rectangle((0.0,strat_height), this_width, 1, facecolor=this_colour, edgecolor='k'))

            # label the unit
            ax.text(0.02, strat_height+0.5, formatting[colour_header][i], horizontalalignment='left', verticalalignment='center')

            # count the stratigraphic height
            strat_height = strat_height + 1

        # force the limits on the lithostratigraphy plot
        ax.set_xlim([0,1])
        ax.set_ylim([0,strat_height])

        # force the size of the plot
        fig.set_figheight(strat_height * 0.3)
        fig.set_figwidth(3)

        # prettify
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_axisbelow(True)
        ax.xaxis.grid()
        ax.set_yticklabels([])
        ax.set_yticks([])

        return fig, ax

    # if they aren't the same... haven't developed this code yet
    else:
        print('colour_header and width_header are not the same - plot_legend still under development for this case.')





def initiate_figure(data, formatting, strat_ratio, figwidth, width_ratios, linewidth=1):
    """
    Initiate a figure with the stratigraphic column.

    Args:
        - data (dataframe): properly formatted data
        - formatting (dataframe): properly formatted formatting
        - strat_ratio (float): scaling ratio for height of figure
        - figwidth (float): figure width
        - width_ratios (list): width ratios of axes
        - linewidth (float): line width (default = 1)

    Returns:
        - fig (figure): figure handle
        - axs (axes): axes handles
    """
    # get the number of axes from the width_ratios list
    ncols = len(width_ratios)

    # initiate fig and axs
    fig, axs = plt.subplots(nrows=1, ncols=ncols, sharey=True, gridspec_kw={'width_ratios':width_ratios})

    # get the colour and width headers being used
    colour_header = formatting.columns[3]
    width_header = formatting.columns[6]

    # if width_header and colour_header are the same, pandas appends a '.1'
    if width_header.endswith('.1'):
        width_header = width_header[:-2]

    # initiate counting of the stratigraphic height
    strat_height = 0.0

    # loop over elements of the data
    for i in range(len(data.index)):

        # only plot non-nan cells
        if pd.notnull(data['THICKNESS'][i]):

            # find the thickness to be used
            this_thickness = data['THICKNESS'][i]

            # find the colour and width to be used
            for j in range(len(formatting.index)):
                if data[colour_header][i] == formatting[colour_header][j]:
                    this_colour = [formatting['r'][j], formatting['g'][j], formatting['b'][j]]
                if data[width_header][i] == formatting[width_header][j]:
                    this_width = formatting['width'][j]

            # create the rectangle
            axs[0].add_patch(patches.Rectangle((0.0,strat_height), this_width, this_thickness, facecolor=this_colour, edgecolor='k', linewidth=linewidth))

            # count the stratigraphic height
            strat_height = strat_height + this_thickness

    # force the limits on the lithostratigraphy plot
    axs[0].set_xlim([0,1])
    axs[0].set_ylim([0,strat_height])

    # force the size of the plot
    fig.set_figheight(strat_height * strat_ratio)
    fig.set_figwidth(figwidth)

    # prettify
    axs[0].set_ylabel('stratigraphic height [m]')
    axs[0].set_xticklabels([])
    axs[0].set_xticks([])

    return fig, axs





def add_data_axis(fig, axs, ax_num, x, y, plot_type, **kwargs):
    """
    Add an arbitrary data axis to a figure initiated by initiate_figure.

    Args:
        - fig (figure): figure handle initiated by initiate_figure
        - axs (axes): axes handles initiated by initiate_figure
        - ax_num (float): axis on which to plot
        - x (list or array): x-data
        - y (list or array): y-data
        - plot_type (string): 'plot', 'scatter', or 'barh'
        - **kwargs: passed to plt.plot, plt.scatter, or plt.barh

    Notes:
        If 'barh' is selected, it is recommended that a 'height' kwarg be passed
        to this function (see the documentation for matplotlib.pyplot.barh).
        Also, note that 'y' becomes the bottom-left coordinate of the bar.
    """
    # get the correct axis
    ax = axs[ax_num]

    # plot
    if plot_type == 'plot':
        ax.plot(x, y, **kwargs)

    # or scatter
    elif plot_type == 'scatter':
        ax.scatter(x, y, **kwargs)

    elif plot_type == 'barh':
        ax.barh(y, x, **kwargs)

    # or print a warning
    else:
        print("Only 'plot', 'scatter', and 'barh' are supported for 'plot_type'.")





##############################################
########## DATA WRANGLING FUNCTIONS ##########
##############################################





def sample_curate(data, recorded_height, remarks):
    """
    Assigns the correct height and unit to collected samples.

    Args:
        - data (dataframe): properly formatted data
        - recorded_height (list or array): recorded height of samples
        - remarks (list or array): field addition errors noted

    Returns:
        - sample_info (dataframe): recorded_height, remarks, height, and unit

    Notes:
        Field addition errors should be noted as "ADD X" or "SUB X" in the
        remarks array, and only needs to be noted at the first sample where the
        correction comes into effect.

        If a sample was collected at a unit boundary, it will be denoted with
        both adjacent units. The user must assign the correct unit for these
        samples manually.

        Units are zero indexed as in the Python convention.
    """
    # remove nans from the nominal_height array
    no_nans = np.array([])
    for i in range(len(recorded_height)):
        if np.isfinite(recorded_height[i]):
            no_nans = np.append(no_nans, recorded_height[i])
    recorded_height = no_nans

    # make sure the sample list is sorted
    sort_ind = recorded_height.argsort()
    recorded_height = recorded_height[sort_ind]
    remarks = remarks[sort_ind]

    # calculate the true height of the samples
    height = np.array([])
    adjustment = 0.0
    for i in range(len(recorded_height)):
        if 'ADD' in remarks[i]:
            value = float(split(remarks[i])[1])
            adjustment = adjustment + value
        elif 'SUB' in remarks[i]:
            value = float(split(remarks[i])[1])
            adjustment = adjustment - value
        height = np.append(height, recorded_height[i] + adjustment)

    # create a stratigraphic height column
    strat_height = np.array([data['THICKNESS'][0]])
    for i in range(1, len(data['THICKNESS'])):
        if np.isfinite(data['THICKNESS'][i]):
            strat_height = np.append(strat_height, data['THICKNESS'][i] + strat_height[-1])

    # calculate the unit from which the sample came
    unit = np.array([])
    for i in range(len(height)):
        unit_ind = 0
        while strat_height[unit_ind] < height[i]:
            unit_ind = unit_ind + 1

        # if the sample comes from a unit boundary, mark with .5
        if strat_height[unit_ind] == height[i]:
            unit = np.append(unit, unit_ind + 0.5)
        else:
            unit = np.append(unit, unit_ind)

    # create the output dataframe
    sample_info = pd.DataFrame({'recorded_height':recorded_height, 'remarks':remarks, 'height':height, 'unit':unit})

    # print instructions and return
    print('Copy and paste the output dataframe into the data .csv.')
    print(' ')
    print("IMPORTANT: Samples marked with a .5 in the 'unit' column come from")
    print('unit boundaries. User input is required to assign the correct unit.')
    print(' ')
    print('If the sample comes from the lower unit, subtract 0.5')
    print('                             upper unit,      add 0.5')

    return sample_info





###########################################
########## CALCULATION FUNCTIONS ##########
###########################################





def lowess_fit(y, x, frac=0.6666666666666666, it=3):
    """
    LOWESS fit for a scatterplot.

    Args:
        - y (list or array): data y-values
        - x (list or array): data x-values
        - frac (float): between 0 and 1 - the fraction of the data used when
                        estimating each y-value
        - it (int): the number of residual-based reweightings to perform

    Returns:
        - xy (array): the first column is the sorted x values and the second
                      column the associated estimated y-values
    """
    xy = sm.nonparametric.lowess(y, x, frac=frac, it=it)

    return xy





def lowess_normalize(strat_height, vals, frac=0.6666666666666666, it=3):
    """
    Normalize values to the LOWESS fit.

    Args:
        - strat_height (list or array): stratigraphic heights of samples
        - vals (list or array): values of samples
        - frac (float): between 0 and 1 - the fraction of the data used when
                        estimating each y-value in the lowess fit
        - it (int): the number of residual-based reweightings to perform

    Returns:
        - xy (array): the first column is the sorted x values and the second
                       column the associated estimated y-values
        - norm_vals (array): normalized values
        - norm_heights (array): associated height values
    """
    # combine strat_height and d13C into a dataframe
    section_data = pd.DataFrame({'strat_height':strat_height, 'vals':vals})

    # pull out the non-nan values
    finite_vals = np.array([])
    norm_heights = np.array([])
    for i in range(len(section_data.index)):
        if np.isfinite(section_data['vals'][i]):
            finite_vals = np.append(finite_vals, section_data['vals'][i])
            norm_heights = np.append(norm_heights, section_data['strat_height'][i])

    # the lowess fit
    xy = lowess_fit(finite_vals, norm_heights, frac, it)

    # normalize
    norm_vals = finite_vals - xy[:,1]

    return xy, norm_vals, norm_heights




def scatter_variance(strat_height, vals, interval, mode, frac=0.6666666666666666, it=3):
    """
    Calculate the variance in scatterplot data.

    Args:
        - strat_height (list or array): stratigraphic heights of samples
        - vals (list or array): values of samples
        - interval (float): window height used to calculate variance
        - mode (string): 'standard' (raw data), 'window_normalized' (windowed
                         lowess fit subtracted), or 'all_normalized' (lowess fit
                         subtracted)
        - frac (float): between 0 and 1 - the fraction of the data used when
                        estimating each y-value in the lowess fit
        - it (int): the number of residual-based reweightings to perform

    Returns ('standard'):
        - variances (array): calculated variances
        - strat_height_mids (array): middle of windows used

    Additional Returns ('normalized'):
        - xys (array): the first column is the sorted x values and the second
                       column the associated estimated y-values
        - norm_vals (array): normalized values
        - norm_heights (array): associated height values
    """
    # initiate storage arrays
    variances = np.array([])
    strat_height_mids = np.array([])

    # all normalized mode
    if mode == 'all_normalized':
        # lowess normalize all the data
        xys, norm_vals, norm_heights = lowess_normalize(strat_height, vals, frac, it)

        # get the first window bot and top
        window_bot = min(norm_heights)
        window_top = window_bot + interval

        # keep iterating until our window moves past the whole section
        while window_bot <= max(norm_heights):

            # pull out normalized values in the window
            window_norm_vals = np.array([])
            for i in range(len(norm_heights)):
                if norm_heights[i]>=window_bot and norm_heights[i]<window_top:
                    window_norm_vals = np.append(window_norm_vals, norm_vals[i])

            # if we have enough data in the window
            if len(window_norm_vals) >= 2:
                variances = np.append(variances, statistics.variance(window_norm_vals))
                strat_height_mids = np.append(strat_height_mids, (window_top + window_bot)/2)

            # increment the window bot and top
            window_bot = window_bot + interval
            window_top = window_top + interval

    # the other modes
    else:
        # combine strat_height and vals into a dataframe
        section_data = pd.DataFrame({'strat_height':strat_height, 'vals':vals})

        # initiate storage arrays
        norm_vals = np.array([])
        norm_heights = np.array([])
        xys_initialized = False

        # get the first window bot and top
        window_bot = min(section_data['strat_height'])
        window_top = window_bot + interval

        # keep iterating until our window moves past the whole section
        while window_bot <= max(section_data['strat_height']):
            # pull out the current window
            window = section_data[(section_data['strat_height']>=window_bot) &\
                                  (section_data['strat_height']<window_top)]
            window.reset_index(drop=True, inplace=True)

            # pull out non-nan values
            window_vals = np.array([])
            window_heights = np.array([])
            for i in range(len(window.index)):
                if np.isfinite(window['vals'][i]):
                    window_vals = np.append(window_vals, window['vals'][i])
                    window_heights = np.append(window_heights, window['strat_height'][i])

            # only move on if we have enough data
            if len(window_vals) >= 2:
                # the standard mode
                if mode == 'standard':
                    # variances on raw data
                    variances = np.append(variances, statistics.variance(window_vals))

                # the window normalized mode
                elif mode == 'window_normalized':
                    # lowess fit the window
                    xy = lowess_fit(window_vals, window_heights, frac, it)
                    # store lowess fit results
                    if xys_initialized:
                        xys = np.concatenate((xys, xy),axis=0)
                    else:
                        xys = xy
                        xys_initialized = True
                    # normalize the values
                    window_norm_vals = window_vals - xy[:,1]
                    # variances on window normalized data
                    variances = np.append(variances, statistics.variance(window_norm_vals))
                    norm_vals = np.append(norm_vals, window_norm_vals)

                norm_heights = np.append(norm_heights, window_heights)

                # append the middle of the window
                strat_height_mids = np.append(strat_height_mids, (window_top + window_bot)/2)

            # increment the window bot and top
            window_bot = window_bot + interval
            window_top = window_top + interval

    if mode == 'standard':
        return variances, strat_height_mids
    elif mode == 'window_normalized' or mode == 'all_normalized':
        return variances, strat_height_mids, xys, norm_vals, norm_heights





def vincenty_inverse(point1, point2, miles=False):
    """
    Vincenty's formula (inverse method) to calculate the distance (in
    kilometers or miles) between two points on the surface of a spheroid

    Doctests:
    >>> vincenty((0.0, 0.0), (0.0, 0.0))  # coincident points
    0.0
    >>> vincenty((0.0, 0.0), (0.0, 1.0))
    111.319491
    >>> vincenty((0.0, 0.0), (1.0, 0.0))
    110.574389
    >>> vincenty((0.0, 0.0), (0.5, 179.5))  # slow convergence
    19936.288579
    >>> vincenty((0.0, 0.0), (0.5, 179.7))  # failure to converge
    >>> boston = (42.3541165, -71.0693514)
    >>> newyork = (40.7791472, -73.9680804)
    >>> vincenty(boston, newyork)
    298.396057
    >>> vincenty(boston, newyork, miles=True)
    185.414657

    Source: Maurycy Pietrzak
    """

    # WGS 84
    a = 6378137  # meters
    f = 1 / 298.257223563
    b = 6356752.314245  # meters; b = (1 - f)a

    MILES_PER_KILOMETER = 0.621371

    MAX_ITERATIONS = 200
    CONVERGENCE_THRESHOLD = 1e-12  # .000,000,000,001

    # short-circuit coincident points
    if point1[0] == point2[0] and point1[1] == point2[1]:
        return 0.0

    U1 = math.atan((1 - f) * math.tan(math.radians(point1[0])))
    U2 = math.atan((1 - f) * math.tan(math.radians(point2[0])))
    L = math.radians(point2[1] - point1[1])
    Lambda = L

    sinU1 = math.sin(U1)
    cosU1 = math.cos(U1)
    sinU2 = math.sin(U2)
    cosU2 = math.cos(U2)

    for iteration in range(MAX_ITERATIONS):
        sinLambda = math.sin(Lambda)
        cosLambda = math.cos(Lambda)
        sinSigma = math.sqrt((cosU2 * sinLambda) ** 2 +
                             (cosU1 * sinU2 - sinU1 * cosU2 * cosLambda) ** 2)
        if sinSigma == 0:
            return 0.0  # coincident points
        cosSigma = sinU1 * sinU2 + cosU1 * cosU2 * cosLambda
        sigma = math.atan2(sinSigma, cosSigma)
        sinAlpha = cosU1 * cosU2 * sinLambda / sinSigma
        cosSqAlpha = 1 - sinAlpha ** 2
        try:
            cos2SigmaM = cosSigma - 2 * sinU1 * sinU2 / cosSqAlpha
        except ZeroDivisionError:
            cos2SigmaM = 0
        C = f / 16 * cosSqAlpha * (4 + f * (4 - 3 * cosSqAlpha))
        LambdaPrev = Lambda
        Lambda = L + (1 - C) * f * sinAlpha * (sigma + C * sinSigma *
                                               (cos2SigmaM + C * cosSigma *
                                                (-1 + 2 * cos2SigmaM ** 2)))
        if abs(Lambda - LambdaPrev) < CONVERGENCE_THRESHOLD:
            break  # successful convergence
    else:
        return None  # failure to converge

    uSq = cosSqAlpha * (a ** 2 - b ** 2) / (b ** 2)
    A = 1 + uSq / 16384 * (4096 + uSq * (-768 + uSq * (320 - 175 * uSq)))
    B = uSq / 1024 * (256 + uSq * (-128 + uSq * (74 - 47 * uSq)))
    deltaSigma = B * sinSigma * (cos2SigmaM + B / 4 * (cosSigma *
                 (-1 + 2 * cos2SigmaM ** 2) - B / 6 * cos2SigmaM *
                 (-3 + 4 * sinSigma ** 2) * (-3 + 4 * cos2SigmaM ** 2)))
    s = b * A * (sigma - deltaSigma)

    s /= 1000  # meters to kilometers
    if miles:
        s *= MILES_PER_KILOMETER  # kilometers to miles

    return round(s, 6)





def compass_bearing(pointA, pointB):
    """
    Calculates the bearing between two points.

    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))

    :Parameters:
      - `pointA: The tuple representing the latitude/longitude for the
        first point. Latitude and longitude must be in decimal degrees
      - `pointB: The tuple representing the latitude/longitude for the
        second point. Latitude and longitude must be in decimal degrees

    :Returns:
      The bearing in degrees

    :Returns Type:
      float

    Source: jeromer
    """

    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing





def dir2cart(data):
    """
    Converts vector directions, in degrees, to cartesian coordinates, in x,y,z.

    Args:
        - data (array): list of [dec,inc]

    Returns:
        - cart (numpy array): array of [x,y,z]

    Notes:
        - Adapted from pmag.py
    """

    ints = np.ones(len(data)).transpose() # get an array of ones to plug into dec,inc pairs
    data = np.array(data)
    rad = np.pi / 180

    # case if array of vectors
    if len(data.shape) > 1:
        decs = data[:,0] * rad
        incs = data[:,1] * rad

        # take the given lengths, if they exist
        if data.shape[1] == 3:
            ints = data[:,2]

    # case if single vector
    else:
        decs = np.array(data[0]) * rad
        incs = np.array(data[1]) * rad
        if len(data) == 3:
            ints = np.array(data[2])
        else:
            ints = np.array([1])

    x = ints * np.cos(decs) * np.cos(incs)
    y = ints * np.sin(decs) * np.cos(incs)
    z = ints * np.sin(incs)

    cart = np.array([x,y,z]).transpose()

    return cart





def cart2dir(cart):
    """
    Converts cartesian coordinates, in x,y,z, to vector directions.

    Args:
        - cart (numpy array): array of [x,y,z]

    Returns:
        - data (array): list of [dec,inc]

    Notes:
        - Adapted from pmag.py
    """

    cart = np.array(cart)
    rad = np.pi / 180

    # case if array of vectors
    if len(cart.shape) > 1:
        Xs = cart[:,0]
        Ys = cart[:,1]
        Zs = cart[:,2]

    # case if single vector
    else:
        Xs = cart[0]
        Ys = cart[1]
        Zs = cart[2]

    if np.iscomplexobj(Xs): Xs = Xs.real
    if np.iscomplexobj(Ys): Ys = Ys.real
    if np.iscomplexobj(Zs): Zs = Zs.real

     # calculate resultant vector length
    Rs = np.sqrt(Xs**2 + Ys**2 + Zs**2)

    # calculate declination taking care of correct quadrants (arctan2) and making modulo 360
    Decs = (np.arctan2(Ys,Xs) / rad) % 360

    try:
         # calculate inclination (converting to degrees)
        Incs = np.arcsin(Zs/Rs) / rad
    except:
        print('trouble in cart2dir - most likely division by zero somewhere')
        return np.zeros(3)

    return np.array([Decs,Incs,Rs]).transpose()





def fisher_mean(data):
    """
    Calculate the Fisher mean.

    Args:
        - data (array): list of [dec,inc]

    Returns:
        - fpars (dictionary): with dec, inc, n, r, k, alpha95, csd

    Notes:
        - Adapted from pmag.py
    """

    N = len(data)
    R = 0
    Xbar = [0,0,0]
    X = []
    fpars = {}

    if N < 2:
        return fpars

    X = dir2cart(data)

    for i in range(len(X)):
        for c in range(3):
            Xbar[c] = Xbar[c] + X[i][c]

    for c in range(3):
        R = R + Xbar[c]**2
    R = np.sqrt(R)

    for c in range(3):
        Xbar[c] = Xbar[c] / R

    direction = cart2dir(Xbar)

    fpars['dec'] = direction[0]
    fpars['inc'] = direction[1]
    fpars['n'] = N
    fpars['r'] = R

    if N != R:
        k = (N-1) / (N-R)
        fpars['k'] = k
        csd = 81 / np.sqrt(k)
    else:
        fpars['k'] = 'inf'
        csd = 0

    b = 20**(1/(N-1)) - 1
    a = 1 - b*(N-R)/R

    if a < -1: a = -1

    a95 = np.arccos(a)*180/np.pi
    fpars['alpha95'] = a95
    fpars['csd'] = csd

    if a < 0: fpars['alpha95'] = 180

    return fpars





def cover_calculator(csv_string):
    """
    Calculate a stratigraphic thickness between the two points.

    Args:
        - csv_string (string): path to covers .csv

    Notes:
        Input .csv must follow the format of the given template,
        'covers_template.csv', and include:
            - latitude (decimal degrees)
            - longitude (decimal degrees)
            - elevation (m)
            - strike of bedding (RHR)
            - dip of bedding

        Saves results to a .csv with name: csv_string + '_calculated.csv'

        Example to run from the command line:
        python -c 'import pyStrat; pyStrat.cover_calculator("Users/yuempark/Documents/Hongzixi_covers.csv")'
    """

    # read in the data
    data = pd.read_csv(csv_string)

    for i in range(len(data.index)):

        # vincenty distance, converted from km to m
        data.loc[i,'s'] = vincenty_inverse((data['start_lat'][i],data['start_lon'][i]),\
                                           (data['end_lat'][i],data['end_lon'][i]))
        data.loc[i,'s'] = data.loc[i,'s'] * 1000

        # compass bearing
        data.loc[i,'bearing'] = compass_bearing((data['start_lat'][i],data['start_lon'][i]),\
                                                (data['end_lat'][i],data['end_lon'][i]))

        # first convert strike to dip direction
        data.loc[i,'start_dip_direction'] = (data['start_strike'][i] + 90) % 360
        data.loc[i,'end_dip_direction'] = (data['end_strike'][i] + 90) % 360

        # convert dip direction and dip of bedding to a pole (perpendicular to bedding) trend and plunge
        data.loc[i,'start_pole_trend'] = (data['start_dip_direction'][i] + 180) % 360
        data.loc[i,'start_pole_plunge'] = 90 - data['start_dip'][i]
        data.loc[i,'end_pole_trend'] = (data['end_dip_direction'][i] + 180) % 360
        data.loc[i,'end_pole_plunge'] = 90 - data['end_dip'][i]

        # get the mean pole
        di_block = [[data['start_pole_trend'][i],data['start_pole_plunge'][i]],\
                    [data['end_pole_trend'][i],data['end_pole_plunge'][i]]]
        fpars = fisher_mean(di_block)
        data.loc[i,'mean_pole_trend'] = fpars['dec']
        data.loc[i,'mean_pole_plunge'] = fpars['inc']

        # convert back to mean dip direction and dip
        data.loc[i,'mean_dip_direction'] = (data['mean_pole_trend'][i] + 180) % 360
        data.loc[i,'mean_dip'] = 90 - data['mean_pole_plunge'][i]

        # distance perpendicular to strike
        data.loc[i,'perp_s'] = data['s'][i] * math.cos(math.radians(abs(data['mean_dip_direction'][i] -\
                                                                        data['bearing'][i])))

        # elevation difference
        data.loc[i,'d_elev'] = data['end_elev'][i] - data['start_elev'][i]

        # true perpendicular distance (accounting for elevation)
        data.loc[i,'R'] = math.sqrt(data['d_elev'][i]**2 + data['perp_s'][i]**2)

        # absolute inclination between the adjusted points
        data.loc[i,'inclination'] = math.degrees(math.atan(data['d_elev'][i] / data['perp_s'][i]))

        # the inclination from the line between the adjusted points
        if data['d_elev'][i] > 0:
            data.loc[i,'angle'] = 90 - data['inclination'][i] - data['mean_dip'][i]
        else:
            data.loc[i,'angle'] = 90 - data['mean_dip'][i] + data['inclination'][i]

        # the stratigraphic height
        data.loc[i,'HEIGHT'] = round(data['R'][i] * math.cos(math.radians(data['angle'][i])), 1)

    # save the .csv
    data.to_csv(csv_string + '_calculated.csv', index=False)

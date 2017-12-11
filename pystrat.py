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

    Parameters
    ----------
    csv_string : string
        Path to data .csv.

    Returns
    -------
    data : dataframe
        Properly formatted data.

    Notes
    -----
    The .csv must follow the form of the template.

    Lines 1-4 can be filled with arbitrary relevant information - the code will
    start reading in data at line 5.

    Line 5 MUST contain at least two headers: one of these headers MUST be named
    'THICKNESS', and one of these headers MUST be named 'FEATURES'. Other
    columns may be named whatever the user desires.
    """
    # the data
    data = pd.read_csv(csv_string, header=4)

    # get rid of blank columns
    cols = []
    for c in data.columns:
        if 'Unnamed' not in c:
            cols.append(c)
    data = data[cols]

    # return the dataframe
    return data





def read_formatting(csv_string):
    """
    Imports formatting from a .csv.

    Parameters
    ----------
    csv_string : string
        Path to formatting .csv.

    Returns
    -------
    formatting : dataframe
        Properly formatted formatting.

    Notes
    -----
    The .csv must follow the form of the template.

    Columns 1-4 are used to set the colour of the boxes: columns 1-3 must be
    called 'r', 'g', and 'b' (for red, green, and blue), and values in columns
    1-3 must be between 0-255. The header of column 4 must match one of the
    headers used in the data .csv, and all values in the data must be a subset
    of the values in this column.

    Columns 6-7 are used to set the width of the boxes: column 6 must be called
    'width'. The header of column 7 must match one of the headers used in the
    data .csv, and all values in the data must be a subset of the values in this
    column. Column 5 should be left blank for readability.
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

    Parameters
    ----------
    data : dataframe
        Properly formatted data.
    formatting : dataframe
        Properly formatted formatting.

    Returns
    -------
    None.

    Notes
    -----
    Prints result of integrity check. If fail, also prints the item which fails
    the check.
    """
    # get the colour and width headers being used
    colour_header = formatting.columns[3]
    width_header = formatting.columns[6]

    # if width_header and colour_header are the same, pandas appends a '.1'
    if width_header.endswith('.1'):
        width_header = width_header[:-2]

    # failed items
    colour_failed = []
    width_failed = []

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

            # only print warning if the check fails for the first time
            if colour_check == False:
                if data[colour_header][i] not in colour_failed:
                    print('Colour check failed for item ' + str(data[colour_header][i]) + '.')
                    colour_failed.append(data[colour_header][i])
                all_check = False
            if width_check == False:
                if data[width_header][i] not in width_failed:
                    print('Width check failed for item ' + str(data[width_header][i]) + '.')
                    width_failed.append(data[width_header][i])
                all_check = False

    # print an all clear statement if the check passes
    if all_check:
        print('Colour and width check passed.')





def plot_legend(formatting):
    """
    Plot all items in the formatting.

    Parameters
    ----------
    formatting : dataframe
        Properly formatted formatting.

    Returns
    -------
    fig : figure
        Figure handle.
    ax : axis
        Axis handles.
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

    # if they aren't the same...
    else:
        # initialize the figure
        fig, ax = plt.subplots(nrows=1, ncols=4, gridspec_kw={'width_ratios':[1.5,1,1.5,1]})

            # now the widths
        height_width = 0
        for i in range(len(formatting.index)):

            # only plot the non-nan cells
            if pd.notnull(formatting[width_header][i]):

                # get the width and label
                this_width = formatting['width'][i]
                this_label = formatting[width_header][i]

                # create the rectangle
                ax[0].add_patch(patches.Rectangle((0.0,height_width), this_width, 1, facecolor='white', edgecolor='k'))

                # the label
                ax[1].text(0.1, height_width+0.5, this_label, verticalalignment='center')

                # count the height
                height_width = height_width + 1

        # prettify the width legend
        ax[0].set_ylim(0,height_width)
        ax[1].set_ylim(0,height_width)
        ax[0].set_yticks([])
        ax[1].set_yticks([])
        ax[1].set_xticks([])
        ax[0].set_title('WIDTH')
        ax[1].set_title(width_header)

        # first the colours
        height_colours = 0
        for i in range(len(formatting.index)):

            # only plot non-nan cells
            if pd.notnull(formatting[colour_header][i]):

                # get the colour and label
                this_colour = [formatting['r'][i], formatting['g'][i], formatting['b'][i]]
                this_label = formatting[colour_header][i]

                # create the rectangle
                ax[2].add_patch(patches.Rectangle((0.0,height_colours), 1, 1, facecolor=this_colour, edgecolor='k'))

                # the label
                ax[3].text(0.1, height_colours+0.5, this_label, verticalalignment='center')

                # count the height
                height_colours = height_colours + 1

        # prettify colour legend
        ax[2].set_ylim(0,height_colours)
        ax[3].set_ylim(0,height_colours)
        ax[2].set_yticks([])
        ax[3].set_yticks([])
        ax[2].set_xticks([])
        ax[3].set_xticks([])
        ax[2].set_title('COLOUR')
        ax[3].set_title(colour_header)

        # force the size of the plot
        ratio = 0.3
        if height_width > height_colours:
            fig.set_figheight(height_width * ratio)
        else:
            fig.set_figheight(height_colours * ratio)
        fig.set_figwidth(10)

        return fig, ax





def initiate_figure(data, formatting, strat_ratio, figwidth, width_ratios, linewidth=1, features=True):
    """
    Initiate a figure with the stratigraphic column.

    Parameters
    ----------
    data : dataframe
        Properly formatted data.
    formatting : dataframe
        Properly formatted formatting.
    strat_ratio : float or int
        Scaling ratio for height of figure. If used for a different
        stratigraphic column, they will have the same scale.
    figwidth : float or int
        Figure width.
    width_ratios : array_like
        Width ratios of axes. The number of elements in this array_like defines
        the number of axes.
    linewidth : float or int, optional
        Line width (default = 1).
    features : boolean, optional
        If True, print feature notes (default = True).

    Returns
    -------
    fig : figure
        Figure handle.
    ax : axis
        Axis handles.
    """
    # get the number of axes from the width_ratios list
    ncols = len(width_ratios)

    # initiate fig and ax
    fig, ax = plt.subplots(nrows=1, ncols=ncols, sharey=True, gridspec_kw={'width_ratios':width_ratios})

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
            ax[0].add_patch(patches.Rectangle((0.0,strat_height), this_width, this_thickness, facecolor=this_colour, edgecolor='k', linewidth=linewidth))

            # if there are any features to be labelled, label it
            if features:
                if pd.notnull(data['FEATURES'][i]):
                    ax[1].text(0.02, strat_height + (this_thickness/2), data['FEATURES'][i], horizontalalignment='left', verticalalignment='center')

            # count the stratigraphic height
            strat_height = strat_height + this_thickness

    # force the limits on the lithostratigraphy plot
    ax[0].set_xlim([0,1])
    ax[0].set_ylim([0,strat_height])

    # force the size of the plot
    fig.set_figheight(strat_height * strat_ratio)
    fig.set_figwidth(figwidth)

    # prettify
    ax[0].set_ylabel('stratigraphic height [m]')
    ax[0].set_xticklabels([])
    ax[0].set_xticks([])

    if features:
        ax[1].set_xticklabels([])
        ax[1].set_xticks([])

    return fig, ax





def add_data_axis(fig, ax, ax_num, x, y, plot_type, **kwargs):
    """
    Add an arbitrary data axis to a figure initiated by initiate_figure.

    Parameters
    ----------
    fig : figure
        Figure handle initiated by initiate_figure.
    ax : axis
        Axis handles initiated by initiate_figure.
    ax_num : int
        Axis on which to plot.
    x : array_like
        x-data.
    y : array_like
        y-data.
    plot_type : string
        'plot', 'scatter', or 'barh'.

    Other Parameters
    ----------------
    **kwargs passed to plt.plot, plt.scatter, or plt.barh.

    Returns
    -------
    None.

    Notes
    -----
    If 'barh' is selected, it is recommended that a 'height' kwarg be passed to
    this function (see the documentation for matplotlib.pyplot.barh). Also, note
    that 'y' becomes the bottom-left coordinate of the bar.
    """
    # get the correct axis
    ax = ax[ax_num]

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

    Parameters
    ----------
    data : dataframe
        Properly formatted data.
    recorded_height : array_like
        Recorded height of samples.
    remarks : array_like
        Field addition errors noted as "ADD X" or "SUB X" in the remarks array,
        and only need to be noted at the first sample where the correction comes
        into effect.

    Returns
    -------
    sample_info : dataframe
        With columns: recorded_height, remarks, height, and unit.

    Notes
    -----
    Samples marked with a .5 in the 'unit' column come from unit boundaries.
    User input is required to assign the correct unit.

    If the sample comes from the lower unit, subtract 0.5. If the sample comes
    from the upper unit, add 0.5

    After fixing all changes, copy and paste the dataframe into the data .csv.

    A useful masking command to only show samples on unit boundaries in jupyter
    notebook:

    >> mask = (sample_info['unit'] != np.floor(sample_info['unit']))
    >> sample_info[mask]

    Units are zero indexed.
    """
    # remove nans from the recorded_height array (appended to the end) and corresponding remarks
    no_nans = np.array([])
    for i in range(len(recorded_height)):
        if np.isfinite(recorded_height[i]):
            no_nans = np.append(no_nans, recorded_height[i])
    recorded_height = no_nans
    remarks = remarks[:len(recorded_height)]

    # make sure the sample list is sorted
    if np.array_equal(np.sort(recorded_height), recorded_height):
        pass
    else:
        print('WARNING: Samples were out of order in the data file. Please check')
        print('         that samples were not logged incorrectly.')
        sort_ind = recorded_height.argsort()
        recorded_height = recorded_height[sort_ind]
        remarks = remarks[sort_ind]

    # calculate the true height of the samples
    height = np.array([])
    adjustment = 0.0
    for i in range(len(recorded_height)):
        if 'ADD' in str(remarks[i]):
            value = float(remarks[i].split()[1])
            adjustment = adjustment + value
        elif 'SUB' in str(remarks[i]):
            value = float(remarks[i].split()[1])
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

        # stop once the strat_height is equal to or greater than the sample height (with some tolerance for computer weirdness)
        while strat_height[unit_ind] < (height[i] - 1e-5):
            unit_ind = unit_ind + 1

        # if the sample comes from a unit boundary, mark with .5 (with some tolerance for computer weirdness)
        if abs(strat_height[unit_ind] - height[i]) < 1e-5:
            unit = np.append(unit, unit_ind + 0.5)
        else:
            unit = np.append(unit, unit_ind)

    # create the output dataframe
    sample_info = pd.DataFrame({'recorded_height':recorded_height, 'remarks':remarks, 'height':height, 'unit':unit})

    # clean up the dataframe
    cols = ['recorded_height', 'remarks', 'height', 'unit']
    sample_info = sample_info[cols]
    sample_info.reset_index(inplace=True, drop=True)

    return sample_info





def print_unit_edit_code(sample_info, variable_string):
    """
    Prints the code for editing units.

    Parameters
    ----------
    sample_info : dataframe
        With columns: recorded_height, remarks, height, and unit.
    variable_string : string
        String of the variable name for sample_info.

    Returns
    -------
    None.

    Notes
    -----
    Copy and paste the printed code into a cell and edit as follows:

    If the sample comes from the lower unit, subtract 0.5. If the sample comes
    from the upper unit, add 0.5

    After fixing all changes, copy and paste the dataframe into the data .csv.
    """
    # get the samples on unit boundaries
    mask = (sample_info['unit'] != np.floor(sample_info['unit']))
    boundary_sample_info = sample_info[mask]

    print('=====')

    # the case where there are no samples on units boundaries
    if len(boundary_sample_info.index) == 0:
        print('No samples are on unit boundaries - no manual edits required.')

    else:

        # print the code, with some formatting
        if np.max(boundary_sample_info.index.values) < 10:
            for index in boundary_sample_info.index.values:
                print(variable_string + '.loc[' + str(index) + ",'unit'] = " + str(sample_info['unit'][index]))

        elif np.max(boundary_sample_info.index.values) < 100:
            for index in boundary_sample_info.index.values:
                if index < 10:
                    print(variable_string + '.loc[ ' + str(index) + ",'unit'] = " + str(sample_info['unit'][index]))
                else:
                    print(variable_string + '.loc[' + str(index) + ",'unit'] = " + str(sample_info['unit'][index]))

        elif np.max(boundary_sample_info.index.values) < 1000:
            for index in boundary_sample_info.index.values:
                if index < 10:
                    print(variable_string + '.loc[  ' + str(index) + ",'unit'] = " + str(sample_info['unit'][index]))
                elif index < 100:
                    print(variable_string + '.loc[ ' + str(index) + ",'unit'] = " + str(sample_info['unit'][index]))
                else:
                    print(variable_string + '.loc[' + str(index) + ",'unit'] = " + str(sample_info['unit'][index]))

        else:
            for index in boundary_sample_info.index.values:
                if index < 10:
                    print(variable_string + '.loc[   ' + str(index) + ",'unit'] = " + str(sample_info['unit'][index]))
                elif index < 100:
                    print(variable_string + '.loc[  ' + str(index) + ",'unit'] = " + str(sample_info['unit'][index]))
                elif index < 1000:
                    print(variable_string + '.loc[ ' + str(index) + ",'unit'] = " + str(sample_info['unit'][index]))
                else:
                    print(variable_string + '.loc[' + str(index) + ",'unit'] = " + str(sample_info['unit'][index]))

    print('=====')





def get_total_thickness(data):
    """
    Returns the total stratigraphic thickness.

    Parameters
    ----------
    data : dataframe
        Properly formatted data.

    Returns
    -------
    thickness : float
        The total stratigraphic thickness.

    Notes
    -----
    Output is rounded to two decimal places.
    """
    thickness = np.round(np.sum(data['THICKNESS']),2)

    return thickness





###########################################
########## CALCULATION FUNCTIONS ##########
###########################################





def lowess_fit(height, val, frac=0.6666666666666666, it=3):
    """
    LOWESS fit for a scatterplot.

    Parameters
    ----------
    height : array_like
        Sample heights.
    val : array_like
        Sample values.
    frac : float, optional
        Between 0 and 1 - the fraction of the data used when estimating each
        y-value (default = 0.6666666666666666).
    it : int, optional
        The number of residual-based reweightings to perform (default = 3).

    Returns
    -------
    height_LOWESS : array
        Sorted heights.
    val_LOWESS : array
        Estimated values.

    Notes
    -----
    height_LOWESS and val_LOWESS arrays will be sorted by height, and any
    duplicate values are removed.
    """
    # the LOWESS fit
    xy_all = sm.nonparametric.lowess(val, height, frac=frac, it=it)

    # remove duplicate values
    height_LOWESS = np.array([xy_all[0,0]])
    val_LOWESS = np.array([xy_all[0,1]])
    for i in range(1,len(xy_all)):
        if xy_all[i,0] != xy_all[i-1,0]:
            height_LOWESS = np.append(height_LOWESS, xy_all[i,0])
            val_LOWESS = np.append(val_LOWESS, xy_all[i,1])

    return height_LOWESS, val_LOWESS





def lowess_normalize(height, val, frac=0.6666666666666666, it=3):
    """
    Normalize values to the LOWESS fit.

    Parameters
    ----------
    height : array_like
        Sample heights.
    val : array_like
        Sample values.
    frac : float, optional
        Between 0 and 1 - the fraction of the data used when estimating each
        y-value (default = 0.6666666666666666).
    it : int, optional
        The number of residual-based reweightings to perform (default = 3).

    Returns:
    height_LOWESS : array
        Sorted heights.
    val_LOWESS : array
        Estimated values.
    val_norm : array
        Normalized values.

    Notes
    -----
    height_LOWESS and val_LOWESS arrays will be sorted by height, and any
    duplicate values are removed.

    val_norm will match the input val array - order and NaN's will be preserved.
    """
    # do the LOWESS fit
    height_LOWESS, val_LOWESS = lowess_fit(height, val, frac, it)

    # remove index on input arrays, if there are any
    temp_height = np.array([])
    temp_val = np.array([])
    for height_i in height:
        temp_height = np.append(temp_height, height_i)
    for val_i in val:
        temp_val = np.append(temp_val, val_i)
    height = temp_height
    val = temp_val

    # normalize
    val_norm = np.array([])
    for i in range(len(height)):
        for j in range(len(height_LOWESS)):
            if height[i] == height_LOWESS[j]:
                val_norm = np.append(val_norm, val[i] - val_LOWESS[j])

    return height_LOWESS, val_LOWESS, val_norm





def scatter_variance(strat_height, vals, interval, mode, frac=0.6666666666666666, it=3):
    """
    Calculate the variance in scatterplot data.

    Parameters
    ----------
    strat_height : array_like
        Stratigraphic heights of samples.
    vals : array_like
        Values of samples.
    interval : float or int
        Window height used to calculate variance, in data coordinates.
    mode : string
        'standard' (raw data), 'window_normalized' (windowed lowess fit
        subtracted), or 'all_normalized' (lowess fit subtracted)
    frac : float, optional
        Between 0 and 1 - the fraction of the data used when estimating each
        y-value (default = 0.6666666666666666).
    it : int, optional
        The number of residual-based reweightings to perform (default = 3).

    Returns
    -------
    variances : array
        Calculated variances.
    strat_height_mids : array
        Middle of windows used.

    The following are only returned in 'window_normalized' or 'all_normalized'
    modes.
    xys : array
        The first column is the sorted x values and the second column the
        associated estimated y-values.
    norm_vals : array
        Normalized values.
    norm_heights : array
        Associated height values.
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
    kilometers or miles) between two points on the surface of a spheroid.

    Parameters
    ----------
    point1 : tuple
        The latitude/longitude for the first point. Latitude and longitude must
        be in decimal degrees.
    point2 : tuple
        The latitude/longitude for the second point. Latitude and longitude must
        be in decimal degrees.
    miles : boolean, optional
        If false, use kilometres (default = False).

    Returns
    -------
    d : float
        Calculated distance.

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

    Parameters
    ----------
    pointA : tuple
        The latitude/longitude for the first point. Latitude and longitude must
        be in decimal degrees.
    pointB : tuple
        The latitude/longitude for the second point. Latitude and longitude must
        be in decimal degrees.

    Returns
    -------
    compass_bearing : float
        The bearing in degrees.

    Notes
    -----
    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))

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

    Parameters
    ----------
    data : list
        List of [dec,inc].

    Returns
    -------
    cart : array
        Array of [x,y,z].

    Notes
    -----
    Adapted from pmag.py.
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

    Parameters
    ----------
    cart : array
        Array of [x,y,z].

    Returns
    -------
    data : list
        List of [dec,inc].

    Notes
    -----
    Adapted from pmag.py.
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

    Parameters
    ----------
    data : list
        List of [dec,inc].

    Returns
    -------
    fpars : dict
        With dec, inc, n, r, k, alpha95, csd.

    Notes
    -----
    Adapted from pmag.py.
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





def calculate_stratigraphic_thickness(lat, lon, elev, strike, dip):
    """
    Calculate a stratigraphic thickness between two points.

    Parameters
    ----------
    lat : tuple
        Start and end latitude, in decimal degrees.
    lon : tuple
        Start and end longitude, in decimal degrees.
    elev : tuple
        Start and end elevation, in metres.
    strike : tuple
        Start and end strike, in decimal degrees.
    dip : tuple
        Start and end dip, in decimal degrees.

    Returns
    -------
    d : float
        Calculated stratigraphic thickness.
    """
    # vincenty distance, converted from km to m
    s = vincenty_inverse((lat[0],lon[0]), (lat[1],lon[1]))
    s = s * 1000

    # compass bearing
    bearing = compass_bearing((lat[0],lon[0]), (lat[1],lon[1]))

    # first convert strike to dip direction
    dip_dir = ((strike[0]+90)%360, (strike[1]+90)%360)

    # convert dip direction and dip of bedding to a pole (perpendicular to bedding) trend and plunge
    pole_trend = ((dip_dir[0]+180)%360, (dip_dir[1]+180)%360)
    pole_plunge = (90-dip[0], 90-dip[1])

    # get the mean pole
    di_block = [[pole_trend[0],pole_plunge[0]], [pole_trend[1],pole_plunge[1]]]
    fpars = fisher_mean(di_block)
    mean_pole_trend = fpars['dec']
    mean_pole_plunge = fpars['inc']

    # convert back to mean dip direction and dip
    mean_dip_dir = (mean_pole_trend + 180) % 360
    mean_dip = 90 - mean_pole_plunge

    # distance perpendicular to strike
    perp_s = s * math.cos(math.radians(abs(mean_dip_dir - bearing)))

    # elevation difference
    d_elev = elev[1] - elev[0]

    # true perpendicular distance (accounting for elevation)
    R = math.sqrt(d_elev**2 + perp_s**2)

    # absolute inclination between the adjusted points
    inclination = math.degrees(math.atan(d_elev / perp_s))

    # the inclination from the line between the adjusted points
    if d_elev > 0:
        angle = 90 - inclination - mean_dip
    else:
        angle = 90 - mean_dip + inclination

    # the stratigraphic height
    d = R * math.cos(math.radians(angle))

    return d





def calculate_stratigraphic_thickness_csv(csv_string):
    """
    Calculate a stratigraphic thickness between two points from a .csv.

    Parameters
    ----------
    csv_string : string
        Path to covers .csv, with the proper formatting.

    Returns
    -------
    data : dataframe
        Dataframe with calculated fields.

    Notes
    -----
    Input .csv must follow the format of the given template,
    'covers_template.csv', and include: latitude (decimal degrees), longitude
    (decimal degrees), elevation (m), strike of bedding (RHR), dip of bedding
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

    return data

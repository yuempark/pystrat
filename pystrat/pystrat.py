########################################################################
############################ PACKAGE DESIGN ############################
########################################################################

# pystrat strives to take advantage of object oriented programming (OOP)
# by organizing stratigraphic data into classes.

# The core class upon which this package is built is the Section class,
# which organizes the measured stratigraphic log (i.e. the thicknesses
# and facies of units).

# Any additional data that is tied to the stratigraphic height, but not
# explicitly tied to the individually measured units, is organized in
# the Data subclass.

# The plotting style for a stratigraphic section is organized in the
# Style class.

########################################################################

# import standard modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import additional modules
import warnings
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import Divider, Size
from PIL import Image

###############
### CLASSES ###
###############

class Fence:
    """
    Organizes sections according to a shared datum
    """

    def __init__(self, sections, datums=None):
        """
        Initialize Section with the two primary attributes.

        Parameters
        ----------
        sections : 1d array_like
            List of Sections to be put into the fence
        
        datums : 1d array_like
            If not specified, the datum for each section will be the bottom. If specified, must be list of same length as number of sections with heights in each section for the datum.
        """
        self.n_sections = len(sections)
        self.sections = sections
        if datums == None:
            datums = np.zeros(self.n_sections) 
            for ii in range(self.n_sections):
                datums[ii] = sections[ii].base_height
        else:
            assert len(datums) == self.n_sections, 'Number of datums should equal number of sections'
        self.datums = datums


class Section:
    """
    Organizes all data associated with a single stratigraphic section.
    """

    def __init__(self, thicknesses, facies):
        """
        Initialize Section with the two primary attributes.

        Parameters
        ----------
        thicknesses : 1d array_like
            Thicknesses of the facies. Any NaNs will be automatically
            removed.

        facies : 1d array_like
            Observed facies. Any NaNs will be automatically removed.
        """
        # convert to arrays and check the dimensionality
        thicknesses = attribute_convert_and_check(thicknesses)
        facies = attribute_convert_and_check(facies)

        # check that the thicknesses are numeric
        if thicknesses.dtype == np.object:
            raise Exception('Thickness data must be floats or ints.')

        # check for NaNs, and get rid of them
        thicknesses_nan_mask = np.isnan(thicknesses)
        if np.sum(thicknesses_nan_mask) > 0:
            warnings.warn('Thickness data contains NaNs. These rows will be '
                          'automatically removed, but you should check to make '
                          'sure that this is appropriate for your dataset.')
            thicknesses = thicknesses[~thicknesses_nan_mask]

        facies_nan_mask = pd.isnull(facies)
        if np.sum(facies_nan_mask) > 0:
            warnings.warn('Facies data contains NaNs. These rows will be '
                          'automatically removed, but you should check to make '
                          'sure that this is appropriate for your dataset.')
            facies = facies[~facies_nan_mask]

        # check that the length of the thicknesses and facies match
        n_thicknesses_units = len(thicknesses)
        n_facies_units = len(facies)
        if n_thicknesses_units != n_facies_units:
            raise Exception('Length of thicknesses and facies data are not '
                            'equal. If a warning was raised regarding the '
                            'presence of NaNs in your data, their presence '
                            'may be the cause of this mismatch.')

        # assign the core data to attributes
        self.thicknesses = thicknesses
        self.facies = facies

        # add some other facies attributes
        self.top_height = np.cumsum(thicknesses)
        self.base_height = np.cumsum(thicknesses) - thicknesses
        self.unit_number = np.arange(n_thicknesses_units)

        # add some generic attributes
        self.n_units = n_thicknesses_units
        self.total_thickness = np.sum(thicknesses)
        self.unique_facies = np.unique(facies)
        self.n_unique_facies = len(np.unique(facies))

        # keep track of attributes
        self.facies_attributes = ['unit_number','thicknesses','base_height','top_height','facies']
        self.generic_attributes = ['n_units','total_thickness','unique_facies',
                                   'n_unique_facies']
        self.data_attributes = []

    def add_facies_attribute(self, attribute_name, attribute_values):
        """
        Add an attribute that is explicitly tied to the stratigraphic
        units.

        This function should only be used to add an attribute which
        corresponds 1:1 with each measured unit. Typically, such
        attributes would add additional detail (e.g. grain size or
        bedforms) to the broad facies description of the unit.

        To add an attribute that is tied to the stratigraphic height,
        but not explicitly tied to the stratigraphic units (e.g.
        chemostratigraphic data), use the Data subclass via the method
        `add_data_attribute()`.

        To add an attribute that is neither tied to the stratigraphic
        height nor the stratigraphic units (e.g. the GPS coordinates of
        the start and end of the section), use
        `add_generic_attribute()`.

        Parameters
        ----------
        attribute_name : string
            The name of the attribute.

        attribute_values : 1d array_like
            The attribute values. NaNs are accepted.
        """
        # convert to arrays and check the dimensionality
        attribute_values = attribute_convert_and_check(attribute_values)

        # check that the length of the attribute data matches the number of units
        n_attribute_units = len(attribute_values)
        if n_attribute_units != self.n_units:
            raise Exception('Length of thicknesses and attribute data are not '
                            'equal.')

        # assign the data to the object
        setattr(self, attribute_name, attribute_values)

        # keep track of the attribute
        facies_attributes = self.facies_attributes
        facies_attributes.append(attribute_name)
        setattr(self, 'facies_attributes', facies_attributes)

    def return_facies_dataframe(self):
        """
        Return a Pandas DataFrame of all attributes associated with the
        stratigraphic units.

        Parameters
        ----------
        None.

        Returns
        -------
        df : DataFrame
            Pandas DataFrame with all attributes associated with the
            stratigraphic units.
        """
        df = pd.DataFrame({'unit_number':np.arange(self.n_units)})

        for attribute in self.facies_attributes:
            df[attribute] = getattr(self, attribute)

        return df

    def add_generic_attribute(self, attribute_name, attribute_values):
        """
        Add an attribute that is neither tied to the stratigraphic
        height nor the stratigraphic units.

        To add an attribute that is explicitly tied to the stratigraphic
        units, use `add_facies_attribute()`.

        To add an attribute that is tied to the stratigraphic height,
        but not explicitly tied to the stratigraphic units (e.g.
        chemostratigraphic data), use the Data subclass via the method
        `add_data_attribute()`.

        Parameters
        ----------
        attribute_name : string
            The name of the attribute.

        attribute_values : any
            The attribute values.
        """
        # convert from pandas series to arrays if necessary
        if type(attribute_values) == pd.core.series.Series:
            attribute_values = attribute.values

        # assign the data to the object
        setattr(self, attribute_name, attribute_values)

        # keep track of the attribute
        generic_attributes = self.generic_attributes
        generic_attributes.append(attribute_name)
        setattr(self, 'generic_attributes', generic_attributes)

    def add_data_attribute(self, attribute_name, attribute_height, attribute_values):
        """
        Add an attribute that is tied to the stratigraphic height, but
        not explicitly tied to the stratigraphic units.

        A typical example of such data would be chemostratigraphic data.

        Parameters
        ----------
        attribute_name : string
            The name of the attribute.

        attribute_height : 1d array_like
            The stratigraphic heights at which the attribute were
            generated.

        attribute_values : 1d array_like
            The attribute values.
        """
        setattr(self, attribute_name, self.Data(attribute_height, attribute_values))

        # keep track of the attribute
        data_attributes = self.data_attributes
        data_attributes.append(attribute_name)
        setattr(self, 'data_attributes', data_attributes)

    class Data:
        """
        This nested class stores any data tied to the stratigraphic
        height, but not explicitly tied to the stratigraphic units.

        A typical example of such data would be chemostratigraphic data.
        """

        def __init__(self, attribute_height, attribute_values):
            """
            Initialize Data with the two primary attributes.

            Parameters
            ----------
            attribute_height : 1d array_like
                The stratigraphic heights at which the attribute were
                generated.

            attribute_values : 1d array_like
                The attribute values.
            """
            # convert to arrays and check the dimensionality
            attribute_height = attribute_convert_and_check(attribute_height)
            attribute_values = attribute_convert_and_check(attribute_values)

            # check that the heights are numeric
            if attribute_height.dtype == np.object:
                raise Exception('Height data must be floats or ints.')

            # check that the X and Y have the same length
            n_height = len(attribute_height)
            n_values = len(attribute_values)
            if n_height != n_values:
                raise Exception('Length of X and Y data are not equal.')

            # assign the attributes
            self.height = attribute_height
            self.values = attribute_values

            # add some other useful attributes
            self.n_values = n_values

            # keep track of attributes
            self.height_attributes = ['height','values']
            self.generic_attributes = ['n_values']

        def add_height_attribute(self, attribute_name, attribute_values):
            """
            Add an attribute that is tied to the stratigraphic heights
            of this instance of Data.

            A typical example would be the sample names associated with
            a given chemostratigraphic profile.

            Parameters
            ----------
            attribute_name : string
                The name of the attribute.

            attribute_values : 1d array_like
                The attribute values. NaNs are accepted.
            """
            # convert to arrays and check the dimensionality
            attribute_values = attribute_convert_and_check(attribute_values)

            # check that the length of the attribute data matches the number of values
            n_attribute_units = len(attribute_values)
            if n_attribute_units != self.n_values:
                raise Exception('Length of heights and attribute data are not '
                                'equal.')

            # assign the data to the object
            setattr(self, attribute_name, attribute_values)

            # keep track of the attribute
            height_attributes = self.height_attributes
            height_attributes.append(attribute_name)
            setattr(self, 'height_attributes', height_attributes)

        def return_data_dataframe(self):
            """
            Return a Pandas DataFrame of all attributes associated with
            this Data object.

            Parameters
            ----------
            None.

            Returns
            -------
            df : DataFrame
                Pandas DataFrame with all attributes associated with
                this Data object,
            """
            df = pd.DataFrame({'height':self.height})

            for attribute in self.height_attributes:
                df[attribute] = getattr(self, attribute)

            return df

        def add_data_facies(self, section):
            """
            Extract the facies associated with each data point.

            Parameters
            ----------
            section : Section
                The parent Section object for this Data object.
                (While the parent Section object is already implied for any
                given Data object, accessing the Section object's attributes
                is not elegant, so we pass it as an argument for now...)

            Notes
            -----
            Samples marked with a .5 in the 'unit' column come from unit boundaries.
            User input is required to assign the correct unit.

            If the sample comes from the lower unit, subtract 0.5. If the sample comes
            from the upper unit, add 0.5.
            """
            # make sure the data points are sorted
            if not np.array_equal(np.sort(self.height), self.height):
                raise Exception('Stratigraphic heights were out of order. '
                                'Please check that the data were not logged incorrectly, '
                                'and sort by stratigraphic height before using '
                                'add_data_attribute().')

            # calculate the unit from which the sample came
            facies = []
            unit_number = []
            for i in range(len(self.height)):
                unit_ind = 0

                # stop once the base_height is equal to or greater than the sample height
                # (with some tolerance for computer weirdness)
                while section.top_height[unit_ind] < (self.height[i] - 1e-5):
                    unit_ind = unit_ind + 1

                # if the sample comes from a unit boundary, mark with .5
                # (with some tolerance for computer weirdness)
                if abs(section.top_height[unit_ind] - self.height[i]) < 1e-5:
                    unit_number.append(unit_ind + 0.5)
                    facies.append('!!!ON BOUNDARY!!!')
                else:
                    unit_number.append(unit_ind)
                    facies.append(section.facies[unit_ind])

            # assign the data to the object
            setattr(self, 'facies', facies)
            setattr(self, 'unit_number', unit_number)

            # keep track of the attributes
            height_attributes = self.height_attributes
            height_attributes.append('facies')
            height_attributes.append('unit_number')
            setattr(self, 'height_attributes', height_attributes)

        def clean_data_facies_helper(self, data_name):
            """
            Prints the code for cleaning up data points on unit boundaries.

            Parameters
            ----------
            data_name : string
                Variable name used for the Data object (e.g. 'section_01.data_01').

            Notes
            -----
            Copy and paste the printed code into a cell and edit as follows:

            If the sample comes from the lower unit, subtract 0.5. If the sample comes
            from the upper unit, add 0.5

            After fixing all changes, copy and paste the dataframe into the data .csv.
            """
            # check that add_data_facies has been run
            if 'facies' not in self.height_attributes:
                raise Exception('Data has not been assigned a facies. '
                                'Data.add_data_facies() will assign facies and '
                                'unit numbers automatically.')
            if 'unit_number' not in self.height_attributes:
                raise Exception('Data has not been assigned a unit_number. '
                                'Data.add_data_facies() will assign facies and '
                                'unit numbers automatically.')

            # get the dataframe
            df = self.return_data_dataframe()

            # get the samples on unit boundaries
            mask = df['unit_number'] != np.floor(df['unit_number'])
            df_slice = df[mask]

            # the case where there are no samples on units boundaries
            if len(df_slice.index) == 0:
                print('No samples are on unit boundaries - no manual edits required.')

            # print the code
            else:
                print('1) Copy and paste the code below into a cell and edit as follows:')
                print('- If the sample comes from the lower unit, subtract 0.5.')
                print('- If the sample comes from the upper unit, add 0.5.')
                print('')
                print('2) Run Data.clean_data_facies().')
                print('===')
                for i in df_slice.index:
                    print(data_name + '.unit_number[' + str(i) + '] = ' + str(df_slice['unit_number'][i]) +
                          ' #height = ' + str(df_slice['height'][i]) + ', sample = ' + str(df_slice['sample'][i]))

        def clean_data_facies(self, section):
            """
            Extract the facies associated with each data point AFTER cleaning up
            the samples that are from unit boundaries using Data.clean_data_facies_helper().

            Parameters
            ----------
            section : Section
                The parent Section object for this Data object.
                (While the parent Section object is already implied for any
                given Data object, accessing the Section object's attributes
                is not elegant, so we pass it as an argument for now...)
            """
            # convert unit_number to int
            self.unit_number = np.array(self.unit_number, dtype=int)

            # reassign facies
            for i in range(len(self.unit_number)):
                self.facies[i] = section.facies[self.unit_number[i]]

class Style():
    """
    Organizes the plotting style for the lithostratigraphy.
    """

    def __init__(self,
                 color_attribute, color_labels, color_values,
                 width_attribute, width_labels, width_values,
                 height_scaling_factor, width_inches):
        """
        Initialize Style with the seven required attributes.

        Note that compatibility of a Style with a Section is not checked
        until explicitly called, or plotting is attempted.

        Parameters
        ----------
        color_attribute : string
            Attribute name from which the color labels are derived. When
            plotting a Section, the Section must have this attribute.

        color_labels : 1d array_like
            The labels to which colors are assigned. When plotting a
            Section, values within the color_attribute of that Section
            must form a subset of the values within this array_like.

        color_values : array_like
            The colors that will be assigned to the associated labels.
            Values must be interpretable by matplotlib.

        width_attribute : string
            Attribute name from which the width labels are derived. When
            plotting a Section, the Section must have this attribute.

        width_labels : 1d array_like
            The labels to which widths are assigned. When plotting a
            Section, values within the width_attribute of that Section
            must form a subset of the values within this array_like.

        width_values : 1d array_like of floats
            The widths that will be assigned to the associated labels.
            Values must be between 0 and 1.

        height_scaling_factor : float
            When plotting, the height of the section will have a height
            in inches that is:
                Section.total_thickness * Style.height_scaling_factor

        width_inches : float
            When plotting, a width value of 1 will be this many inches
            wide.
        """
        # convert to arrays and check the dimensionality
        color_labels = attribute_convert_and_check(color_labels)
        width_labels = attribute_convert_and_check(width_labels)
        width_values = attribute_convert_and_check(width_values)

        # convert from pandas series/dataframes to arrays if necessary
        if type(color_values) == pd.core.series.Series:
            color_values = color_values.values
        if type(color_values) == pd.core.frame.DataFrame:
            color_values = color_values.values

        # check that the widths are between 0 and 1
        if np.max(width_values)>1 or np.min(width_values)<0:
            raise Exception('Width values must be floats between 0 and 1.')

        # assign the attributes
        self.color_attribute = color_attribute
        self.color_labels = color_labels
        self.color_values = color_values
        self.width_attribute = width_attribute
        self.width_labels = width_labels
        self.width_values = width_values
        self.height_scaling_factor = height_scaling_factor
        self.width_inches = width_inches

        # add some other useful attributes
        self.n_color_labels = len(color_labels)
        self.n_width_labels = len(width_labels)

    def plot_legend(self, legend_unit_height=0.25):
        """
        Plot a legend for this Style object.

        If the color and width labels are the same, a single legend will
        be created. Otherwise, two legends will be created - one with
        the color labels, and the other with the width labels.

        Parameters
        ----------
        legend_unit_height : float
            A scaling factor to modify the height of each unit in the
            legend only.

        Returns
        -------
        fig : matplotlib Figure
            Figure handle.

        ax : matplotlib Axes
            Axis handle.
        """
        # print some plotting values
        print('stratigraphic height scaling : 1 distance unit = 1 inch * {}'.format(self.height_scaling_factor))
        print('width value of 1 will be     : {} inches'.format(self.width_inches))

        # extract attributes
        color_labels = self.color_labels
        width_labels = self.width_labels
        color_values = self.color_values
        width_values = self.width_values

        # if the color and width labels are different
        if np.any(~(color_labels == width_labels)):

            # sort the widths
            width_sort_inds = np.argsort(width_values)
            width_labels = width_labels[width_sort_inds]
            width_values = width_values[width_sort_inds]

            # initialize the figure
            fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)

            # determine the axis height and limits first
            if self.n_color_labels > self.n_width_labels:
                ax_height = legend_unit_height * self.n_color_labels
                ax[0].set_ylim(0, self.n_color_labels)
                ax[1].set_ylim(0, self.n_color_labels)
            else:
                ax_height = legend_unit_height * self.n_width_labels
                ax[0].set_ylim(0, self.n_width_labels)
                ax[1].set_ylim(0, self.n_width_labels)

            # plot the colors
            strat_height_colors = 0

            for i in range(len(color_labels)):

                # create the rectangle - with thickness of 1
                ax[0].add_patch(Rectangle((0.0,strat_height_colors), 1,
                                          1, facecolor=color_values[i], edgecolor='k'))

                # label the unit
                ax[0].text(1.2, strat_height_colors+0.5, color_labels[i],
                           horizontalalignment='left', verticalalignment='center')

                # count the height
                strat_height_colors = strat_height_colors + 1

            # set the axis dimensions (values below are all in inches)
            ax0_lower_left_x = 0.5
            ax0_lower_left_y = 0.5
            ax0_width = 0.5
            h = [Size.Fixed(ax0_lower_left_x), Size.Fixed(ax0_width)]
            v = [Size.Fixed(ax0_lower_left_y), Size.Fixed(ax_height)]
            divider = Divider(fig, (0.0, 0.0, 1., 1.), h, v, aspect=False)
            ax[0].set_axes_locator(divider.new_locator(nx=1, ny=1))

            # set the limits
            ax[0].set_xlim(0,1)

            # prettify
            ax[0].set_xticks([])
            ax[0].set_yticklabels([])
            ax[0].set_yticks([])

            # plot the widths
            strat_height_widths = 0

            for i in range(len(width_labels)):

                # create the rectangle
                ax[1].add_patch(Rectangle((0.0,strat_height_widths), width_values[i],
                                          1, facecolor='grey', edgecolor='k'))

                # the label
                ax[1].text(1.1, strat_height_widths+0.5, width_labels[i],
                           horizontalalignment='left', verticalalignment='center')

                # count the height
                strat_height_widths = strat_height_widths + 1

            # set the axis dimensions (values below are all in inches)
            ax1_lower_left_x = ax0_lower_left_x + ax0_width + 2
            ax1_lower_left_y = ax0_lower_left_y
            ax1_width = self.width_inches
            h = [Size.Fixed(ax1_lower_left_x), Size.Fixed(ax1_width)]
            v = [Size.Fixed(ax1_lower_left_y), Size.Fixed(ax_height)]
            divider = Divider(fig, (0.0, 0.0, 1., 1.), h, v, aspect=False)
            ax[1].set_axes_locator(divider.new_locator(nx=1, ny=1))

            # prettify
            ax[1].set_xlim(0,1)
            ax[1].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            for label in ax[1].get_xticklabels():
                label.set_rotation(270)
                label.set_ha('center')
                label.set_va('top')
            ax[1].set_axisbelow(True)
            ax[1].xaxis.grid(ls='--')
            ax[1].set_yticklabels([])
            ax[1].set_yticks([])
            ax[1].spines['top'].set_visible(False)
            ax[1].spines['right'].set_visible(False)

            # turning the spines off creates some clipping mask issues
            # so just turn the clipping masks off
            for obj in fig.findobj():
                obj.set_clip_on(False)

            # not really necessary since the axis sizes are already
            # forced, but we may as well set the figsize here too
            fig.set_size_inches(ax1_lower_left_x + ax1_width + ax0_lower_left_x,
                                ax1_lower_left_y*2 + ax_height)

        # if the color and width labels are the same
        else:

            # sort by width
            width_sort_inds = np.argsort(width_values)
            color_labels = color_labels[width_sort_inds]
            width_labels = width_labels[width_sort_inds]
            color_values = color_values[width_sort_inds]
            width_values = width_values[width_sort_inds]

            # initiate fig and ax
            fig, ax = plt.subplots()

            # determine the axis height and limits first
            ax_height = legend_unit_height * self.n_color_labels
            ax.set_ylim(0, self.n_color_labels)

            # initiate counting of the stratigraphic height
            strat_height = 0

            # loop over each item
            for i in range(len(color_labels)):

                # create the rectangle - with thickness of 1
                ax.add_patch(Rectangle((0.0,strat_height), width_values[i],
                                       1, facecolor=color_values[i], edgecolor='k'))

                # label the unit
                ax.text(1.1, strat_height+0.5, color_labels[i],
                        horizontalalignment='left', verticalalignment='center')

                # count the stratigraphic height
                strat_height = strat_height + 1

            # set the axis dimensions (values below are all in inches)
            ax_lower_left_x = 0.5
            ax_lower_left_y = 0.5
            ax_width = self.width_inches
            h = [Size.Fixed(ax_lower_left_x), Size.Fixed(ax_width)]
            v = [Size.Fixed(ax_lower_left_y), Size.Fixed(ax_height)]
            divider = Divider(fig, (0.0, 0.0, 1., 1.), h, v, aspect=False)
            ax.set_axes_locator(divider.new_locator(nx=1, ny=1))

            # prettify
            ax.set_xlim(0,1)
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            for label in ax.get_xticklabels():
                label.set_rotation(270)
                label.set_ha('center')
                label.set_va('top')
            ax.set_axisbelow(True)
            ax.xaxis.grid(ls='--')
            ax.set_yticklabels([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # turning the spines off creates some clipping mask issues
            # so just turn the clipping masks off
            for obj in fig.findobj():
                obj.set_clip_on(False)

            # not really necessary since the axis sizes are already
            # forced, but we may as well set the figsize here too
            fig.set_size_inches(ax_lower_left_x*2 + ax_width,
                                ax_lower_left_y*2 + ax_height)

        return fig, ax

#################
### FUNCTIONS ###
#################

def attribute_convert_and_check(attribute):
    """
    Convert pandas series to arrays (if necessary) and check that the
    data are 1d.

    This function assists the addition of attributes to Section and
    Data.

    Parameters
    ----------
    attribute : 1d array_like
        The attribute to be added.

    Returns
    -------
    attribute : 1d array
        The attribute after the conversion and check.
    """
    # convert from pandas series to arrays if necessary
    if type(attribute) == pd.core.series.Series:
        attribute = attribute.values

    # check that the data are 1d
    if attribute.ndim != 1:
        raise Exception('Data must be 1d.')

    return attribute

def section_style_compatibility(section, style):
    """
    Check that a `Section` and `Style` are compatible.

    Parameters
    ----------
    section : Section
        A Section object.

    style : Style
        A Style object.
    """
    # get the attributes - implicitly checks if the attributes exist
    color_attribute = getattr(section, style.color_attribute)
    width_attribute = getattr(section, style.width_attribute)

    # store failed labels
    color_failed = []
    width_failed = []

    all_check = True

    # loop over values in the Section
    for i in range(section.n_units):
        color_check = False
        width_check = False

        # check to see if the value is in Style
        for j in range(style.n_color_labels):
            if color_attribute[i] == style.color_labels[j]:
                color_check = True
        for j in range(style.n_width_labels):
            if width_attribute[i] == style.width_labels[j]:
                width_check = True

        # only print warning if the check fails for the first time
        if color_check == False:
            if color_attribute[i] not in color_failed:
                print('Color label in Section but not Style: ' + str(color_attribute[i]))
                color_failed.append(color_attribute[i])
            all_check = False
        if width_check == False:
            if width_attribute[i] not in width_failed:
                print('Width label in Section but not Style: ' + str(width_attribute[i]))
                width_failed.append(width_attribute[i])
            all_check = False

    # print an all clear statement if the check passes
    if all_check:
        print('Section and Style are compatible.')

def plot_swatch(swatch, extent, ax, swatch_wid=1.5):
    """
    plot a tesselated USGS geologic swatch to fit a desired extent

    Parameters
    ----------  
    swatch : PIL image
        PIL image of the swatch to tesselate

    extent : 1d array_like
        rectangular area in which to tesselate the swatch [x0, x1, y0, y1]
    ax : matplotlib axis
        axis in which to plot
    swatch_wid : float
        width of original swatch image file in inches
    mask : [to be implemented]
         masking geometry to apply to tesselated swatch (probably in axis coordinates?)
    """
    
    # dimensions of extent (data coordinates)
    dx_ex = x1 - x0
    dy_ex = y1 - y0
    ar = dy/dx
    
    # size of axis in inches
    
    # first get figure size
    fig = ax.get_figure()
    figsize = fig.get_size_inches()
    
    # axis inches per data unit
    ax_x_in = np.diff(fig.transFigure.inverted().transform(ax.transData.transform([(0, 0), (1, 0)])), axis=0)[0][0] * \
                         figsize[0]
    ax_y_in = np.diff(fig.transFigure.inverted().transform(ax.transData.transform([(0, 0), (0, 1)])), axis=0)[0][1] * \
                         figsize[1]
    dx_ex_in = dx_ex * ax_x_in
    dy_ex_in = dy_ex * ax_y_in
    
    # size of image
    dx_sw, dy_sw = swatch.size  # in pixels
    asp_sw = dy_sw/dx_sw
    dx_sw_in = swatch_wid
    dy_sw_in = asp_sw*dx_sw_in
    
    # tile image to overlap with extent
    sw_np = np.array(swatch)
    ny_tile = np.ceil(dy_ex_in/dy_sw_in).astype(int)
    nx_tile = np.ceil(dx_ex_in/dx_sw_in).astype(int)
    sw_tess = np.tile(sw_np, [ny_tile, nx_tile, 1])
    
    # now crop tessalated image to fit
    x_idx_crop = int(dx_ex_in/dx_sw_in/nx_tile*sw_tess.shape[1])
    y_idx_crop = int(dy_ex_in/dy_sw_in/ny_tile*sw_tess.shape[0])
    sw_tess = sw_tess[0:y_idx_crop, 0:x_idx_crop, :]
    
    ax.imshow(sw_tess, extent=extent, zorder=2)

def plot_stratigraphy(section, style, ncols=1, linewidth=1,
                      col_spacings=0.5, col_widths=1):
    """
    Initialize a figure and subplots, with the stratigraphic section
    already plotted on the first axis.

    This function is intended to act similar to `plt.subplots()` in that
    it will initialize the figure and axis handles. However, given that
    controlling the exact width and height of the axes is critical, this
    function is necessary to initialize the handles correctly.

    Note that setting the figsize itself is not sufficient to control
    the width and height of the axes exactly, since the figsize includes
    the size of the padding around the axes.

    Parameters
    ----------
    section : Section
        A Section object.

    style : Style
        A Style object.

    ncols : int
        The number of axes that will be in the figure (including the
        axis with the stratigraphic section.)

    linewidth : float
        The linewidth when drawing the stratigraphic section.

    col_spacings : float or array_like
        The spacing between the axes in inches. If a float, this value
        will be interpreted as uniform spacing between the axes. If an
        array, the length of the array must be ncols - 1, and the values
        will be the spacing between the individual axes.

    col_widths : float or array_like
        The width of the axes as a ratio relative to the width of the
        stratigraphic column. If a float, this value will be interpreted
        as a uniform width of the axes, excluding the stratigraphic
        column, for which the width is explicitly set in the Style. If
        an array, the length of the array must be ncols - 1, and the
        values will be the widths of the individual axes excluding the
        stratigraphic column.

    Returns
    -------
    fig : matplotlib Figure
        Figure handle.

    ax : matplotlib Axes
        Axis handle.
    """
    # get the attributes - implicitly checks if the attributes exist
    color_attribute = getattr(section, style.color_attribute)
    width_attribute = getattr(section, style.width_attribute)

    # initialize
    fig, ax = plt.subplots(nrows=1, ncols=ncols, sharey=True)

    # get the first axis
    if ncols == 1:
        ax0 = ax
    else:
        ax0 = ax[0]

    # determine the axis height and limits first
    ax_height = style.height_scaling_factor * section.total_thickness
    ax0.set_ylim(0, section.total_thickness)

    # initiate counting of the stratigraphic height
    strat_height = 0.0

    # loop over elements of the data
    for i in range(section.n_units):

        # pull out the thickness
        this_thickness = section.thicknesses[i]

        # loop over the elements in Style to get the color and width
        for j in range(style.n_color_labels):
            if color_attribute[i] == style.color_labels[j]:
                this_color = style.color_values[j]
        for j in range(style.n_width_labels):
            if width_attribute[i] == style.width_labels[j]:
                this_width = style.width_values[j]

        # create the rectangle
        ax0.add_patch(Rectangle((0.0,strat_height), this_width, this_thickness,
                      facecolor=this_color, edgecolor='k', linewidth=linewidth))

        # count the stratigraphic height
        strat_height = strat_height + this_thickness

    # set the axis dimensions (values below are all in inches)
    ax0_lower_left_x = 0.5
    ax0_lower_left_y = 0.5
    ax0_width = style.width_inches
    h = [Size.Fixed(ax0_lower_left_x), Size.Fixed(ax0_width)]
    v = [Size.Fixed(ax0_lower_left_y), Size.Fixed(ax_height)]
    divider = Divider(fig, (0.0, 0.0, 1., 1.), h, v, aspect=False)
    ax0.set_axes_locator(divider.new_locator(nx=1, ny=1))

    # prettify
    ax0.set_xlim(0,1)
    ax0.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    for label in ax0.get_xticklabels():
        label.set_rotation(270)
        label.set_ha('center')
        label.set_va('top')
    ax0.set_axisbelow(True)
    ax0.xaxis.grid(ls='--')
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.set_ylabel('stratigraphic height [m]')

    # turning the spines off creates some clipping mask issues
    # so just turn the clipping masks off
    for obj in ax0.findobj():
        obj.set_clip_on(False)

    # check if the col_spacing and col_widths is of the correct format
    if type(col_spacings)==list or type(col_spacings)==np.ndarray:
        if len(col_spacings) != ncols-1:
            raise Exception('col_spacings must be either a float or '
                            'array_like with length ncols-1.')
    if type(col_widths)==list or type(col_widths)==np.ndarray:
        if len(col_widths) != ncols-1:
            raise Exception('col_widths must be either a float or '
                            'array_like with length ncols-1.')

    # set up the other axes
    if ncols != 1:

        # iterate through the axes
        for i in range(1, ncols):

            # get the spacing and width values
            if type(col_spacings)==list or type(col_spacings)==np.ndarray:
                col_spacing = col_spacings[i-1]
            else:
                col_spacing = col_spacings
            if type(col_widths)==list or type(col_widths)==np.ndarray:
                col_width = col_widths[i-1] * ax0_width
            else:
                col_width = col_widths * ax0_width

            # adjust the axis
            if i==1:
                axn_lower_left_x = ax0_lower_left_x + ax0_width + col_spacing
            else:
                axn_lower_left_x = axn_lower_left_x + axn_width + col_spacing
            axn_lower_left_y = ax0_lower_left_y
            axn_width = col_width
            h = [Size.Fixed(axn_lower_left_x), Size.Fixed(axn_width)]
            v = [Size.Fixed(axn_lower_left_y), Size.Fixed(ax_height)]
            divider = Divider(fig, (0.0, 0.0, 1., 1.), h, v, aspect=False)
            ax[i].set_axes_locator(divider.new_locator(nx=1, ny=1))

    # set the figsize here too to account for the axis size manipulation
    if ncols != 1:
        fig.set_size_inches(ax0_lower_left_x*2 + np.sum(col_spacings) + np.sum(col_widths) + ax0_width,
                            ax0_lower_left_y*2 + ax_height)
    else:
        fig.set_size_inches(ax0_lower_left_x*2 + ax0_width,
                            ax0_lower_left_y*2 + ax_height)

    return fig, ax

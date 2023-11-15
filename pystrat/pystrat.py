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
import os
import copy

# import additional modules
import warnings
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1 import Divider, Size
from PIL import Image
import cvxpy as cp

##
## Global vars
##
mod_dir = os.path.dirname(os.path.realpath(__file__))

###############
### CLASSES ###
###############


class Fence:
    """
    Organizes sections according to a shared datum
    """

    def __init__(self,
                 sections,
                 datums=[],
                 correlations=[],
                 coordinates=[],
                 legend=False):
        """
        Initialize Section with the two primary attributes.

        Parameters
        ----------
        sections : 1d array_like
            List of Sections to be put into the fence
        
        datums : 1d array_like
            If not specified, the datum for each section will be the bottom. If specified, 
            must be list of same length as number of sections with heights in each section for the datum.

        correlations : 2d array_like
            each column is a correlated horizon where the rows are the heights of this horizon in each section.
            will plot as a line between fence posts.

        coordinates : 1d array_like
            1D coordinates of sections. distances between sections will be used to scale the plotting distances 
            in the fence diagram

        legend : boolean
            Whether or not to plot a legend

        To Do:
        ------
        - [ ] finish implementing distances in plotting
        """
        self.n_sections = len(sections)
        self.sections = [copy.deepcopy(section) for section in sections]
        datums = np.array(datums)
        if len(datums) == 0:
            datums = np.zeros(self.n_sections)
            for ii in range(self.n_sections):
                datums[ii] = sections[ii].base_height[0]
        else:
            assert len(
                datums
            ) == self.n_sections, 'Number of datums should equal number of sections'
        self.datums = datums

        correlations = np.array(correlations)
        if len(correlations) != 0:
            assert correlations.shape[
                0] == self.n_sections, 'Number of correlated horizons should match number of sections'
        self.correlations = correlations

        coordinates = np.array(coordinates)
        if len(coordinates) != 0:
            assert len(
                coordinates
            ) == self.n_sections, 'Number of section distances should match number of sections'

        # if coordinates not provided, assume sections are equally spaced in order provided
        if len(coordinates) == 0:
            self.coordinates = np.cumsum(np.ones(self.n_sections))
        else:
            self.coordinates = coordinates

        # order sections, correlations, datums by coordinates
        if len(self.coordinates) > 0:
            idx = np.argsort(self.coordinates)
            self.coordinates = self.coordinates[idx]
            self.sections = [self.sections[x] for x in idx]
            self.datums = self.datums[idx]
            if len(self.correlations) > 0:
                self.correlations = self.correlations[idx]

        # apply datums as shift to sections and correlations
        for ii in range(self.n_sections):
            self.sections[ii].shift_heights(-self.datums[ii])
            if len(self.correlations) > 0:
                self.correlations[
                    ii, :] = self.correlations[ii, :] - self.datums[ii]

    def plot(self,
             style,
             fig=None,
             legend=False,
             legend_wid=0.1,
             legend_hei=0.5,
             sec_wid=0.8,
             distance_spacing=False,
             plot_distances=[],
             distance_labels=False,
             plot_correlations=False,
             data_attributes=None,
             data_attribute_styles=None,
             section_plot_style={},
             **kwargs):
        """
        Plot a fence diagram

        Parameters
        ----------
        style : Style
            A Style object.

        fig : matplotlib.figure.Figure
            Figure to plot into if desired.

        legend : boolean
            Whether or not to include a legend for facies

        legend_wid : float [0, 1]
            Fractional width (figure coordinates) that legend occupies in fence diagram

        legend_hei : float [0, 1]
            fractional height of legend

        sec_wid : float (0, 1]
            width of section as a fraction of the columns containing the sections.
            1 means that the right limit of the section will be adjacent to the left
            limit of the subsequent section.

            Will automatically be divided by the maximum number of data attributes to be
            plotted.

        distance_spacing : boolean
            whether or not to scale the distances between sections according to the
            distances between self.coordinates, or plot_distances, if set

        plot_distances : 1d array like
            distances between sections to use for plotting. length is one less than
            n_sections. 

        distance labels : 1d array like or boolean
            length is (n_sections - 1) and permits manual labeling of distances 
            sections for schematic distances. if true, uses actual distances between
            sections.

        plot_correlations : boolean
            whether or not to plot correlated horizons

        data_attributes : 1d array like (defaults to None)
            list of data attributes to plot. if the attribute is not defined for a 
            particular section, it is not plotted.

        data_attribute_styles : 1d array like (defaults to None)
            style dictionary or dictionaries to use to plot data attributes. Either same
            length as data_attributes, or length of one
        
        section_plot_styles : dictionary
            dictionary of style parameters passed to section plotting
        """
        # before setting anything up, need to know if we're plotting data attributes and
        # how many
        if data_attributes is not None:
            # number of attributes to plot per section
            n_att_sec = np.zeros(self.n_sections).astype(int)
            for ii in range(self.n_sections):
                for attribute in data_attributes:
                    if hasattr(self.sections[ii], attribute):
                        n_att_sec[ii] = n_att_sec[ii] + 1
            assert np.sum(n_att_sec) > 0, 'data attribute not found in any sections'
            # update sec_width to reflect the number of data attributes being plotted
            sec_wid = sec_wid/np.max(n_att_sec+1)

            # set up a default style if none supplied
            if data_attribute_styles is None:
                data_attribute_styles = np.max(n_att_sec) * \
                                        [{'marker': '.', 
                                         'color': 'k',
                                         'linestyle': ''}]
            elif len(data_attribute_styles) == 1:
                data_attribute_styles = np.max(n_att_sec) * data_attribute_styles

        # set up axes to plot sections into (will share vertical coordinates)
        if fig == None:
            fig_provided = False
            fig = plt.figure(**kwargs)
        else:
            fig_provided = True
        min_height = np.min(
            [section.base_height[0] for section in self.sections])
        max_height = np.max(
            [section.top_height[-1] for section in self.sections])

        # if spacing sections realistically, set up axes as such
        axes = []
        n_axes = self.n_sections
        # if user wants non-uniform spacing between sections in fence diagram
        if distance_spacing:
            # distances between sections
            if len(plot_distances) == 0:
                distances = np.diff(self.coordinates)
                coordinates = self.coordinates
            else:
                assert len(plot_distances) == (
                    self.n_sections -
                    1), 'length of plot_distances must be n_sections - 1'
                distances = plot_distances
                coordinates = np.insert(np.cumsum(plot_distances), 0, 0)

            beta = np.min(distances) / (np.max(coordinates) -
                                        np.min(coordinates))
            if legend:
                # x is width of section axes in figure coordinates
                # x = beta*sec_wid/(1 + beta + beta*sec_wid)
                x = beta * sec_wid * (1 - legend_wid) / (1 + beta * sec_wid)
                # D is width in figure coordinates of area to space sections
                # D = (1-x)/(1+beta)
                D = 1 - x - legend_wid
                delta = beta * D
            else:
                x = beta * sec_wid / (1 + beta * sec_wid)
                D = 1 - x

            # now set up axis in figure coordinates
            coordinates = coordinates - np.min(
                coordinates)  # center coordinates
            ax_left_coords = coordinates / np.max(coordinates) * D

            for ii in range(self.n_sections):
                axes.append(
                    plt.axes([ax_left_coords[ii], 0, x * sec_wid, 1],
                             xlim=[0, 1], zorder=10))
            
            # if plotting data attributes, make those axes (list of lists)
            if data_attributes is not None:
                axes_dat = []
                for ii in range(self.n_sections):
                    # dont make an axis if the section doesn't have any data attributes
                    if n_att_sec[ii] == 0:
                        axes_dat.append(None)
                        continue
                    else:
                        cur_sec_dat_axes = []
                    for jj in range(len(data_attributes)):
                        if hasattr(self.sections[ii], data_attributes[jj]):
                            cur_sec_dat_axes.append(
                                plt.axes([ax_left_coords[ii] + (jj+1) * x * sec_wid, 0,
                                          x * sec_wid, 1], zorder=1))
                        else:
                            cur_sec_dat_axes.append(None)
                    axes_dat.append(cur_sec_dat_axes)

        # if user wants uniform spacing between sections in fence diagram
        else:
            x = sec_wid * (1 - legend_wid) / (self.n_sections - 1 + sec_wid)
            D = 1 - x - legend_wid
            delta = D / (self.n_sections - 1)
            ax_left_coords = np.arange(0, D + delta, delta)
            for ii in range(self.n_sections):
                # axes.append(fig.add_subplot(1, n_axes, ii+1))
                axes.append(
                    plt.axes([ax_left_coords[ii], 0, x * sec_wid, 1],
                             xlim=[0, 1]))

        if legend:
            leg_left_coord = 1 - delta
            axes.append(
                plt.axes([leg_left_coord, 0, x * sec_wid, legend_hei],
                         xlim=[0, 1]))

        # enforce axis limits before plotting to maintain swatch scaling
        for ii in range(self.n_sections):
            axes[ii].set_xlim([0, 1])
            axes[ii].set_ylim([min_height, max_height])

        # also enforce axis limits for data attributes
        if data_attributes is not None:
            for ii in range(self.n_sections):
                for jj, attribute in enumerate(data_attributes):
                    if hasattr(self.sections[ii], attribute):
                        axes_dat[ii][jj].set_ylim([min_height, max_height])

        # then plot sections
        for ii, section in enumerate(self.sections):
            section.plot(style, ax=axes[ii], **section_plot_style)

        # the plot data attributes, if specified
        if data_attributes is not None:
            for ii in range(self.n_sections):
                for jj, attribute in enumerate(data_attributes):
                    if hasattr(self.sections[ii], attribute):
                        self.sections[ii].plot_data_attribute(attribute, 
                                                              ax=axes_dat[ii][jj],
                                                              style=data_attribute_styles[jj])

        # plot and format legend
        if legend:
            style.plot_legend(ax=axes[-1])
            axes[-1].set_xticklabels([])

        # then move and scale axes so that vertical coordinate is consistent and datum is at same height in figure
        for ii in range(self.n_sections):
            # ax.set_xlim([0, 1])
            # ax.set_ylim([min_height, max_height])
            axes[ii].set_title(self.sections[ii].name, rotation=90, ha='right')

        # clean up plotting sections
        for ii in range(1, self.n_sections):
            # axes[ii].set_yticklabels = []
            # axes[ii].set_ylabel = None
            axes[ii].get_yaxis().set_visible(False)
            axes[ii].spines['left'].set_visible(False)
        for ii in range(self.n_sections):
            axes[ii].get_xaxis().set_visible(False)
            axes[ii].spines['bottom'].set_visible(False)

        # plot correlated beds as connections
        if plot_correlations:
            for ii in range(self.correlations.shape[1]):
                for jj in range(self.n_sections - 1):
                    # need to account for data attribute axes
                    if data_attributes is not None:
                        xyA = [1 + np.max(n_att_sec[jj]), self.correlations[jj, ii]]
                        xyB = [0, self.correlations[jj + 1, ii]]
                    else:
                        xyA = [1, self.correlations[jj, ii]]
                        xyB = [0, self.correlations[jj + 1, ii]]
                    if np.any(np.isnan(xyA)) or np.any(np.isnan(xyB)):
                        continue
                    con = ConnectionPatch(xyA=xyA,
                                          coordsA=axes[jj].transData,
                                          xyB=xyB,
                                          coordsB=axes[jj + 1].transData, zorder=15)
                    fig.add_artist(con)

        # plot datum
        for jj in range(self.n_sections - 1):
            xyA = [1, 0]
            xyB = [0, 0]
            con = ConnectionPatch(xyA=xyA,
                                  coordsA=axes[jj].transData,
                                  xyB=xyB,
                                  coordsB=axes[jj + 1].transData)
            fig.add_artist(con)

        # label distances between sections
        if distance_labels == False:
            pass
        elif (distance_labels == True) or (len(distance_labels) > 0):
            if type(distance_labels) == bool:
                distance_labels = np.diff(self.coordinates)
            else:
                assert len(distance_labels) == (
                    self.n_sections - 1), 'incorrect number of distance labels'

            fig_bbox = fig.get_tightbbox(
                fig.canvas.get_renderer())._bbox.get_points()
            x_offset = fig_bbox[0, 0]
            cur_y_coord = (fig_bbox[1, 1] - fig_bbox[0, 1])
            wid = fig_bbox[1,0]
            # hei = fig_bbox[1,1]
            for ii, distance in enumerate(distance_labels):

                cur_ax_bbox = axes[ii].get_window_extent().get_points()
                nex_ax_bbox = axes[ii + 1].get_window_extent().get_points()
                cur_x_coord = nex_ax_bbox[0, 0] - (
                    nex_ax_bbox[0, 0] - cur_ax_bbox[1, 0]) / 2 - x_offset

                # this value is the factor to account for the fact that the
                # figure coordinates need to be larger than one for the y
                # coordinate by the amount that the figur is larger than the 
                # axis bounding box
                vert_axis_fact = (cur_y_coord-cur_ax_bbox[1,1])/(2*cur_ax_bbox[1,1]) + 1
                plt.annotate(distance, (cur_x_coord/wid, vert_axis_fact),
                             xycoords='figure fraction',
                             ha='center',
                             fontsize=11)

        if not fig_provided:
            if data_attributes is not None:
                return fig, axes, axes_dat
            return fig, axes
        if data_attributes is not None:
            return axes, axes_dat
        return axes


class Section:
    """
    Organizes all data associated with a single stratigraphic section.
    """

    def __init__(self, thicknesses, facies, 
                 name=None, annotations=None, units=None):
        """
        Initialize Section with the two primary attributes.

        Parameters
        ----------
        thicknesses : 1d array_like
            Thicknesses of the facies. Any NaNs will be automatically
            removed.

        facies : 1d array_like
            Observed facies. Any NaNs will be automatically removed.

        name : str
            Name of the section

        annotations : 1d array_like
            Same length as facies. Names of annotations to plot alongside corresponding units
            in the column. Can be None (in which case nothing will be plotted). Otherwise, 
            comma separated list of the annotatoins to plot. Must match the names of png files
            in the annotations folder.

        units : array
            array of unit names. Each unit in the section must have a name.
            long dimension of the array should be len(thicknesses)
        """
        # convert to arrays and check the dimensionality
        thicknesses = attribute_convert_and_check(thicknesses)
        facies = attribute_convert_and_check(facies)
        if annotations is not None:
            annotations = attribute_convert_and_check(annotations)

        # check that the thicknesses are numeric
        # if thicknesses.dtype == np.object:
        #     raise Exception('Thickness data must be floats or ints.')

        # check for NaNs, and get rid of them
        thicknesses_nan_mask = np.isnan(thicknesses)
        if np.sum(thicknesses_nan_mask) > 0:
            warnings.warn(
                'Thickness data contains NaNs. These rows will be '
                'automatically removed, but you should check to make '
                'sure that this is appropriate for your dataset.')
            thicknesses = thicknesses[~thicknesses_nan_mask]

        facies_nan_mask = pd.isnull(facies)
        if np.sum(facies_nan_mask) > 0:
            warnings.warn(
                'Facies data contains NaNs. These rows will be '
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
        
        # save units
        if units is None:
            self.units = None
        else:
            units = np.atleast_2d(units)
            # make longest dimension into rows
            if units.shape[0] < units.shape[1]:
                units = units.T
            assert units.shape[0] == n_thicknesses_units, \
                'Unit names need to match number of thicknesses.'
            self.units = units

        # assign the core data to attributes
        self.thicknesses = thicknesses
        self.facies = facies
        self.annotations = annotations

        # add some other facies attributes
        self.top_height = np.cumsum(thicknesses)
        self.base_height = np.cumsum(thicknesses) - thicknesses
        self.unit_number = np.arange(n_thicknesses_units)

        # add some generic attributes
        self.name = name
        self.n_units = n_thicknesses_units
        self.total_thickness = np.sum(thicknesses)
        self.unique_facies = np.unique(facies)
        self.n_unique_facies = len(np.unique(facies))

        # keep track of attributes
        self.facies_attributes = [
            'unit_number', 'thicknesses', 'base_height', 'top_height', 'facies'
        ]
        self.generic_attributes = [
            'name', 'n_units', 'total_thickness', 'unique_facies',
            'n_unique_facies'
        ]
        self.data_attributes = []

    def shift_heights(self, shift):
        """
        Shift all heights by some fixed distance. (additive)

        Parameters
        ----------
        shift : int or float
            Amount by which to shift stratigraphic heights
        """
        # update facies attributes
        self.base_height = self.base_height + shift
        self.top_height = self.top_height + shift

        # data attributes
        for attribute in self.data_attributes:
            data_attribute = getattr(self, attribute)
            data_attribute.height = data_attribute.height + shift
            setattr(self, attribute, data_attribute)

    def plot_data_attribute(self, attribute, ax=None, style=None):
        if ax is None:
            ax = plt.axes()

        if style is None:
            style = {'marker': '.', 
                    'color': 'k',
                    'linestyle': ''}
        assert hasattr(self, attribute), 'Section does not have requested attribute.'

        cur_att = getattr(self, attribute)
        ax.plot(cur_att.values, cur_att.height, **style)
        ax.set_xlabel(attribute)
        # clean up axes
        ax.get_yaxis().set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)



    def plot(self, style, 
             ax=None, 
             linewidth=1, 
             annotation_height=0.25,
             label_units=False):
        """
        Plot this section using a Style object.

        Parameters
        ----------
        style : Style
            A Style object.

        ax : matplotlib axis
            Axis in which to plot, if desired. Otherwise makes a new axes object.
            AXIS LIMITS MUST BE APPLIED IN ADVANCE FOR SWATCHES TO PLOT CORRECTLY

        linewidth : float
            The linewidth when drawing the stratigraphic section.

        annotation_height : float
            The height in inches for annotation graphics. Set to zero to not plot
            annotations.

        label_units : boolean (default: False)
            Whether or not to label units on the left. If True, then section must
            have unit names specified.

        """
        # get the attributes - implicitly checks if the attributes exist
        style_attribute = getattr(self, style.style_attribute)

        # initialize
        if ax == None:
            ax = plt.axes()
            ax.set_ylim([self.base_height[0], self.top_height[-1]])

        # set up x axis
        unit_label_wid_tot = 0.2
        if label_units:
            ax.set_xlim([-unit_label_wid_tot, 1])

        # determine the axis height and limits first
        # ax_height = style.height_scaling_factor * section.total_thickness

        # initiate counting of the stratigraphic height
        strat_height = self.base_height[0]

        # loop over elements of the data
        for i in range(self.n_units):

            # pull out the thickness
            this_thickness = self.thicknesses[i]

            # loop over the elements in Style to get the color and width
            for j in range(style.n_labels):
                if style_attribute[i] == style.labels[j]:
                    this_color = style.color_values[j]
                    this_width = style.width_values[j]

            # create the rectangle
            ax.add_patch(
                Rectangle((0.0, strat_height),
                          this_width,
                          this_thickness,
                          facecolor=this_color,
                          edgecolor='k',
                          linewidth=linewidth))

            # if swatch is defined, plot it
            # if style.swatch_values[0] != None:
            ax.autoscale(False)
            if style.swatch_values is not None:
                for j in range(style.n_labels):
                    if style_attribute[i] == style.labels[j]:
                        this_swatch = style.swatch_values[j]
                        if this_swatch == 0:
                            continue
                        extent = [
                            0, this_width, strat_height,
                            strat_height + this_thickness
                        ]
                        plot_swatch(this_swatch, extent, ax,
                                    swatch_wid=style.swatch_wid)             

            # count the stratigraphic height
            strat_height = strat_height + this_thickness

        # plot annotations
        if (self.annotations is not None) and (annotation_height != 0):
            # preprocess annotation heights to determine and correct for overlap of symbols
            # get bounding vertical coordinates for all annotations
            idx = self.annotations != None
            
            # get height in data coordinates
            height = annotation_height/get_inch_per_dat(ax)[1]
            width = height/0.5*get_axis_aspect(ax)
            # these are the bottom coordinates for each annotation
            bot_coords = self.base_height[idx] + self.thicknesses[idx]/2 - height/2
            # these are the top coordinates
            top_coords = bot_coords + height

            # get the annotations
            annotations = self.annotations[idx]
            n_annotations = len(annotations)

            # correct for any overlap, output still in data coordinates
            if n_annotations > 1:
                bot_coords, top_coords = solve_overlap(bot_coords, top_coords)

            # iterate over the annotations
            for ii in range(n_annotations):
                for jj, annotation in enumerate(annotations[ii].split(',')):
                    # remember to adjust starting position for the number of annotations
                    pos = [jj*width + 1.1, bot_coords[ii]]
                    # check that annotation is in style
                    if annotation in list(style.annotations.keys()):
                        plot_annotation(style.annotations[annotation], pos, height, ax)

        # label units
        if label_units and (self.units is not None):
            # number of unit levels
            n_levels = self.units.shape[1]
            # width of each hierarchical label set
            unit_label_wid = unit_label_wid_tot / n_levels

            # loop over each level
            for ii in range(n_levels):
                # unique unit names
                cur_names_unique = np.unique(self.units[:, ii])
                seq_idxs = []
                for name in cur_names_unique:
                    # find sequences of unit names
                    seq_idxs.append(findseq(self.units, name))
                seq_idxs = np.concatenate(seq_idxs, axis=0)

                # create boxes and label with unit names
                for jj, name in enumerate(cur_names_unique):
                    # box
                    cur_x = -unit_label_wid_tot + ii * unit_label_wid
                    cur_y = self.base_height[seq_idxs[jj, 0]]
                    cur_orig = (cur_x, cur_y)
                    cur_height = self.top_height[seq_idxs[jj, 1]] - cur_y
                    cur_rect = Rectangle(cur_orig, unit_label_wid, cur_height,
                                         edgecolor='k', facecolor=(1, 1, 1, 0))
                    ax.add_patch(cur_rect)
                    # label
                    cur_x_text = cur_x + unit_label_wid/2
                    cur_y_text = cur_y + cur_height/2
                    cur_text = ax.text(cur_x_text, cur_y_text, name, 
                            va='center', ha='center',
                            rotation='vertical')
                    # get resulting data coordinates of the plotted label
                    transf = ax.transData.inverted()
                    bb = cur_text.get_window_extent()
                    bb_datacoords = bb.transformed(transf)
                    # if overstepping the box, give a warning and remove
                    if (bb_datacoords.y0 < cur_y) | (bb_datacoords.y1 > cur_y + cur_height):
                        print(f'Unit label {name} too big for box, removing.')
                        cur_text.remove()

        # prettify
        # ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        for label in ax.get_xticklabels():
            label.set_rotation(270)
            label.set_ha('center')
            label.set_va('top')
        ax.set_axisbelow(True)
        ax.xaxis.grid(ls='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel('Height (m)')

        # turning the spines off creates some clipping mask issues
        # so just turn the clipping masks off
        for obj in ax.findobj():
            obj.set_clip_on(False)

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
        df = pd.DataFrame({'unit_number': np.arange(self.n_units)})

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

    def add_data_attribute(self, attribute_name, attribute_height,
                           attribute_values):
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
        setattr(self, attribute_name,
                self.Data(attribute_height, attribute_values))

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
            # if attribute_height.dtype == np.object:
            #     raise Exception('Height data must be floats or ints.')

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
            self.height_attributes = ['height', 'values']
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
            df = pd.DataFrame({'height': self.height})

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
                raise Exception(
                    'Stratigraphic heights were out of order. '
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
                raise Exception(
                    'Data has not been assigned a facies. '
                    'Data.add_data_facies() will assign facies and '
                    'unit numbers automatically.')
            if 'unit_number' not in self.height_attributes:
                raise Exception(
                    'Data has not been assigned a unit_number. '
                    'Data.add_data_facies() will assign facies and '
                    'unit numbers automatically.')

            # get the dataframe
            df = self.return_data_dataframe()

            # get the samples on unit boundaries
            mask = df['unit_number'] != np.floor(df['unit_number'])
            df_slice = df[mask]

            # the case where there are no samples on units boundaries
            if len(df_slice.index) == 0:
                print(
                    'No samples are on unit boundaries - no manual edits required.'
                )

            # print the code
            else:
                print(
                    '1) Copy and paste the code below into a cell and edit as follows:'
                )
                print(
                    '- If the sample comes from the lower unit, subtract 0.5.')
                print('- If the sample comes from the upper unit, add 0.5.')
                print('')
                print('2) Run Data.clean_data_facies().')
                print('===')
                for i in df_slice.index:
                    print(data_name + '.unit_number[' + str(i) + '] = ' +
                          str(df_slice['unit_number'][i]) + ' #height = ' +
                          str(df_slice['height'][i]) + ', sample = ' +
                          str(df_slice['sample'][i]))

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
                 labels,
                 color_values,
                 width_values,
                 style_attribute='facies',
                 swatch_values=None,
                 annotations=None,
                 swatch_wid=1.5):
        """
        Initialize Style

        Note that compatibility of a Style with a Section is not checked
        until explicitly called, or plotting is attempted.

        Parameters
        ----------
        style_attribute : string (default 'facies')
            Section attribute name on which styling is based. When
            plotting a Section, the Section must have this attribute.

        labels : 1d array_like
            The labels to which colors and widths are assigned. When plotting a
            Section, values within the style_attribute of that Section
            must form a subset of the values within this array_like.

        color_values : array_like
            The colors that will be assigned to the associated labels.
            Values must be interpretable by matplotlib.

        width_values : 1d array_like of floats
            The widths that will be assigned to the associated labels.
            Values must be between 0 and 1.

        swatch_values : 1d array_like
            USGS swatch codes (see swatches/png/) for labels.
            Give zero for no swatch.

        annotations : dict or None
            Dictionary linking annotation names to png file paths for plotting
            annotations.

        swatch_wid : float (default 1.5)
            Width of the swatch pattern in inches.

        """
        # convert to arrays and check the dimensionality
        labels = attribute_convert_and_check(labels)
        width_values = attribute_convert_and_check(width_values)

        if swatch_values is not None:
            swatch_values = np.asarray(swatch_values).ravel().tolist()

        # convert from pandas series/dataframes to arrays if necessary
        if type(color_values) == pd.core.series.Series:
            color_values = color_values.values
        if type(color_values) == pd.core.frame.DataFrame:
            color_values = color_values.values

        # check that the widths are between 0 and 1
        if np.max(width_values) > 1 or np.min(width_values) < 0:
            raise Exception('Width values must be floats between 0 and 1.')

        # assign the attributes
        self.style_attribute = style_attribute
        self.labels = labels
        self.color_values = color_values
        self.width_values = width_values
        self.swatch_values = swatch_values
        self.annotations = annotations
        self.swatch_wid = swatch_wid

        # add some other useful attributes
        self.n_labels = len(labels)

    def plot_legend(self, 
                    ax=None, 
                    legend_unit_height=0.25,
                    fontsize=10):
        """
        Plot a legend for this Style object.

        Parameters
        ----------
        legend_unit_height : float
            A scaling factor to modify the height of each unit in the
            legend only.

        ax : matplotlib.Axes (default None)
            Axis to plot into. If None, creates an axis

        fontsize : float (default 10)
            Fontsize for text in the legend.

        Returns
        -------
        fig : matplotlib Figure
            Figure handle.

        ax : matplotlib Axes
            Axis handle.
        """

        # extract attributes
        labels = self.labels
        color_values = self.color_values
        width_values = self.width_values
        swatch_values = self.swatch_values

        # sort by width
        width_sort_inds = np.argsort(width_values)
        labels = labels[width_sort_inds]
        color_values = color_values[width_sort_inds]
        width_values = width_values[width_sort_inds]

        # what does this do??
        if swatch_values is not None:
            swatch_values = [swatch_values[x] for x in width_sort_inds]

        # initiate ax
        if ax == None:
            ax = plt.axes()

        # determine the axis height and limits first
        ax_height = legend_unit_height * self.n_labels
        ax.set_ylim(0, self.n_labels)
        ax.set_xlim([0, 1])

        # initiate counting of the stratigraphic height
        strat_height = 0

        # loop over each item
        for i in range(len(labels)):

            # create the rectangle - with thickness of 1
            ax.add_patch(
                Rectangle((0.0, strat_height),
                          width_values[i],
                          1,
                          facecolor=color_values[i],
                          edgecolor='k'))

            # if swatch is defined, plot it
            # if swatch_values[0] != None:
            if swatch_values is not None:
                if swatch_values[i] != 0:
                    extent = [
                        0, width_values[i], strat_height, strat_height + 1
                    ]
                    plot_swatch(swatch_values[i], extent, ax,
                                swatch_wid=self.swatch_wid)

            # label the unit
            # ax.text(-0.01,
            #         strat_height + 0.5,
            #         labels[i],
            #         horizontalalignment='right',
            #         verticalalignment='center')

            # count the stratigraphic height
            strat_height = strat_height + 1

        # plot annotations 
        if self.annotations is not None:
            n_annotations = len(self.annotations)

            # plot each one beside the column
            for ii in range(n_annotations):
                height = 0.6
                pos = [1.1, ii+0.5-height/2]

                # for text, assume min aspect ratio of image of 0.5
                width = height/0.5*get_axis_aspect(ax)
                ax.text(1.1+width, ii+0.5, list(self.annotations)[ii], 
                        va='center',
                        fontsize=fontsize)
                plot_annotation(list(self.annotations.values())[ii], pos, height, ax)


        # prettify
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        for label in ax.get_xticklabels():
            label.set_rotation(270)
            label.set_ha('center')
            label.set_va('top')
        ax.set_axisbelow(True)
        ax.xaxis.grid(ls='--')
        # ax.set_yticklabels([])
        # ax.set_yticks([])
        ax.set_yticks(np.arange(len(labels))+0.5)
        ax.set_yticklabels(labels, fontsize=fontsize)
        ax.tick_params(axis='y', width=0, pad=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # turning the spines off creates some clipping mask issues
        # so just turn the clipping masks off
        for obj in ax.findobj():
            obj.set_clip_on(False)



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
                print('Color label in Section but not Style: ' +
                      str(color_attribute[i]))
                color_failed.append(color_attribute[i])
            all_check = False
        if width_check == False:
            if width_attribute[i] not in width_failed:
                print('Width label in Section but not Style: ' +
                      str(width_attribute[i]))
                width_failed.append(width_attribute[i])
            all_check = False

    # print an all clear statement if the check passes
    if all_check:
        print('Section and Style are compatible.')


def plot_swatch(swatch_code, extent, ax, swatch_wid=1.5):
    """
    plot a tesselated USGS geologic swatch to fit a desired extent

    Parameters
    ----------  
    swatch_code : int
        USGS swatch code

    extent : 1d array_like
        rectangular area in which to tesselate the swatch [x0, x1, y0, y1]

    ax : matplotlib axis
        axis in which to plot

    swatch_wid : float
        width of original swatch image file in inches

    mask : [to be implemented]
        masking geometry to apply to tesselated swatch (probably in axis coordinates?)
    """

    x0, x1, y0, y1 = extent

    # dimensions of extent (data coordinates)
    dx_ex = x1 - x0
    dy_ex = y1 - y0

    if dx_ex == 0 or dy_ex == 0:
        warnings.warn(
            'Extent has no width and/or height. Swatch cannot be plotted.')
        return

    # load swatch
    swatch = Image.open(mod_dir + '/swatches/png/%s.png' % swatch_code)

    # first get figure size
    # fig = ax.get_figure()
    # figsize = fig.get_size_inches()

    # # axis inches per data unit
    # ax_x_in = np.diff(fig.transFigure.inverted().transform(ax.transData.transform([(0, 0), (1, 0)])), axis=0)[0][0] * \
    #                      figsize[0]
    # ax_y_in = np.diff(fig.transFigure.inverted().transform(ax.transData.transform([(0, 0), (0, 1)])), axis=0)[0][1] * \
    #                      figsize[1]
    
    ax_x_in, ax_y_in = get_inch_per_dat(ax)

    dx_ex_in = dx_ex * ax_x_in
    dy_ex_in = dy_ex * ax_y_in

    # size of image
    dx_sw, dy_sw = swatch.size  # in pixels
    asp_sw = dy_sw / dx_sw
    dx_sw_in = swatch_wid
    dy_sw_in = asp_sw * dx_sw_in

    # tile image to overlap with extent
    sw_np = np.array(swatch)
    ny_tile = np.ceil(dy_ex_in / dy_sw_in).astype(int)
    nx_tile = np.ceil(dx_ex_in / dx_sw_in).astype(int)
    sw_tess = np.tile(sw_np, [ny_tile, nx_tile, 1])

    # now crop tessalated image to fit
    x_idx_crop = int(dx_ex_in / dx_sw_in / nx_tile * sw_tess.shape[1])
    y_idx_crop = int(dy_ex_in / dy_sw_in / ny_tile * sw_tess.shape[0])
    sw_tess = sw_tess[0:y_idx_crop, 0:x_idx_crop, :]

    if sw_tess.shape[0] == 0 or sw_tess.shape[1] == 0:
        warnings.warn(
            'Extent has no width and/or height. Swatch cannot be plotted.')
        return

    ax.imshow(sw_tess, extent=extent, zorder=2, aspect='auto')
    # ax.imshow(sw_tess, extent=extent, zorder=2)
    ax.autoscale(False)


def plot_annotation(annotation_path, pos, height, ax,):
    """
    plot a png of 

    Parameters
    ----------  
    annotation_path : str
        Path to the file of the annotation

    pos : 1d array_like
        left and bottom of the annotation plotting, data units [x0, y0]

    height : numeric
        vertical thickness to plot (vertical extent will be y0 to y0 + height)

    ax : matplotlib axis
        axis in which to plot

    """

    # load annotation
    annotation = Image.open(annotation_path)
    ann_arr = np.array(annotation)
    
    # axis inches per data unit
    aspect_axis = get_axis_aspect(ax)

    # size of image
    dx, dy= annotation.size  # in pixels
    aspect_ann = dy / dx
    width = height/aspect_ann*aspect_axis

    extent = [pos[0], pos[0]+width, pos[1], pos[1]+height]

    ax.imshow(ann_arr, extent=extent, zorder=2, aspect='auto')
    ax.autoscale(False)


def get_inch_per_dat(ax):
    """Returns the inches per data unit of the given axis 
    (x, y)

    :param ax: axis handle
    :type ax: axis handle
    """
    fig = ax.get_figure()
    figsize = fig.get_size_inches()

    x_inch_per_dat = np.diff(fig.transFigure.inverted().transform(ax.transData.transform([(0, 0), (1, 0)])), axis=0)[0][0] * \
                         figsize[0]
    y_inch_per_dat = np.diff(fig.transFigure.inverted().transform(ax.transData.transform([(0, 0), (0, 1)])), axis=0)[0][1] * \
                         figsize[1]
    return x_inch_per_dat, y_inch_per_dat


def get_axis_aspect(ax):
    """Returns the aspect ratio of the axis in terms of inches per data unit for each axis

    :param ax: axis
    :type ax: axis handle
    """
    x_inch_per_dat, y_inch_per_dat = get_inch_per_dat(ax)
    return y_inch_per_dat/x_inch_per_dat


def solve_overlap(bot_coords, top_coords):
    """Returns coordinates that have solved for overlap between things that shouldn't be overlapping.
    assume that the coordinates are already appropriately ordered.

    :param bot_coords: bottom coordinates for objects
    :type bot_coords: 1d array like
    :param top_coords: top coordinates for objects
    :type top_coords: 1d array like
    """
    n = len(bot_coords)

    G = np.zeros([n-1, n])
    h = np.zeros(n-1)
    for ii in range(n-1):
        G[ii, ii] = 1
        G[ii, ii+1] = -1
        h[ii] = bot_coords[ii+1] - top_coords[ii]
    
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(0.5*cp.quad_form(x, np.identity(n))),
                    [G @ x <= h])
    prob.solve(verbose=False)

    delta = x.value

    bot_coords_free = bot_coords + delta
    top_coords_free = top_coords + delta

    return bot_coords_free, top_coords_free


###
### HELPER FUNCTIONS
###
def findseq(x, val, noteq=False):
    """
    Find sequences of a given value within an input vector.

    IN:
     x: vector of values in which to find sequences
     val: value to find sequences of in x
     noteq: (false) whether to find sequences equal or not equal to the supplied
        value

    OUT:
     idx: array that contains in rows the number of total sequences of val, with
       the first column containing the begin indices of each sequence, the second
       column containing the end indices of sequences, and the third column
       contains the length of the sequence.
    """
    x = x.copy().squeeze()
    assert len(x.shape) == 1, "x must be vector"
    # indices of value in x, and
    # compute differences of x, since subsequent occurences of val in x will
    # produce zeros after differencing. append nonzero value at end to make
    # x and difx the same size
    if noteq:
        validx = np.atleast_1d(np.argwhere(x != val).squeeze())
        logidx = x != val
        x[validx] = val+1
        difx = np.append(np.diff(logidx),1)
    else:
        validx = np.atleast_1d(np.argwhere(x == val).squeeze())
        logidx = x == val
        # difx = np.append(np.diff(x),1)
        difx = np.append(np.diff(logidx),1)
    nval = len(validx)
    # if val not in x, warn user
    if nval == 0:
        warnings.warn("value val not found in x")
        return 0

    # now, where validx is one and difx is zero, we know that we have
    # neighboring values of val in x. Where validx is one and difx is nonzero,
    # we have end of a sequence

    # now loop over all occurrences of val in x and construct idx
    c1 = 0
    idx = np.empty((1,3))
    while c1 < nval:
        curidx = np.array([[validx[c1],validx[c1],1]])
        c2 = 0
        while difx[validx[c1]+c2] == 0:
            curidx[0,1] += 1
            curidx[0,2] += 1
            c2 += 1
        idx = np.append(idx,curidx,axis=0)
        c1 = c1+c2+1
    idx = idx[1:,:].astype(int)
    return idx
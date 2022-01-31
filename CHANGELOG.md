# Changelog

## unreleased

**Added**
- USGS swatches as optional components of `Style`. Labels are the same as those supplied for the color style.
- Fence class for combining Sections
  - this class is currently capable of plotting a Fence diagram
  - traces between correlative units can be plotted
  - distances between sections can be scaled and labeled
- Added class method `Section.shift_heights()` to Section to shift stratigraphic heights by a fixed amount (also shifts data attribute heights)
- Sections can now have names

**Changed**

Styling
Colors and widths are no longer separate stylistic decisions tied to potentially separate attributes. Rather, styling is now tied to an single facies attribute, and every facies has its own color and width. This change reflects the opinion that stratigraphic sections are visualizations of subjectively determined phenomena of interest, which inevitably resolve at the facies scale. Any stylistic considerations should then reflect the facies interpretation directly, rather than distributing facies descriptors into multiple, less interpretable attributes. This means that `Style.plot_legend()` will only ever plot a single style.

Plotting
- Plotting is less prescriptive now. Decisions about subplots and figure/axis dimensions are made externally to `pystrat`, and the user has more flexibility in supplying axes to plot into.
- Plotting a section is now a class method (`plot_stratigraphy()` &rarr; `Section.plot()`)
- Section plotting no longer starts strictly at 0; instead the base of the section is taken as the first entry of `Section.base_heights` (which may be non-zero if the user has used `Section.shift_heights()`)
- Plotting the legend for a style is more flexible


## To Do

- [ ] drawing annotations for beds as part of Section.plot()
- [ ] when plotting swatches, if the axis y-limits are smaller than the total height of the stratigraphy, all of the swatches are still plotted even though they extend beyond the axis limits. This behavior should be changed to respect the axis limits.
- [ ] traces between correlative units in `Fence.plot()` should snap to the x-coordinate of the unit on the left
- [ ] update tutorial to reflect new features and changes

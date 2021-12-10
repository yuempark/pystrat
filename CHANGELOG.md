# Changelog

## unreleased

**Added**
- USGS swatches as optional components of `Style`. Labels are the same as those supplied for the color style.
- Fence class for combining Sections
  - this class is currently capable of plotting a Fence diagram
- Added class method `Section.shift_heights()` to Section to shift stratigraphic heights by a fixed amount (also shifts data attribute heights)
- Sections can now have names

**Changed**
- Plotting is less prescriptive now. Decisions about subplots and figure/axis dimensions are made externally to `pystrat`, and the user has more flexibility in supplying axes to plot into.
- Plotting a section is now a class method (`plot_stratigraphy()` &rarr; `Section.plot()`)
- Section plotting no longer starts strictly at 0; instead the base of the section is taken as the first entry of `Section.base_heights` (which may be non-zero if the user has used `Section.shift_heights()`)


## To Do

- [ ] drawing annotations for beds as part of Section.plot()
- [ ] when plotting swatches, if the axis y-limits are smaller than the total height of the stratigraphy, all of the swatches are still plotted even though they extend beyond the axis limits. This behavior should be changed to respect the axis limits.

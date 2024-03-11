"""
helper script to convert svgs into pngs
"""

import cairosvg
import glob
import os

svgs = glob.glob('SVG/*.svg')

for svg in svgs:
    svg = os.path.basename(svg)
    cairosvg.svg2png(url='SVG/' + svg, 
                     write_to='png/' + svg.split('.')[0] + '.png', 
                     dpi=300, scale=5)

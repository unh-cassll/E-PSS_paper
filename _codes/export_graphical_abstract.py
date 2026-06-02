"""
Export the graphical abstract SVG to PDF.

Created: 2026-03-01
@author: nathanlaxague
"""

from drawsvg import Drawing

input_file = '../_figures/e-pss_graphical_abstract.svg'
output_file = '../_figures/e-pss_graphical_abstract.pdf'

d = Drawing(filename=input_file)
d.saveas(output_file)


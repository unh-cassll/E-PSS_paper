"""
Created on Sun Mar 2026 10:24:01 2026

@author: nathanlaxague
"""

from drawsvg import Drawing

input_file = '../_figures/e-pss_graphical_abstract.svg'
output_file = '../_figures/e-pss_graphical_abstract.pdf'

d = Drawing(filename=input_file)
d.saveas(output_file)


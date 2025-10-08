# E-PSS

**A project in which we extend the Polarimetric Slope Sensing (PSS) technique for remotely observing ocean waves.**

Manuscript in preparation for *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing* (*J-STARS*):

"E-PSS: the Extended Polarimetric Slope Sensing technique for measuring ocean surface waves"
Nathan J. M. Laxague, Z. Göksu Duvarci, Lindsay Hogan, Junzhe Liu, Christopher Bouillon, and Christopher J. Zappa

## Contents

**Figure-generating codes**

Start with *aa_e_pss_figure_gen.m* ("aa" in filename pushes it to top of alphabetically-sorted lists)

... rest of codes in root directory are MATLAB functions called by the *figure_gen* script.

**Codes used in preparation of data**

These live within *codes*.

**Data used to produce graphics**

These files live within *data*. This folder is meant to be the smallest collection of example data which could be feasibly used for demonstration/testing purposes. The original raw datasets from which the spectra/statistics were produced are substantial in size (of order 300 TB).

**Graphics**

These images live within *figures*. The *figure_gen* script in the root directory handles the printing of these to file, with user-set option to print as vector (.svg) or raster (.png) images.

**Manuscript**

These files live within *manuscript*. This directory is meant to be a self-contained LaTeX project.

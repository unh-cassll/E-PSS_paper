
"""
Created on Fri Oct 31 09:13:23 2025

@author: nathanlaxague
"""

import subprocess
import os
import time

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# define URL and list of filenames to grab
URL = "https://zenodo.org/records/17388350/files/"
filenames_to_grab = ["ASIT2019_supporting_environmental_observations.nc", "ASIT2019_wave_spectra_stats_timeseries_empirical_gain.nc", "ASIT2019_wave_spectra_stats_timeseries_lab_gain.nc", "ASIT2019_wave_spectra_stats_timeseries_no_gain.nc", "Elfouhaily_et_al_1997_cumulative_mss.nc", "Piermont2025_DoLP_AoI_observations.nc", "Run051_example_Stokes_parameters.nc"]
suffix = "?download=1"

local_dir = '_data'

# loop over the list of files to grab; download each one if a local copy doesn't exist
for short_fname in filenames_to_grab:
    
    external_fname = URL + short_fname + suffix
    local_fname = os.path.join(local_dir,short_fname)
    
    if not os.path.exists(local_fname):
        
        printname = "Downloading " + short_fname
        print(printname)
        command = f"wget --show-progress {external_fname} -O {local_fname}"
        subprocess.call(command, shell=True)
        
        time.sleep(3)
        
    else:
        
        printname = short_fname + " already exists. Skipping download"
        print(printname)

"""
Download observational data files from Zenodo if not already present locally.

Resolves the evergreen Zenodo *concept* record, which always redirects to the
latest published version, so this script never needs editing when a new dataset
version is uploaded (concept DOI: 10.5281/zenodo.17388349).

Created: 2025-10-31
@author: nathanlaxague
"""

import json
import os
import subprocess
import time
import urllib.request

import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Evergreen Zenodo concept record -- the API redirects to the newest version.
concept_recid = "17388349"
filenames_to_grab = [
    "ASIT2019_supporting_environmental_observations.nc",
    "ASIT2019_slope_fields_reduced.nc",
    "ASIT2019_wave_spectra_stats_timeseries_empirical_gain.nc",
    "ASIT2019_wave_spectra_stats_timeseries_lab_gain.nc",
    "ASIT2019_wave_spectra_stats_timeseries_no_gain.nc",
    "ASIT2019_EPSS_directional_spectra.nc",
    "ASIT2019_omnidirectional_spectra.nc",
    "aperture_field_stitch.nc",
    "Elfouhaily_et_al_1997_cumulative_mss.nc",
    "Piermont2025_DoLP_AoI_observations.nc",
    "Run051_example_Stokes_parameters.nc",
]

local_dir = '_data'

# Resolve the concept record to the latest version, mapping filename -> URL
api_url = f"https://zenodo.org/api/records/{concept_recid}"
with urllib.request.urlopen(api_url) as response:
    record = json.load(response)
file_urls = {f["key"]: f["links"].get("content", f["links"].get("self"))
             for f in record["files"]}
print(f"Latest Zenodo record: {record['id']} ({len(file_urls)} files available)")

os.makedirs(local_dir, exist_ok=True)

# Download each file if a local copy does not exist
for short_fname in filenames_to_grab:

    local_fname = os.path.join(local_dir, short_fname)

    if os.path.exists(local_fname):

        print(short_fname + " already exists. Skipping download")
        continue

    external_fname = file_urls.get(short_fname)

    if external_fname is None:

        print("WARNING: " + short_fname + " not in latest record; skipping")
        continue

    print("Downloading " + short_fname)
    command = f"wget --show-progress {external_fname} -O {local_fname}"
    subprocess.call(command, shell=True)

    time.sleep(3)

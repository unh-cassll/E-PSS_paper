.PHONY: all

all: data

data: compile_pseudo_buoy_directional_spectra \
	compute_all_directional_spectra \
	compute_all_omnidirectional_spectra \
	compute_slope_statistics_from_PDFs

# single-triplet estimator suite. The per-run source files come from the
# companion wave-direction-estimation repository; set EPSS_PSEUDOBUOY_DIR to
# relocate it. Without that tree the script leaves any existing product alone
# and returns success, so the rest of the chain still builds.
compile_pseudo_buoy_directional_spectra: ../_codes/compile_pseudo_buoy_directional_spectra.py
	python3 ../_codes/compile_pseudo_buoy_directional_spectra.py

compute_all_directional_spectra: ../_codes/compute_all_directional_spectra.py
	python3 ../_codes/compute_all_directional_spectra.py

compute_all_omnidirectional_spectra: ../_codes/compute_all_omnidirectional_spectra.py
	python3 ../_codes/compute_all_omnidirectional_spectra.py

compute_slope_statistics_from_PDFs: ../_codes/compute_slope_statistics_from_PDFs.py
	python3 ../_codes/compute_slope_statistics_from_PDFs.py

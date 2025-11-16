.PHONY: all

all: data

data: compute_all_directional_spectra \
	compute_all_omnidirectional_spectra \
	compute_slope_statistics_from_PDFs

compute_all_directional_spectra: ../_codes/compute_all_directional_spectra.py
	python3 ../_codes/compute_all_directional_spectra.py
	
compute_all_omnidirectional_spectra: ../_codes/compute_all_omnidirectional_spectra.py
	python3 ../_codes/compute_all_omnidirectional_spectra.py

compute_slope_statistics_from_PDFs: ../_codes/compute_slope_statistics_from_PDFs.py
	python3 ../_codes/compute_slope_statistics_from_PDFs.py

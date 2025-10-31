.PHONY: all data

all: grab \
    data

grab: grab_observational_data \

data: compute_all_directional_spectra \
	compute_slope_statistics_from_PDFs

grab_observational_data: ../_codes/grab_observational_data.py
	python3 ../_codes/grab_observational_data.py

compute_all_directional_spectra: ../_codes/compute_all_directional_spectra.py
	python3 ../_codes/compute_all_directional_spectra.py

compute_slope_statistics_from_PDFs: ../_codes/compute_slope_statistics_from_PDFs.py
	python3 ../_codes/compute_slope_statistics_from_PDFs.py

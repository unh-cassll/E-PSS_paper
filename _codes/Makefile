.PHONY: all figures

all: figures

figures: binned_omnispect \
	comparison_of_directional_information \
	directional_spectra \
	DoLP_AOI_field \
	DoLP_AOI_gain \
	Hm0_scatterplots_lidar_EPSS \
	mueller_calc \
	slope_elev_timeseries_and_spectra \
	slope_PDFs \
	slope_stats \
	T_E_scatterplots_lidar_EPSS \
	wave_measurement_stommel_diagram
	
binned_omnispect: plot_binned_omnispect.py
	python3 plot_binned_omnispect.py
	
comparison_of_directional_information: plot_comparison_of_directional_information.py
	python3 plot_comparison_of_directional_information.py
	
directional_spectra: plot_directional_spectra.py
	python3 plot_directional_spectra.py
	
DoLP_AOI_field: plot_DoLP_AOI_field.py
	python3 plot_DoLP_AOI_field.py

DoLP_AOI_gain: plot_DoLP_AOI_gain.py
	python3 plot_DoLP_AOI_gain.py

Hm0_scatterplots_lidar_EPSS: plot_Hm0_scatterplots_lidar_EPSS.py
	python3 plot_Hm0_scatterplots_lidar_EPSS.py

mueller_calc: plot_mueller_calc.py
	python3 plot_mueller_calc.py

slope_elev_timeseries_and_spectra: plot_slope_elev_timeseries_and_spectra.py
	python3 plot_slope_elev_timeseries_and_spectra.py

slope_PDFs: plot_slope_PDFs.py
	python3 plot_slope_PDFs.py

slope_stats: plot_slope_stats.py
	python3 plot_slope_stats.py

T_E_scatterplots_lidar_EPSS: plot_T_E_scatterplots_lidar_EPSS.py
	python3 plot_T_E_scatterplots_lidar_EPSS.py

wave_measurement_stommel_diagram: plot_wave_measurement_stommel_diagram.py
	python3 plot_wave_measurement_stommel_diagram.py


.PHONY: all figures

all: figures

figures: aperture_field_stitch \
	binned_omnispect \
	comparison_of_directional_information \
	directional_spectra \
	DoLP_AOI_inference \
	DoLP_AOI_gain \
	Hm0_scatterplots_lidar_EPSS \
	longwave_shortwave_demo \
	mueller_calc \
	slope_elev_timeseries_and_spectra \
	slope_stats \
	Tm02_scatterplots_lidar_EPSS \
	wave_measurement_stommel_diagram

aperture_field_stitch: plot_aperture_field_stitch.py
	@python3 plot_aperture_field_stitch.py
	@echo "saved aperture_field_stitch.pdf to file"

longwave_shortwave_demo: plot_longwave_shortwave_demo.py
	@python3 plot_longwave_shortwave_demo.py
	@echo "saved longwave_shortwave_demo_stacked.pdf to file"

binned_omnispect: plot_binned_omnispect.py
	@python3 plot_binned_omnispect.py
	@echo "saved omnidirectional_spectra_binned_by_wind.pdf to file"

comparison_of_directional_information: plot_comparison_of_directional_information.py
	@python3 plot_comparison_of_directional_information.py
	@echo "saved delta_theta_nought.pdf to file"
	@echo "saved directional_spreading_comparison.pdf to file"
	@echo "saved directional_values.tex to file"

directional_spectra: plot_directional_spectra.py
	@python3 plot_directional_spectra.py
	@echo "saved directional_spectra_polar.pdf to file"
	@echo "saved directional_spectra_combined.pdf to file"

DoLP_AOI_inference: plot_DoLP_AOI_inference.py
	@python3 plot_DoLP_AOI_inference.py
	@echo "saved DoLP_AOI_inference.pdf to file"

DoLP_AOI_gain: plot_DoLP_AOI_gain.py
	@python3 plot_DoLP_AOI_gain.py
	@echo "saved DoLP_AOI_gain_example.pdf to file"

Hm0_scatterplots_lidar_EPSS: plot_Hm0_scatterplots_lidar_EPSS.py
	@python3 plot_Hm0_scatterplots_lidar_EPSS.py
	@echo "saved Hm0_comparison_lidar_epss.pdf to file"
	@echo "saved Hm0_values.tex to file"

mueller_calc: plot_mueller_calc.py
	@python3 plot_mueller_calc.py
	@echo "saved mueller_calc_example.pdf to file"

slope_elev_timeseries_and_spectra: plot_slope_elev_timeseries_and_spectra.py
	@python3 plot_slope_elev_timeseries_and_spectra.py
	@echo "saved elevation_omnispect.pdf to file"

slope_stats: plot_slope_stats.py
	@python3 plot_slope_stats.py
	@echo "saved mss_upwind_crosswind.pdf to file"
	@echo "saved slope_distribution_GC_coeffs.pdf to file"

Tm02_scatterplots_lidar_EPSS: plot_Tm02_scatterplots_lidar_EPSS.py
	@python3 plot_Tm02_scatterplots_lidar_EPSS.py
	@echo "saved Tm02_comparison_lidar_epss.pdf to file"
	@echo "saved Tm02_values.tex to file"

wave_measurement_stommel_diagram: plot_wave_measurement_stommel_diagram.py
	@python3 plot_wave_measurement_stommel_diagram.py
	@echo "saved wave_measurement_stommel_diagram.pdf to file"

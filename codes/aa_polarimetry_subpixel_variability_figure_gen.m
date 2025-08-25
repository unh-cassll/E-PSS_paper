% Figure generation script for
% "The Extended Polarimetric Slope Sensing Technique"
% 
% N. J. M. Laxague and co-authors, 2025
%

addpath subroutines

figure_style

close all;clear;clc

corner_x = 1225;
corner_y = 500;

full_width = 1500;
full_height = 600;

fsize = 18;

n = 1;
print_options = {'none','svg','png'};
print_option = print_options{n};

dpi_val = 100;
dpi_string = ['-r' num2str(dpi_val)];

figpos = [corner_x corner_y full_width full_height];

out_path = '../figures/';


%% Figure 1: TBD

fignum = 1;
set(fignum,'Position',figpos.*[1 1 0.5 1])

switch print_option
    case 'none'
    case 'svg'
        figure(fignum);print([out_path 'Fig01.svg'],'-dsvg')
    case 'png'
        figure(fignum);print([out_path 'Fig01.png'],'-dpng',dpi_string);
end

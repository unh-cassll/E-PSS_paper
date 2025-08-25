%set(0,'DefaultAxesFontName','SansSerif')
%set(0,'DefaultAxesFontName','Ubuntu')
% set(0,'DefaultTextFontName','DejaVu Math TeX Gyre')
% set(0,'DefaultAxesFontName','DejaVu Math TeX Gyre')
% set(0,'DefaultTextFontName','Static')
% set(0,'DefaultAxesFontName','Static')
set(0,'DefaultTextFontName','Noto Mono')
set(0,'DefaultAxesFontName','Noto Mono')
set(0,'DefaultAxesFontSize',18)
set(0,'DefaultFigureColormap',viridis)
set(0,'DefaultAxesLineWidth',2)
set(0,'DefaultLineMarkerSize',20)
set(0,'DefaultAxesXGrid','on','DefaultAxesYGrid','on')
set(groot, 'defaultAxesTickDir', 'out');
set(groot,  'defaultAxesTickDirMode', 'manual');
set(groot,'defaultFigureRenderer','painters')
v_order = [7 6 3 1 5 2];
cmap = spectral(7);
set(0,'DefaultAxesColorOrder',cmap(v_order,:));
clear
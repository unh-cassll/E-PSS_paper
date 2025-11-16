"""
Created on Wed Sep  3 11:05:07 2025

@author: nathanlaxague
"""

import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid",palette="deep",font="DejaVu Sans Mono")

g = 9.81
sigma = 0.072
rho = 1020
k = np.logspace(-2,4, 1000)
omega = np.sqrt(g*k+sigma/rho*k**3)

x = 2*np.pi/k
y = 2*np.pi/omega

fig = plt.figure(figsize=(6, 4.25))

ax = fig.add_axes([0.1, 0.1, 0.7, 0.8])
plt.loglog(x, y, color='k')

colors = ['#4C2882', '#367588', '#A52A2A', '#C39953', '#2A52BE']

methods =['polarimetry','stereo video']

rectangles = [
    (0.0025, 0.033, 2.5, 1.25),  # polarimetry (as it has been done)
    (0.4, 0.5, 100, 8),        # stereo (as it has been done)
    (5, 10, 0.8, 0.3),
    (15, 50, 1.5, 0.5),
    (30, 100, 0.5, 2.0)
]

for (x_pos, y_pos, width, height), color, method in zip(rectangles,colors,methods):
    rect = plt.Rectangle((x_pos, y_pos), width, height, color=color, alpha=0.25)
    plt.gca().add_patch(rect)
    plt.text(x_pos+width*0.85,y_pos*1.15,method,color=color,horizontalalignment='right')

# plt.text(30,40,'gravity-capillary linear dispersion',horizontalalignment='right')

plt.xlim(0.001, 100)
plt.ylim(0.01, 100)

plt.xlabel('spatial scale [m]')
plt.ylabel('temporal scale [s]')

plt.grid(which='major', linestyle='-', linewidth=0.75)
plt.grid(which='minor', linestyle=':', linewidth=0.75)

line_ax = fig.add_axes([0.8, 0.1, 0.05, 0.8])  
line_ax.set_ylim(0.01, 100) 
line_ax.axis('off') 

line_ax.loglog([1, 1], [1.5, 80], color=colors[4], linewidth=3)

line_ax.text(3, 10, 'buoys, gauges', fontsize=12, rotation=90, color=colors[4], va='center', ha='left')

line_ax.annotate('', xy=(1, 100), xytext=(1, 11), 
                 arrowprops=dict(arrowstyle='->', color=colors[4], lw=3))

plt.savefig('../_figures/wave_measurement_stommel_diagram.pdf',bbox_inches='tight')

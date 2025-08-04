#FILE SUMMARY:
### plot of sigmoid function with different parameters on top of each other

#LIBRARIES
import os
import re
import glob
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd
import netCDF4 as nc
from skimage.io import imread
from skimage import exposure
from skimage import transform
from skimage import filters

def sigmoid(x, cut_off, gain):
    """
    sigmoid function
    """
    return 1 / (1 + np.exp(-(x - cut_off) * gain)) # sigmoid function

x = np.linspace(0, 1, 100) # creating values for displaying the function

#different cut off factors for comparison
parameters_cut_off = [
    {'cut off factor': 0.2, 'gain factor': 7},
    {'cut off factor': 0.3, 'gain factor': 7},
    {'cut off factor': 0.4, 'gain factor': 7},
    {'cut off factor': 0.5, 'gain factor': 7}
]
#different gain factors for comparison
parameters_gain_factor = [
    {'cut off factor': 0.3, 'gain factor': 6},
    {'cut off factor': 0.3, 'gain factor': 7},
    {'cut off factor': 0.3, 'gain factor': 8},
    {'cut off factor': 0.3, 'gain factor': 9}
]

save_to_cut_off = "C:/Users/Lenovo/Desktop/image_material/27_sigmoid_function_plot" # path for saving
fig, ax = plt.subplots(figsize=(10, 6)) # creating figure
# loop through cut-off factors
for param in parameters_cut_off: # looping through the cut-off factors defined earlier
    y = sigmoid(x, param['cut off factor'], param['gain factor']) # computing the output based on index cut-off and gain factor
    ax.plot(x, y, label=f"cut off: {param['cut off factor']}, gain: {param['gain factor']}") # adding title with cut-off and gain factor
# create the plot
ax.set_title('Sigmoid Function - different cut-off factors')
ax.set_xlabel('x')
ax.set_ylabel('S(x)')
ax.legend()
plt.savefig(f'{save_to_cut_off}/sigmoid_cut-off.png', dpi=300) # saving the figure
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
# loop through cut-off factors
for param in parameters_gain_factor:
    y = sigmoid(x, param['cut off factor'], param['gain factor'])
    ax.plot(x, y, label=f"cut off: {param['cut off factor']}, gain: {param['gain factor']}")
# create the plot
ax.set_title('Sigmoid Function - different gain factors')
ax.set_xlabel('x')
ax.set_ylabel('S(x)')
ax.legend()
plt.savefig(f'{save_to_cut_off}/sigmoid_gain.png', dpi=300)
plt.show()
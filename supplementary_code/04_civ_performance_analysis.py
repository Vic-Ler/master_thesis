#FILE SUMMARY:
### extraction of civ data
### looping through data to create a data summary for each sequence in CIV1 and CIV2
### scatter plot of average correlation for each sequence and image variation in CIV1 and CIV2
### box plot of correlation data in CIV1 and CIV2
### scatter plot of f warnings in CIV1 and CIV2
### scatter plot of ff errors in CIV1 and CIV2

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
import netCDF4 as nc

#INITIAL EXPLORATION OF NC FILE
#investigation = nc.Dataset("C:/Users/Lenovo/Desktop/image_material/03_cropped_images/22_11_2021/10_min.civ/image_1-2.nc")
#print(investigation)
#print(investigation.variables)
#variable_names = [] # create array for storing variable names that are available
#for var_name in investigation.variables:
#    variable_names.append(var_name) # adding the variable names to the array
#print(variable_names) # printing overview of variables

#FUNCTION - CIV DATA TO DATAFRAME
def civ_dictionary(file_path):
    """
    Input: file path of specified civ file
    Returns: dictionary of respective civ file including all variable names
    """
    data = nc.Dataset(file_path) # extract data via file path
    variable_names = [] # initialise empty array for variable names in file
    for var_name in data.variables:
        variable_names.append(var_name) # loop through variable names and append the names to array
    dictionary = {} # initialise empty dictionary
    for var_name in variable_names:
        dictionary[var_name] = data.variables[var_name][:] # add entries for each variable name
    data.close() # close nc file
    print(variable_names) # print variables names as confirmation
    return dictionary # return final dictionary

#FUNCTION - EXTRACT DATA OF A SPECIFIC SEQUENCE FOR ALL IMAGE TYPES
def data_sequence(file_path_cropped, file_path_matched, file_path_hist_equal, file_path_adaptive_hist_equal, file_path_sigmoid):
    """
    Input: file path of the image sequence civ data for each image variation:
    cropped image civ data
    histogram matched civ data
    histogram equalized civ data
    adaptive histogram equalized civ data
    sigmoid civ data
    Returns: array containing dictionary-type data of each civ data file for each image variation for a specified sequence
    """
    #initialise empty arrays for civ data
    cropped = [] # array for civ data from original cropped images
    matched = [] # array for civ data from histo-matched images
    histo = [] # array for civ data from histo-equal images
    adapt = [] # array for civ data from CLAHE images
    sigmoid = [] # array for civ data from sigmoid images
    #append cropped image civ data to array
    for filename in os.listdir(file_path_cropped): # looping through the file path for civ data from original images
        if filename.endswith('.nc'): # extracting files that end with .nc
            file_path = os.path.join(file_path_cropped, filename) # joining the file path with .nc
            file_path = os.path.normpath(file_path) # creating a new path to the respective .nc files
            cropped.append(civ_dictionary(file_path)) # apply dictionary function before appending
    # append matched histogram image civ data to array
    for filename in os.listdir(file_path_matched):
        if filename.endswith('.nc'):
            file_path = os.path.join(file_path_matched, filename)
            file_path = os.path.normpath(file_path)
            matched.append(civ_dictionary(file_path))
    # append histogram equalization image civ data to array
    for filename in os.listdir(file_path_hist_equal):
        if filename.endswith('.nc'):
            file_path = os.path.join(file_path_hist_equal, filename)
            file_path = os.path.normpath(file_path)
            histo.append(civ_dictionary(file_path))
    # append adaptive histogram equalization image civ data to array
    for filename in os.listdir(file_path_adaptive_hist_equal):
        if filename.endswith('.nc'):
            file_path = os.path.join(file_path_adaptive_hist_equal, filename)
            file_path = os.path.normpath(file_path)
            adapt.append(civ_dictionary(file_path))
    # append sigmoid image civ data to array
    for filename in os.listdir(file_path_sigmoid):
        if filename.endswith('.nc'):
            file_path = os.path.join(file_path_sigmoid, filename)
            file_path = os.path.normpath(file_path)
            sigmoid.append(civ_dictionary(file_path))
    all_data = [cropped, matched, histo, adapt, sigmoid] # combining all the civ data
    return all_data

#extracting the paths of the civ data for the 24.12.2021
file_path_cropped = "C:/Users/Lenovo/Desktop/image_material/03_cropped_images/24_12_2021/10_min.civ"
file_path_matched = "C:/Users/Lenovo/Desktop/image_material/16_histo_matched_images/24_12_2021/10_min.civ"
file_path_hist_equal = "C:/Users/Lenovo/Desktop/image_material/18_histo_equal_images/24_12_2021/10_min.civ"
file_path_adaptive_hist_equal = "C:/Users/Lenovo/Desktop/image_material/20_adapt_histo_images/24_12_2021/10_min.civ"
file_path_sigmoid = "C:/Users/Lenovo/Desktop/image_material/22_sigmoid_filter_images/24_12_2021/10_min.civ"

#extracting and summarizing all civ data available for the 24.12.2021
civ_24_12_2021 = data_sequence(file_path_cropped,
                               file_path_matched,
                               file_path_hist_equal,
                               file_path_adaptive_hist_equal,
                               file_path_sigmoid)
#print(len(civ_24_12_2021)) #number of image variations: cropped, matched, histo, adapt, sigmoid
#print(len(civ_24_12_2021[1])) #number of civ data for each image variation: 1, 2, 3, 4 (5 images were available for 22.11.2021)
#print(len(civ_24_12_2021[1][1]['Civ2_C'])) #e.g. correlation data of second civ image in histogram matched variation

#FUNCTION - AVERAGE CORRELATION OF SEQUENCE FOR EACH IMAGE VARIATION
path_for_average_correlation_plot = "C:/Users/Lenovo/Desktop/image_material/28_average_correlation_comparison"
#23.10.2023 = plt.ylim([0.80, 0.88])
def scatter_plot_average_correlation(data_overview, file_name, save_to_path):
    """
    Input: data overview that was created using the previous functions, file path for saving and file name
    Returns: scatter plot of average correlation value for each civ data in respective file and each variation
    saved the resulting figure to specified file with file name
    """
    y_cropped = [] # initialising empty arrays to store average correlation for each civ data
    y_matched = []
    y_hist = []
    y_adapt = []
    y_sigmoid = []
    array_list = [y_cropped, y_matched, y_hist, y_adapt, y_sigmoid] # creating list i can use for indexing
    sequence_length = list(range(len(data_overview[0]))) # defining the number of image civ data in sequence
    x_labels = [f"image {num+1}" for num in sequence_length] # creating discrete labels
    for i in range(len(data_overview)): # looping through image variations
        for j in range(len(data_overview[i])): # within image variation loop through each data
            removed_error_flags = data_overview[i][j]['Civ1_C'][(data_overview[i][j]['Civ1_FF'] == 0) & (data_overview[i][j]['Civ1_F'] == 0)]
            # removing all the data from CIV_C that has non-zero values in FF or F
            array_list[i].append(np.mean(removed_error_flags)) # append mean of correlation of data
    plt.plot(x_labels, y_cropped, marker='o', linestyle='-', color='black', label='original') # creating scatter plot for each image variation
    plt.plot(x_labels, y_matched, marker='o', linestyle='-', color='#595959', label='histogram matched')
    plt.plot(x_labels, y_hist, marker='o', linestyle='-.', color='#A6A6A6', label='histogram equalized')
    plt.plot(x_labels, y_adapt, marker='o', linestyle=':', color='#275317', label='adaptive histogram equalized')
    plt.plot(x_labels, y_sigmoid, marker='o', linestyle='--', color='#51AE30', label='sigmoid')
    plt.legend(loc='lower left', fontsize=8) # adding legend
    plt.ylim([0.80, 0.88]) # setting coherent y limit for comparison
    plt.savefig(f'{save_to_path}/average_correlation_{file_name}.png', format='png', dpi=300)  # saving the figure
    plt.show() # plot to console

#implementing scatter plot for 24.12.2021
scatter_plot_average_correlation(civ_24_12_2021, "24_12_2021", path_for_average_correlation_plot)

#CREATING DATA OVERVIEW FOR ALL SEQUENCES
#doing this for all the image sequences (6 in total, 5 remaining)
#04.12.2021
file_path_cropped = "C:/Users/Lenovo/Desktop/image_material/03_cropped_images/04_12_2021/10_min.civ"
file_path_matched = "C:/Users/Lenovo/Desktop/image_material/16_histo_matched_images/04_12_2021/10_min.civ"
file_path_hist_equal = "C:/Users/Lenovo/Desktop/image_material/18_histo_equal_images/04_12_2021/10_min.civ"
file_path_adaptive_hist_equal = "C:/Users/Lenovo/Desktop/image_material/20_adapt_histo_images/04_12_2021/10_min.civ"
file_path_sigmoid = "C:/Users/Lenovo/Desktop/image_material/22_sigmoid_filter_images/04_12_2021/10_min.civ"
civ_04_12_2021 = data_sequence(file_path_cropped,
                               file_path_matched,
                               file_path_hist_equal,
                               file_path_adaptive_hist_equal,
                               file_path_sigmoid)
#22.11.2021 TO BE DONE
file_path_cropped = "C:/Users/Lenovo/Desktop/image_material/03_cropped_images/24_12_2021/10_min.civ"
file_path_matched = "C:/Users/Lenovo/Desktop/image_material/16_histo_matched_images/24_12_2021/10_min.civ"
file_path_hist_equal = "C:/Users/Lenovo/Desktop/image_material/18_histo_equal_images/24_12_2021/10_min.civ"
file_path_adaptive_hist_equal = "C:/Users/Lenovo/Desktop/image_material/20_adapt_histo_images/24_12_2021/10_min.civ"
file_path_sigmoid = "C:/Users/Lenovo/Desktop/image_material/22_sigmoid_filter_images/24_12_2021/10_min.civ"
civ_22_11_2021 = data_sequence(file_path_cropped,
                               file_path_matched,
                               file_path_hist_equal,
                               file_path_adaptive_hist_equal,
                               file_path_sigmoid)
#23.10.2023 01
file_path_cropped = "C:/Users/Lenovo/Desktop/image_material/03_cropped_images/23_10_2023_01/30_min.civ"
file_path_matched = "C:/Users/Lenovo/Desktop/image_material/16_histo_matched_images/23_10_2023_01/30_min.civ"
file_path_hist_equal = "C:/Users/Lenovo/Desktop/image_material/18_histo_equal_images/23_10_2023_01/30_min.civ"
file_path_adaptive_hist_equal = "C:/Users/Lenovo/Desktop/image_material/20_adapt_histo_images/23_10_2023_01/30_min.civ"
file_path_sigmoid = "C:/Users/Lenovo/Desktop/image_material/22_sigmoid_filter_images/23_10_2023_01/30_min.civ"
civ_23_10_2023_01 = data_sequence(file_path_cropped,
                               file_path_matched,
                               file_path_hist_equal,
                               file_path_adaptive_hist_equal,
                               file_path_sigmoid)
#23.10.2023 02
file_path_cropped = "C:/Users/Lenovo/Desktop/image_material/03_cropped_images/23_10_2023_02/30_min.civ"
file_path_matched = "C:/Users/Lenovo/Desktop/image_material/16_histo_matched_images/23_10_2023_02/30_min.civ"
file_path_hist_equal = "C:/Users/Lenovo/Desktop/image_material/18_histo_equal_images/23_10_2023_02/30_min.civ"
file_path_adaptive_hist_equal = "C:/Users/Lenovo/Desktop/image_material/20_adapt_histo_images/23_10_2023_02/30_min.civ"
file_path_sigmoid = "C:/Users/Lenovo/Desktop/image_material/22_sigmoid_filter_images/23_10_2023_02/30_min.civ"
civ_23_10_2023_02 = data_sequence(file_path_cropped,
                               file_path_matched,
                               file_path_hist_equal,
                               file_path_adaptive_hist_equal,
                               file_path_sigmoid)
#23.10.2023 03
file_path_cropped = "C:/Users/Lenovo/Desktop/image_material/03_cropped_images/23_10_2023_03/30_min.civ"
file_path_matched = "C:/Users/Lenovo/Desktop/image_material/16_histo_matched_images/23_10_2023_3/30_min.civ"
file_path_hist_equal = "C:/Users/Lenovo/Desktop/image_material/18_histo_equal_images/23_10_2023_03/30_min.civ"
file_path_adaptive_hist_equal = "C:/Users/Lenovo/Desktop/image_material/20_adapt_histo_images/23_10_2023_03/30_min.civ"
file_path_sigmoid = "C:/Users/Lenovo/Desktop/image_material/22_sigmoid_filter_images/23_10_2023_03/30_min.civ"
civ_23_10_2023_03 = data_sequence(file_path_cropped,
                               file_path_matched,
                               file_path_hist_equal,
                               file_path_adaptive_hist_equal,
                               file_path_sigmoid)

#applying scatter_plot_average_correlation function to all sequences
scatter_plot_average_correlation(civ_04_12_2021, "04_12_2021", path_for_average_correlation_plot)
scatter_plot_average_correlation(civ_22_11_2021, "22_11_2021", path_for_average_correlation_plot)
scatter_plot_average_correlation(civ_23_10_2023_01, "23_10_2023_01", path_for_average_correlation_plot)
scatter_plot_average_correlation(civ_23_10_2023_02, "23_10_2023_02", path_for_average_correlation_plot)
scatter_plot_average_correlation(civ_23_10_2023_03, "23_10_2023_03", path_for_average_correlation_plot)

#FUNCTION - BOX PLOT CORRELATION COMPARISON
path_for_box_plot = "C:/Users/Lenovo/Desktop/image_material/29_box_plot_comparison"
def box_plot_correlation(data_overview, file_name, save_to_path):
    """
    Input: data overview that was created using the previous functions, file path for saving and file name
    Returns: box plot of average correlation value for each civ data in respective file and each variation
    saved the resulting figure to specified file with file name
    """
    y_cropped = []  # initializing empty arrays to store average correlation for each civ data
    y_matched = []
    y_hist = []
    y_adapt = []
    y_sigmoid = []
    array_list = [y_cropped, y_matched, y_hist, y_adapt, y_sigmoid]  # creating list for indexing
    labels = ['Cropped', 'Histogram Matched', 'Histogram Equalized', 'Adaptive Histogram Equalized', 'Sigmoid'] # creating labels
    sequence_length = list(range(len(data_overview[0])))  # defining the number of image civ data in sequence
    x_labels = [f"image {num + 1}" for num in sequence_length]  # creating discrete labels
    for i in range(len(data_overview)):  # looping through image variations
        for j in range(len(data_overview[i])):  # within image variation loop through each data
            array_list[i].append(data_overview[i][j]['Civ1_C'])  # append correlation of data
    # creating a figure with subplots for each image type
    fig, axs = plt.subplots(1, len(array_list), figsize=(15, 6), sharey=True)
    # setting up figure - sharey means that all figures share same y axis
    for ax, data, label in zip(axs, array_list, labels): # looping through array list and labels
        ax.boxplot(data) # setting up box plot
        ax.set_title(label) # title based on labels defined earlier
        ax.set_xticklabels(x_labels, rotation=45, ha="right") # ticks (ha = horizontal alignment)
        ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='lightgray')  # adding grid lines
        ax.set_ylim(0, 1)  # y-axis range is from 0 to 1
        ax.set_yticks(np.linspace(0, 1, 21))  # more grid lines
    plt.tight_layout() # layout option
    plt.savefig(f'{save_to_path}/box_plot_{file_name}.png', format='png', dpi=300)  # saving the figure
    plt.show() # plot to console

#applying box_plot_correlation function to all sequences
box_plot_correlation(civ_22_11_2021, "22_11_2021", path_for_box_plot)
box_plot_correlation(civ_04_12_2021, "04_12_2021", path_for_box_plot)
box_plot_correlation(civ_24_12_2021, "24_12_2021", path_for_box_plot)
box_plot_correlation(civ_23_10_2023_01, "23_10_2023_01", path_for_box_plot)
box_plot_correlation(civ_23_10_2023_02, "23_10_2023_02", path_for_box_plot)
box_plot_correlation(civ_23_10_2023_03, "23_10_2023_03", path_for_box_plot)

#FUNCTION - SCATTER PLOT FOR FF ERRORS COMPARISON
#first i check the contained errors
#unique_values, counts = np.unique(civ_22_11_2021[0][0]['Civ1_F'], return_counts=True)
#print("Unique values:", unique_values)
#print("Counts:", counts)
#Unique values: [-2.  0.]
#Counts: [456 791]
#unique_values, counts = np.unique(civ_22_11_2021[0][0]['Civ1_FF'], return_counts=True)
#print("Unique values:", unique_values)
#print("Counts:", counts)
#Unique values: [ 0  1 20]
#Counts: [1132    6  109]

path_for_scatter_errors = "C:/Users/Lenovo/Desktop/image_material/30_scatter_errors_comparison"
def scatter_plot_errors_ff(data_overview, file_name, save_to_path):
    """
    Input: data overview that was created using the previous functions, file path for saving and file name
    Returns: scatter plot of ff errors for each civ1 data in respective file and each variation
    saved the resulting figure to specified file with file name
    """
    y_cropped = [] # initialising empty arrays to store error count for each civ data
    y_matched = []
    y_hist = []
    y_adapt = []
    y_sigmoid = []
    array_list = [y_cropped, y_matched, y_hist, y_adapt, y_sigmoid] # creating list i can use for indexing
    sequence_length = list(range(len(data_overview[0]))) # defining the number of image civ data in sequence
    x_labels = [f"image {num+1}" for num in sequence_length] # creating discrete labels
    for i in range(len(data_overview)): # looping through image variations
        for j in range(len(data_overview[i])): # within image variation loop through each data
            amount_total = len(data_overview[i][j]['Civ1_FF'])
            amount_flags = np.count_nonzero(data_overview[i][j]['Civ1_FF'])
            percentage = amount_flags/amount_total*100
            array_list[i].append(percentage) # append percentage of non-zero flags
    plt.plot(x_labels, y_cropped, marker='o', linestyle='-', color='black', label='original')  # creating scatter plot for each image variation
    plt.plot(x_labels, y_matched, marker='o', linestyle='-', color='#595959', label='histogram matched')
    plt.plot(x_labels, y_hist, marker='o', linestyle='-.', color='#A6A6A6', label='histogram equalized')
    plt.plot(x_labels, y_adapt, marker='o', linestyle=':', color='#275317', label='adaptive histogram equalized')
    plt.plot(x_labels, y_sigmoid, marker='o', linestyle='--', color='#51AE30', label='sigmoid')
    plt.legend(loc='lower left', fontsize=8) # adding legend
    plt.ylim([0, 50]) # setting the y limits for better comparison
    y_ticks = plt.gca().get_yticks() # address y ticks
    plt.gca().set_yticks(y_ticks)
    # set y ticks (otherwise the following step somehow didn't work which is why i had to do it explicitly)
    plt.gca().set_yticklabels([f'{int(tick)}%' for tick in y_ticks]) # adding percentages to ticks
    plt.savefig(f'{save_to_path}/errors_ff_{file_name}.png', format='png', dpi=300)  # saving the figure
    plt.show() # plot to console

#applying scatter_plot_errors_ff function to all image sequences
scatter_plot_errors_ff(civ_22_11_2021, "22_11_2021", path_for_scatter_errors)
scatter_plot_errors_ff(civ_04_12_2021, "04_12_2021", path_for_scatter_errors)
scatter_plot_errors_ff(civ_24_12_2021, "24_12_2021", path_for_scatter_errors)
scatter_plot_errors_ff(civ_23_10_2023_01, "23_10_2023_01", path_for_scatter_errors)
scatter_plot_errors_ff(civ_23_10_2023_02, "23_10_2023_02", path_for_scatter_errors)
scatter_plot_errors_ff(civ_23_10_2023_03, "23_10_2023_03", path_for_scatter_errors)

#FUNCTION - SCATTER PLOT FOR F WARNING COMPARISON
def scatter_plot_errors_f(data_overview, file_name, save_to_path):
    """
    Input: data overview that was created using the previous functions, file path for saving and file name
    Returns: scatter plot of ff errors for each civ1 data in respective file and each variation
    saved the resulting figure to specified file with file name
    """
    y_cropped = [] # initialising empty arrays to store error count for each civ data
    y_matched = []
    y_hist = []
    y_adapt = []
    y_sigmoid = []
    array_list = [y_cropped, y_matched, y_hist, y_adapt, y_sigmoid] # creating list i can use for indexing
    sequence_length = list(range(len(data_overview[0]))) # defining the number of image civ data in sequence
    x_labels = [f"image {num+1}" for num in sequence_length] # creating discrete labels
    for i in range(len(data_overview)): # looping through image variations
        for j in range(len(data_overview[i])): # within image variation loop through each data
            amount_total = len(data_overview[i][j]['Civ1_F']) # amount of CIV1_F data point occurences in data
            amount_flags = np.count_nonzero(data_overview[i][j]['Civ1_F']) # amount of all points that are not 0 in CIV1_F
            percentage = amount_flags / amount_total * 100 # calculating the percentage
            array_list[i].append(percentage) # append count of non-zero errors
    plt.plot(x_labels, y_cropped, marker='o', linestyle='-', color='black', label='original')  # creating scatter plot for each image variation
    plt.plot(x_labels, y_matched, marker='o', linestyle='-', color='#595959', label='histogram matched')
    plt.plot(x_labels, y_hist, marker='o', linestyle='-.', color='#A6A6A6', label='histogram equalized')
    plt.plot(x_labels, y_adapt, marker='o', linestyle=':', color='#275317', label='adaptive histogram equalized')
    plt.plot(x_labels, y_sigmoid, marker='o', linestyle='--', color='#51AE30', label='sigmoid')
    plt.legend(loc='lower left', fontsize=8)  # adding legend
    plt.ylim([0, 50]) # setting the y limits for better comparison
    y_ticks = plt.gca().get_yticks() # address y ticks
    plt.gca().set_yticks(y_ticks)
    # set y ticks (otherwise the following step somehow didn't work which is why i had to do it explicitly)
    plt.gca().set_yticklabels([f'{int(tick)}%' for tick in y_ticks]) # adding percentages to ticks
    plt.savefig(f'{save_to_path}/errors_f_{file_name}.png', format='png', dpi=300)  # saving the figure
    plt.show() # plot to console

#applying scatter_plot_errors_ff function to all image sequences
scatter_plot_errors_f(civ_22_11_2021, "22_11_2021", path_for_scatter_errors)
scatter_plot_errors_f(civ_04_12_2021, "04_12_2021", path_for_scatter_errors)
scatter_plot_errors_f(civ_24_12_2021, "24_12_2021", path_for_scatter_errors)
scatter_plot_errors_f(civ_23_10_2023_01, "23_10_2023_01", path_for_scatter_errors)
scatter_plot_errors_f(civ_23_10_2023_02, "23_10_2023_02", path_for_scatter_errors)
scatter_plot_errors_f(civ_23_10_2023_03, "23_10_2023_03", path_for_scatter_errors)

#FUNCTION FOR CORRELATION CIV 2
path_for_average_correlation_plot = "C:/Users/Lenovo/Desktop/image_material/28_average_correlation_comparison"
def scatter_plot_average_correlation_civ2(data_overview, file_name, save_to_path):
    """
    Input: data overview that was created using the previous functions, file path for saving and file name
    Returns: scatter plot of average correlation value for each civ2 data in respective file and each variation
    saved the resulting figure to specified file with file name
    """
    y_cropped = [] # initialising empty arrays to store average correlation for each civ data
    y_matched = []
    y_hist = []
    y_adapt = []
    y_sigmoid = []
    array_list = [y_cropped, y_matched, y_hist, y_adapt, y_sigmoid] # creating list i can use for indexing
    sequence_length = list(range(len(data_overview[0]))) # defining the number of image civ data in sequence
    x_labels = [f"image {num+1}" for num in sequence_length] # creating discrete labels
    for i in range(len(data_overview)): # looping through image variations
        for j in range(len(data_overview[i])): # within image variation loop through each data
            removed_error_flags = data_overview[i][j]['Civ2_C'][(data_overview[i][j]['Civ2_FF'] == 0) & (data_overview[i][j]['Civ2_F'] == 0)]
            array_list[i].append(np.mean(removed_error_flags)) # append mean of correlation of data
    plt.plot(x_labels, y_cropped, marker='o', linestyle='-', color='black',
             label='original')  # creating scatter plot for each image variation
    plt.plot(x_labels, y_matched, marker='o', linestyle='-', color='#595959', label='histogram matched')
    plt.plot(x_labels, y_hist, marker='o', linestyle='-.', color='#A6A6A6', label='histogram equalized')
    plt.plot(x_labels, y_adapt, marker='o', linestyle=':', color='#275317', label='adaptive histogram equalized')
    plt.plot(x_labels, y_sigmoid, marker='o', linestyle='--', color='#51AE30', label='sigmoid')
    plt.legend(loc='lower left', fontsize=8)  # adding legend
    plt.ylim([0.80, 0.88])
    plt.savefig(f'{save_to_path}/average_correlation_civ2_{file_name}.png', format='png', dpi=300)  # saving the figure
    plt.show()

#implementing scatter plot for 22.11.2021
#24.12.2021 = plt.ylim([0.80, 0.88])
scatter_plot_average_correlation_civ2(civ_22_11_2021, "22_11_2021", path_for_average_correlation_plot)
scatter_plot_average_correlation_civ2(civ_04_12_2021, "04_12_2021", path_for_average_correlation_plot)
scatter_plot_average_correlation_civ2(civ_24_12_2021, "24_12_2021", path_for_average_correlation_plot)
scatter_plot_average_correlation_civ2(civ_23_10_2023_01, "23_10_2023_01", path_for_average_correlation_plot)
scatter_plot_average_correlation_civ2(civ_23_10_2023_02, "23_10_2023_02", path_for_average_correlation_plot)
scatter_plot_average_correlation_civ2(civ_23_10_2023_03, "23_10_2023_03", path_for_average_correlation_plot)

#FUNCTION - BOX PLOT CORRELATION COMPARISON CIV 2
path_for_box_plot = "C:/Users/Lenovo/Desktop/image_material/29_box_plot_comparison"
def box_plot_correlation_civ2(data_overview, file_name, save_to_path):
    """
    Input: data overview that was created using the previous functions, file path for saving and file name
    Returns: box plot of average correlation value for each civ2 data in respective file and each variation
    saved the resulting figure to specified file with file name
    """
    y_cropped = []  # initializing empty arrays to store average correlation for each civ data
    y_matched = []
    y_hist = []
    y_adapt = []
    y_sigmoid = []
    array_list = [y_cropped, y_matched, y_hist, y_adapt, y_sigmoid]  # creating list for indexing
    labels = ['Cropped', 'Histogram Matched', 'Histogram Equalized', 'Adaptive Histogram Equalized', 'Sigmoid']
    sequence_length = list(range(len(data_overview[0])))  # defining the number of image civ data in sequence
    x_labels = [f"image {num + 1}" for num in sequence_length]  # creating discrete labels
    for i in range(len(data_overview)):  # looping through image variations
        for j in range(len(data_overview[i])):  # within image variation loop through each data
            array_list[i].append(data_overview[i][j]['Civ2_C'])  # append correlation of data
    # creating a figure with subplots for each image type
    fig, axs = plt.subplots(1, len(array_list), figsize=(15, 6), sharey=True)
    for ax, data, label in zip(axs, array_list, labels):
        ax.boxplot(data)
        ax.set_title(label)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='lightgray')  # adding grid lines
        ax.set_ylim(0, 1)  # y-axis range is from 0 to 1
        ax.set_yticks(np.linspace(0, 1, 21))  # more grid lines
    plt.tight_layout()
    plt.savefig(f'{save_to_path}/box_plot_civ2_{file_name}.png', format='png', dpi=300)  # saving the figure
    plt.show()

box_plot_correlation_civ2(civ_22_11_2021, "22_11_2021", path_for_box_plot)
box_plot_correlation_civ2(civ_04_12_2021, "04_12_2021", path_for_box_plot)
box_plot_correlation_civ2(civ_24_12_2021, "24_12_2021", path_for_box_plot)
box_plot_correlation_civ2(civ_23_10_2023_01, "23_10_2023_01", path_for_box_plot)
box_plot_correlation_civ2(civ_23_10_2023_02, "23_10_2023_02", path_for_box_plot)
box_plot_correlation_civ2(civ_23_10_2023_03, "23_10_2023_03", path_for_box_plot)

#FUNCTION - SCATTER PLOT FOR FF WARNING COMPARISON CIV 2
path_for_scatter_errors = "C:/Users/Lenovo/Desktop/image_material/30_scatter_errors_comparison"
def scatter_plot_errors_ff_civ2(data_overview, file_name, save_to_path):
    """
    Input: data overview that was created using the previous functions, file path for saving and file name
    Returns: scatter plot of ff errors for each civ2 data in respective file and each variation
    saved the resulting figure to specified file with file name
    """
    y_cropped = [] # initialising empty arrays to store error count for each civ data
    y_matched = []
    y_hist = []
    y_adapt = []
    y_sigmoid = []
    array_list = [y_cropped, y_matched, y_hist, y_adapt, y_sigmoid] # creating list i can use for indexing
    sequence_length = list(range(len(data_overview[0]))) # defining the number of image civ data in sequence
    x_labels = [f"image {num+1}" for num in sequence_length] # creating discrete labels
    for i in range(len(data_overview)): # looping through image variations
        for j in range(len(data_overview[i])): # within image variation loop through each data
            amount_total = len(data_overview[i][j]['Civ2_FF'])
            amount_flags = np.count_nonzero(data_overview[i][j]['Civ2_FF'])
            percentage = amount_flags / amount_total * 100
            array_list[i].append(percentage) # append count of non-zero errors
    plt.plot(x_labels, y_cropped, marker='o', linestyle='-', color='black', label='original')  # creating scatter plot for each image variation
    plt.plot(x_labels, y_matched, marker='o', linestyle='-', color='#595959', label='histogram matched')
    plt.plot(x_labels, y_hist, marker='o', linestyle='-.', color='#A6A6A6', label='histogram equalized')
    plt.plot(x_labels, y_adapt, marker='o', linestyle=':', color='#275317', label='adaptive histogram equalized')
    plt.plot(x_labels, y_sigmoid, marker='o', linestyle='--', color='#51AE30', label='sigmoid')
    plt.legend(loc='lower left', fontsize=8)  # adding legend
    plt.ylim([0, 50])
    y_ticks = plt.gca().get_yticks()
    plt.gca().set_yticks(y_ticks)
    plt.gca().set_yticklabels([f'{int(tick)}%' for tick in y_ticks])
    plt.savefig(f'{save_to_path}/errors_ff_civ2_{file_name}.png', format='png', dpi=300)  # saving the figure
    plt.show()

scatter_plot_errors_ff_civ2(civ_22_11_2021, "22_11_2021", path_for_scatter_errors)
scatter_plot_errors_ff_civ2(civ_04_12_2021, "04_12_2021", path_for_scatter_errors)
scatter_plot_errors_ff_civ2(civ_24_12_2021, "24_12_2021", path_for_scatter_errors)
scatter_plot_errors_ff_civ2(civ_23_10_2023_01, "23_10_2023_01", path_for_scatter_errors)
scatter_plot_errors_ff_civ2(civ_23_10_2023_02, "23_10_2023_02", path_for_scatter_errors)
scatter_plot_errors_ff_civ2(civ_23_10_2023_03, "23_10_2023_03", path_for_scatter_errors)

#FUNCTION - SCATTER PLOT FOR F WARNING COMPARISON CIV 1
def scatter_plot_errors_f_civ2(data_overview, file_name, save_to_path):
    """
    Input: data overview that was created using the previous functions, file path for saving and file name
    Returns: scatter plot of ff errors for each civ2 data in respective file and each variation
    saved the resulting figure to specified file with file name
    """
    y_cropped = [] # initialising empty arrays to store error count for each civ data
    y_matched = []
    y_hist = []
    y_adapt = []
    y_sigmoid = []
    array_list = [y_cropped, y_matched, y_hist, y_adapt, y_sigmoid] # creating list i can use for indexing
    sequence_length = list(range(len(data_overview[0]))) # defining the number of image civ data in sequence
    x_labels = [f"image {num+1}" for num in sequence_length] # creating discrete labels
    for i in range(len(data_overview)): # looping through image variations
        for j in range(len(data_overview[i])): # within image variation loop through each data
            amount_total = len(data_overview[i][j]['Civ2_F'])
            amount_flags = np.count_nonzero(data_overview[i][j]['Civ2_F'])
            percentage = amount_flags / amount_total * 100
            array_list[i].append(percentage) # append count of non-zero errors
    plt.plot(x_labels, y_cropped, marker='o', linestyle='-', color='black', label='original')  # creating scatter plot for each image variation
    plt.plot(x_labels, y_matched, marker='o', linestyle='-', color='#595959', label='histogram matched')
    plt.plot(x_labels, y_hist, marker='o', linestyle='-.', color='#A6A6A6', label='histogram equalized')
    plt.plot(x_labels, y_adapt, marker='o', linestyle=':', color='#275317', label='adaptive histogram equalized')
    plt.plot(x_labels, y_sigmoid, marker='o', linestyle='--', color='#51AE30', label='sigmoid')
    plt.legend(loc='lower left', fontsize=8)  # adding legend
    plt.ylim([0, 50])
    y_ticks = plt.gca().get_yticks()
    plt.gca().set_yticks(y_ticks)
    plt.gca().set_yticklabels([f'{int(tick)}%' for tick in y_ticks])
    plt.savefig(f'{save_to_path}/errors_f_civ2_{file_name}.png', format='png', dpi=300)  # saving the figure
    plt.show()

scatter_plot_errors_f_civ2(civ_22_11_2021, "22_11_2021", path_for_scatter_errors)
scatter_plot_errors_f_civ2(civ_04_12_2021, "04_12_2021", path_for_scatter_errors)
scatter_plot_errors_f_civ2(civ_24_12_2021, "24_12_2021", path_for_scatter_errors)
scatter_plot_errors_f_civ2(civ_23_10_2023_01, "23_10_2023_01", path_for_scatter_errors)
scatter_plot_errors_f_civ2(civ_23_10_2023_02, "23_10_2023_02", path_for_scatter_errors)
scatter_plot_errors_f_civ2(civ_23_10_2023_03, "23_10_2023_03", path_for_scatter_errors)
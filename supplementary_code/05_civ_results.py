#FILE SUMMARY:
### extraction of civ data
### conversion of axis to longitude and latitude
### conversion of vectors to meters/ second
### display final velocity fields on topography map

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

#PATHS ARE DEFINED HERE DEPENDING ON IMAGE PROCESSING VERSION
#in the thesis the wind fields derive from the civ data obtained by the sigmoid transformation images,
#however, you can also pick other data as a basis
#SIGMOID PATHS
path_22_11_2021 = "C:/Users/Lenovo/Desktop/image_material/22_sigmoid_filter_images/22_11_2021/10_min.civ"
path_04_12_2021 = "C:/Users/Lenovo/Desktop/image_material/22_sigmoid_filter_images/04_12_2021/10_min.civ"
path_24_12_2021 = "C:/Users/Lenovo/Desktop/image_material/22_sigmoid_filter_images/24_12_2021/10_min.civ"
path_23_10_2023_01 = "C:/Users/Lenovo/Desktop/image_material/22_sigmoid_filter_images/23_10_2023_01/30_min.civ"
path_23_10_2023_02 = "C:/Users/Lenovo/Desktop/image_material/22_sigmoid_filter_images/23_10_2023_02/30_min.civ"
path_23_10_2023_03 = "C:/Users/Lenovo/Desktop/image_material/22_sigmoid_filter_images/23_10_2023_03/30_min.civ"
#CLAHE PATHS
#path_22_11_2021 = "C:/Users/Lenovo/Desktop/image_material/20_adapt_histo_images/22_11_2021/10_min.civ"
#path_04_12_2021 = "C:/Users/Lenovo/Desktop/image_material/20_adapt_histo_images/04_12_2021/10_min.civ"
#path_24_12_2021 = "C:/Users/Lenovo/Desktop/image_material/20_adapt_histo_images/24_12_2021/10_min.civ"
#path_23_10_2023_01 = "C:/Users/Lenovo/Desktop/image_material/20_adapt_histo_images/23_10_2023_01/30_min.civ"
#path_23_10_2023_02 = "C:/Users/Lenovo/Desktop/image_material/20_adapt_histo_images/23_10_2023_02/30_min.civ"
#path_23_10_2023_03 = "C:/Users/Lenovo/Desktop/image_material/20_adapt_histo_images/23_10_2023_03/30_min.civ"
#HISTOGRAM EQUALIZATION PATHS
#path_22_11_2021 = "C:/Users/Lenovo/Desktop/image_material/18_histo_equal_images/22_11_2021/10_min.civ"
#path_04_12_2021 = "C:/Users/Lenovo/Desktop/image_material/18_histo_equal_images/04_12_2021/10_min.civ"
#path_24_12_2021 = "C:/Users/Lenovo/Desktop/image_material/18_histo_equal_images/24_12_2021/10_min.civ"
#path_23_10_2023_01 = "C:/Users/Lenovo/Desktop/image_material/18_histo_equal_images/23_10_2023_01/30_min.civ"
#path_23_10_2023_02 = "C:/Users/Lenovo/Desktop/image_material/18_histo_equal_images/23_10_2023_02/30_min.civ"
#path_23_10_2023_03 = "C:/Users/Lenovo/Desktop/image_material/18_histo_equal_images/23_10_2023_03/30_min.civ"
#MATCHED HISTOGRAM PATHS
#path_22_11_2021 = "C:/Users/Lenovo/Desktop/image_material/16_histo_matched_images/22_11_2021/10_min.civ"
#path_04_12_2021 = "C:/Users/Lenovo/Desktop/image_material/16_histo_matched_images/04_12_2021/10_min.civ"
#path_24_12_2021 = "C:/Users/Lenovo/Desktop/image_material/16_histo_matched_images/24_12_2021/10_min.civ"
#path_23_10_2023_01 = "C:/Users/Lenovo/Desktop/image_material/16_histo_matched_images/23_10_2023_01/30_min.civ"
#path_23_10_2023_02 = "C:/Users/Lenovo/Desktop/image_material/16_histo_matched_images/23_10_2023_02/30_min.civ"
#path_23_10_2023_03 = "C:/Users/Lenovo/Desktop/image_material/16_histo_matched_images/23_10_2023_3/30_min.civ"
#ORIGINAL PATHS
#path_22_11_2021 = "C:/Users/Lenovo/Desktop/image_material/03_cropped_images/22_11_2021/10_min.civ"
#path_04_12_2021 = "C:/Users/Lenovo/Desktop/image_material/03_cropped_images/04_12_2021/10_min.civ"
#path_24_12_2021 = "C:/Users/Lenovo/Desktop/image_material/03_cropped_images/24_12_2021/10_min.civ"
#path_23_10_2023_01 = "C:/Users/Lenovo/Desktop/image_material/03_cropped_images/23_10_2023_01/30_min.civ"
#path_23_10_2023_02 = "C:/Users/Lenovo/Desktop/image_material/03_cropped_images/23_10_2023_02/30_min.civ"
#path_23_10_2023_03 = "C:/Users/Lenovo/Desktop/image_material/03_cropped_images/23_10_2023_03/30_min.civ"

#FUNCTION - EXTRACT IMAGE DATA
def extract_pixel_data_2d(folder_path):
    """
    Input: path to folder containing image files
    Returns: nested list with pixel data of each image in the folder in 2D
    Note: this function was created in 02_data_exploration
    """
    pattern = os.path.join(folder_path, '*.png')  # pattern for identifying png files in folder
    image_files = glob.glob(pattern)  # retrieving png files from folder
    pixel_data = [] # initialising empty array
    for i in image_files:
        image = imread(i, as_gray=True)  # reading the image
        pixel_data.append(image) # appending the 2D data to the initialised array
    print("pixel data saved") # since this codes take a while to run, confirmation after it has been executed
    return pixel_data

images_22_11_2021 = extract_pixel_data_2d(path_22_11_2021.replace(".civ", ""))
images_04_12_2021 = extract_pixel_data_2d(path_04_12_2021.replace(".civ", ""))
images_24_12_2021 = extract_pixel_data_2d(path_24_12_2021.replace(".civ", ""))
images_23_10_2023_01 = extract_pixel_data_2d(path_23_10_2023_01.replace(".civ", ""))
images_23_10_2023_02 = extract_pixel_data_2d(path_23_10_2023_02.replace(".civ", ""))
images_23_10_2023_03 = extract_pixel_data_2d(path_23_10_2023_03.replace(".civ", ""))

#EXPLORATION - FUNCTION - PLOT IMAGES WITH SPECIFIED INTENSITY SET TO 0
#def plot_threshold_zero(image_data, zero_up_to_pixel_intensity):
#    data_copy = np.copy(image_data)
#    data_copy[data_copy < zero_up_to_pixel_intensity] = 0
#    plt.imshow(data_copy, cmap='gray')
#    plt.axis('off')
#    plt.show()
#plot_threshold_zero(images_22_11_2021[0], 0.5)
#at 0.17 it seems like there is a good cut-off
#at 0.28 the next cut-off
#at 0.5 the highest cut-off
#FOR THE FIRST CIV FINAL PLOTS, I DIDN'T APPLY THRESHOLDING

#SIGMOID DATA USED FOR FINAL PLOTS
#FUNCTION - EXTRACT CIV DATA
def civ_dictionary_sequence(folder_path):
    """
    Input: folder path of specified civ files
    Returns: array with dictionary of each civ file including all variable names
    """
    civ_array = []
    for filename in os.listdir(folder_path): # looping through files in folder
        if filename.endswith('.nc'): # extracting files that end with .nc
            file_path = os.path.join(folder_path, filename) # file path
            file_path = os.path.normpath(file_path) # make sure / instead of \
            data = nc.Dataset(file_path) # extract data from file path
            variable_names = [] # create array for variable names contained in civ data
            for var_name in data.variables:
                variable_names.append(var_name) # extracting variables
            dictionary = {} # initialising dictionary
            for var_name in variable_names:
                dictionary[var_name] = data.variables[var_name][:] # adding dictionary entries based on variables
            data.close()
            print(variable_names) # printing variable names as confirmation
            civ_array.append(dictionary) # appending the final dictionary of the file to civ_array
    return civ_array # returning all dictionaries from files in the folder within an array

#22_11_2021
civ_22_11_2021_sequence = civ_dictionary_sequence(path_22_11_2021)
#print(len(civ_22_11_2021_sequence)) # 4 images in sequence
#civ_22_11_2021_sequence[0] # first dictionary of first civ data
#04_12_2021
civ_04_12_2021_sequence = civ_dictionary_sequence(path_04_12_2021)
#24_12_2021
civ_24_12_2021_sequence = civ_dictionary_sequence(path_24_12_2021)
#23_10_2023 01
civ_23_10_2023_01_sequence = civ_dictionary_sequence(path_23_10_2023_01)
#23_10_2023 02
civ_23_10_2023_02_sequence = civ_dictionary_sequence(path_23_10_2023_02)
#23_10_2023 03
civ_23_10_2023_03_sequence = civ_dictionary_sequence(path_23_10_2023_03)

#FUNCTION - PLOT CIV RESULT
def plot_civ_02(array_sequence, condition="['Civ2_C'] == 0"):
    """
    Input: array of all civ data for an image sequence (note: CIV 2), optional: condition that is highlighted
    Returns: displayed map of vectors for CIV 2 (won't work for CIV 1)
    """
    for i in range(len(array_sequence)): # looping through sequence
        mask = eval(f"array_sequence[i]{condition}") # for each CIV data file, mask is created based on condition
        x = array_sequence[i]['Civ2_X'] # extracting the CIV data of each file
        y = array_sequence[i]['Civ2_Y']
        u = array_sequence[i]['Civ2_U_smooth']
        v = array_sequence[i]['Civ2_V_smooth']
        plt.quiver(x, y, u, v, alpha=0.5, color='black', scale=0.5, scale_units='xy', angles='xy') # plotting general CIV data
        plt.quiver(x[mask], y[mask], u[mask], v[mask], alpha=0.5, color='red', scale=0.5, scale_units='xy', angles='xy') # plotting vectors of condition in red
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

plot_civ_02(civ_04_12_2021_sequence)
plot_civ_02(civ_04_12_2021_sequence, "['Civ2_FF'] != 0")
plot_civ_02(civ_22_11_2021_sequence)
plot_civ_02(civ_04_12_2021_sequence)
plot_civ_02(civ_24_12_2021_sequence)
plot_civ_02(civ_23_10_2023_01_sequence)
plot_civ_02(civ_23_10_2023_02_sequence)
plot_civ_02(civ_23_10_2023_03_sequence)

#FUNCTION - REMOVE ERRORS (FF) FROM DATA
def remove_errors_ff(array_sequence):
    """
    Input: array of all civ data for an image sequence (note: CIV 2)
    Returns: a copy-like array of the input array with all ff removed that are not 0
    """
    filtered_sequence = []  # new array for storing updated values
    for i in range(len(array_sequence)): # looping through the array sequence to extract each file
        data = array_sequence[i] # data for each file
        remove_ff = data['Civ2_FF'] != 0 # condition to remove
        mask = np.ones(data['Civ2_FF'].shape, dtype=bool) # mask for condition
        mask &= ~remove_ff # filling my mask with booleans based on condition
        new_data = {} # storing new dictionary data for each file
        for key in data.keys():
            if data[key].shape == data['Civ2_FF'].shape: # only apply the condition to variables of same shape
                new_data[key] = data[key][mask]
            else:
                new_data[key] = data[key]  # if not of same shape, keep old variables
        filtered_sequence.append(new_data)  # append the file data to the sequence array
    return filtered_sequence # return array

#22_11_2021
civ_22_11_2021_sequence_updated = remove_errors_ff(civ_22_11_2021_sequence)
#04_12_2021
civ_04_12_2021_sequence_updated = remove_errors_ff(civ_04_12_2021_sequence)
#24_12_2021
civ_24_12_2021_sequence_updated = remove_errors_ff(civ_24_12_2021_sequence)
#23_10_2023 01
civ_23_10_2023_01_sequence_updated = remove_errors_ff(civ_23_10_2023_01_sequence)
#23_10_2023 02
civ_23_10_2023_02_sequence_updated = remove_errors_ff(civ_23_10_2023_02_sequence)
#23_10_2023 03
civ_23_10_2023_03_sequence_updated = remove_errors_ff(civ_23_10_2023_03_sequence)

#FUNCTION - CONVERSION TO LATITUDE AND LONGITUDE
def lat_long(array_sequence, crop_x_start, crop_x_end, crop_y_start, crop_y_end, scale_factor):
    """
    Input: array of all civ data for an image sequence (note: CIV 2), cropping borders of the image, scale_factor for degrees
    Returns: a copy-like array of the input array including two new entries: LAT and LONG
    Note: this assumes the cropped region is the same for all images in a sequence (which it should be)
    """
    updated_sequence = [] # array for storing all dictionaries of the updated sequence data
    #longitude and latitude borders of cropped image
    long_x_start_west = -180 + crop_x_start * scale_factor
    long_x_end_east = -180 + crop_x_end * scale_factor
    lat_y_start_south = -90 + crop_y_start * scale_factor
    lat_y_end_north = -90 + crop_y_end * scale_factor
    #pixel height and width of cropped image
    pix_height = crop_y_end - crop_y_start
    pix_width = crop_x_end - crop_x_start
    for i in range(len(array_sequence)): # for each civ file data, looping through the X and Y variables, converting them
        dictionary = array_sequence[i].copy()
        dictionary['LONG'] = long_x_start_west + ((long_x_end_east - long_x_start_west) * (array_sequence[i]['Civ2_X'] / pix_width))
        dictionary['LAT'] = lat_y_start_south + ((lat_y_end_north - lat_y_start_south) * (array_sequence[i]['Civ2_Y'] / pix_height))
        updated_sequence.append(dictionary) # appending each updated dictionary to the updated sequence
    return updated_sequence # returning the updated dictionaries

#22_11_2021
civ_22_11_2021_sequence_latlon = lat_long(civ_22_11_2021_sequence_updated,500, 2000, 1000, 2000, 0.0625)
#04_12_2021
civ_04_12_2021_sequence_latlon = lat_long(civ_04_12_2021_sequence_updated, 200, 1500, 1100, 1900, 0.0625)
#24_12_2021
civ_24_12_2021_sequence_latlon = lat_long(civ_24_12_2021_sequence_updated, 400, 1900, 900, 1900, 0.0625)
#23_10_2023 01
civ_23_10_2023_01_sequence_latlon = lat_long(civ_23_10_2023_01_sequence_updated, 1200, 2500, 1000, 2000, 0.0625)
#23_10_2023 02
civ_23_10_2023_02_sequence_latlon = lat_long(civ_23_10_2023_02_sequence_updated, 900, 2000, 1100, 2000, 0.0625)
#23_10_2023 03
civ_23_10_2023_03_sequence_latlon = lat_long(civ_23_10_2023_03_sequence_updated, 500, 1300, 1200, 2000, 0.0625)

#cropped regions taken from 01_data_extraction file
#['22.11.2021', (500, 2000, 1000, 2000)],
#['04.12.2021', (200, 1500, 1100, 1900)],
#['24.12.2021', (400, 1900, 900, 1900)],
#['23.10.2023 - 05:40-06:40', (1200, 2500, 1000, 2000)],
#['23.10.2023 - 08:18-09:18', (900, 2000, 1100, 2000)],
#['23.10.2023 - 10:56-11:56', (500, 1300, 1200, 2000)]
#see if it worked
#print(civ_22_11_2021_sequence_latlon[0]['LONG'])

#FUNCTION - CONVERSION OF PIXEL VECTORS TO METERS PER SECOND
def velocity_fields(array_sequence, crop_x_start, crop_x_end, crop_y_start, crop_y_end, scale_factor, time_difference_in_min):
    """
    Input: array of all civ data for an image sequence (note: CIV 2), cropping borders of the image, scale_factor for degrees
    Returns: a copy-like array of the input array including two new entries: U in m/s and V in m/s and magnitude
    Note: this assumes the cropped region is the same for all images in a sequence (which it should be)
    For input U AND V, u and v smooth are used
    Latitude and Longitude need to be included (previous function executed)
    """
    updated_sequence = [] # array for storing all dictionaries of the updated sequence data
    #longitude and latitude borders of cropped image
    long_x_start_west = -180 + crop_x_start * scale_factor
    long_x_end_east = -180 + crop_x_end * scale_factor
    lat_y_start_south = -90 + crop_y_start * scale_factor
    lat_y_end_north = -90 + crop_y_end * scale_factor
    for i in range(len(array_sequence)): # for each civ file data, looping through the X and Y variables, converting them
        dictionary = array_sequence[i].copy()
        dictionary['V_CON'] = (array_sequence[i]['Civ2_V_smooth'] / (time_difference_in_min * 60)) * ((lat_y_end_north - lat_y_start_south) / ((lat_y_end_north - lat_y_start_south) / scale_factor)) * (np.pi / 180) * 3389500
        dictionary['U_CON'] = (array_sequence[i]['Civ2_U_smooth']  / (time_difference_in_min * 60)) * ((long_x_end_east - long_x_start_west) / ((lat_y_end_north - lat_y_start_south) / scale_factor)) * (np.pi / 180) * 3389500 * abs(np.cos(array_sequence[i]['LAT']))
        dictionary['MAG'] = np.sqrt(dictionary['U_CON']**2 + dictionary['V_CON']**2)
        updated_sequence.append(dictionary) # appending each updated dictionary to the updated sequence
    return updated_sequence # returning the updated dictionaries

#22_11_2021
civ_22_11_2021_sequence_final = velocity_fields(civ_22_11_2021_sequence_latlon,500, 2000, 1000, 2000, 0.0625, 10)
#04_12_2021
civ_04_12_2021_sequence_final = velocity_fields(civ_04_12_2021_sequence_latlon, 200, 1500, 1100, 1900, 0.0625, 10)
#24_12_2021
civ_24_12_2021_sequence_final = velocity_fields(civ_24_12_2021_sequence_latlon, 400, 1900, 900, 1900, 0.0625, 10)
#23_10_2023 01
civ_23_10_2023_01_sequence_final = velocity_fields(civ_23_10_2023_01_sequence_latlon, 1200, 2500, 1000, 2000, 0.0625, 30)
#23_10_2023 02
civ_23_10_2023_02_sequence_final = velocity_fields(civ_23_10_2023_02_sequence_latlon, 900, 2000, 1100, 2000, 0.0625, 30)
#23_10_2023 03
civ_23_10_2023_03_sequence_final = velocity_fields(civ_23_10_2023_03_sequence_latlon, 500, 1300, 1200, 2000, 0.0625, 30)

#FUNCTION - DISPLAYING FINAL VELOCITY FIELDS ON TOPOGRAPHY MAP
topography_path = "C:/Users/Lenovo/Desktop/image_material/mola32.nc"
path_for_velocity_visualisation = "C:/Users/Lenovo/Desktop/image_material/32_velocity_fields"

def velocity_plot(array_sequence, topography_path, file_name,  save_to_path):
    """
    Input:array of civ data in a sequence, path to topography map, file name and path for saving
    Returns: visualisation of velocity fields with color bar for magnitude and altitude and saved figure
    """
    #extract and rearrange topography data
    topography_dataset = nc.Dataset(topography_path)  # open topography map
    center = len(topography_dataset.variables['longitude'][:]) // 2  # determine center of map horizontally
    left_half = topography_dataset.variables['alt'][:][:, :center]  # extract left half
    right_half = topography_dataset.variables['alt'][:][:, center:]  # extract right half
    altitudes = np.hstack((right_half, left_half))  # switch left and right half
    longitudes = topography_dataset.variables['longitude'][:] - 180  # change range of longitudes
    latitudes = topography_dataset.variables['latitude'][:]  # extract latitudes
    #plot velocity fields on topography map
    for i in range(len(array_sequence)):
        #extract respective range from topography map
        lon_mask = (longitudes >= np.min(array_sequence[i]['LONG'])) & (longitudes <= np.max(array_sequence[i]['LONG']))
        lat_mask = (latitudes >= np.min(array_sequence[i]['LAT'])) & (latitudes <= np.max(array_sequence[i]['LAT']))
        extracted_longitudes = longitudes[lon_mask]
        extracted_latitudes = latitudes[lat_mask]
        extracted_altitudes = altitudes[np.ix_(lat_mask, lon_mask)]
        #plot velocity fields on top of topography extraction
        fig, ax = plt.subplots()
        map_plot = ax.contourf(extracted_longitudes, extracted_latitudes, extracted_altitudes, cmap='OrRd')
        cbar_map = fig.colorbar(map_plot, ax=ax, orientation='horizontal', pad = 0.0, aspect=50)
        cbar_map.set_label('Altitude (meters)', fontsize=7, color='#940000')
        cbar_map.ax.tick_params(labelsize=7)
        vectors = ax.quiver(array_sequence[i]['LONG'], array_sequence[i]['LAT'], array_sequence[i]['U_CON'],
                            array_sequence[i]['V_CON'], array_sequence[i]['MAG'], cmap='viridis', scale_units='xy')
        cbar_vectors = fig.colorbar(vectors, ax=ax, orientation='horizontal', pad = 0.2, aspect=50)
        cbar_vectors.set_label('Magnitude (meters / second)', fontsize=7, color='#482576')
        cbar_vectors.ax.tick_params(labelsize=7)
        vectors.set_clim(0, 90)  # set color limits for magnitude
        plt.xlabel('Longitude', fontsize=7)
        plt.ylabel('Latitude', fontsize=7)
        plt.xticks(ticks=plt.xticks()[0], labels=[f'{tick}°' for tick in plt.xticks()[0]])
        plt.yticks(ticks=plt.yticks()[0], labels=[f'{tick}°' for tick in plt.yticks()[0]])
        plt.xlim(np.min(array_sequence[i]['LONG']), np.max(array_sequence[i]['LONG']))
        plt.ylim(np.min(array_sequence[i]['LAT']), np.max(array_sequence[i]['LAT']))
        ax.set_aspect('equal', 'box')
        fig.tight_layout()
        plt.savefig(f'{save_to_path}/velocity_fields_{file_name}_{i}.png', format='png', dpi=300)  # saving the figure
        plt.show()

velocity_plot(civ_22_11_2021_sequence_final, topography_path, "22_11_2021", path_for_velocity_visualisation)
velocity_plot(civ_04_12_2021_sequence_final, topography_path, "04_12_2021", path_for_velocity_visualisation)
velocity_plot(civ_24_12_2021_sequence_final, topography_path, "24_12_2021", path_for_velocity_visualisation)
velocity_plot(civ_23_10_2023_01_sequence_final, topography_path, "23_10_2023_01", path_for_velocity_visualisation)
velocity_plot(civ_23_10_2023_02_sequence_final, topography_path, "23_10_2023_02", path_for_velocity_visualisation)
velocity_plot(civ_23_10_2023_03_sequence_final, topography_path, "23_10_2023_03", path_for_velocity_visualisation)

#FUNCTION - ZONAL MEAN (actually meridonal mean)
path_for_zonal_means = "C:/Users/Lenovo/Desktop/image_material/31_zonal_mean"
def zonal_mean(array_sequence, file_name, save_to_path, labels_as_array):
    """
    Input: array of CIV data in a sequence, file name and path for saving, label names for the individual velocity fields
    Returns: visualisation of zonal means of each image in the sequence plotted together
    """
    zonal_means = []  # empty list for zonal means
    for i in range(len(array_sequence)):
        magnitude_data = array_sequence[i][
            'V_CON']  # extracting all the magnitude data from the respective velocity field
        latitude_data = np.round(array_sequence[i]['LAT'])  # extracting the latitudes and rounding to integers
        latitude_buckets = np.unique(latitude_data)  # creating a list of unique latitudes
        zonal_mean_data = []  # initializing list for zonal mean magnitudes
        for bucket in latitude_buckets:
            latitude_bucket_values = magnitude_data[
                latitude_data == bucket]  # extracting data that belongs to a latitude bucket
            zonal_mean_data.append(
                np.mean(latitude_bucket_values))  # appending the mean magnitude for each latitude bucket
        zonal_means.append((latitude_buckets, zonal_mean_data))  # storing latitudes and corresponding means as a tuple
    color_list = ['red', 'blue', 'green', 'orange', 'purple']  # colors for individual image sequences
    all_latitudes = []
    all_means = []
    for latitudes, means in zonal_means:
        all_latitudes.extend(latitudes) # adding all data to one array flattened
        all_means.extend(means)
    dataframe = pd.DataFrame({'Latitudes': all_latitudes, 'Means': all_means}) # dataframe makes calculation easier here
    overall_mean_values = dataframe.groupby('Latitudes')['Means'].mean().reset_index()  # calculating the mean for grouped latitudes
    avg_latitudes = overall_mean_values['Latitudes'].tolist() # converting it to list again
    avg_means = overall_mean_values['Means'].tolist()
    for i in range(len(zonal_means)):
        latitudes, means = zonal_means[i]
        plt.plot(means, latitudes, linestyle='dotted', linewidth=0.5, color=color_list[i], label=labels_as_array[i])
    plt.plot(avg_means, avg_latitudes, linestyle='-', linewidth=1, color='black', label='Average') # overall average
    plt.xlabel('Zonal mean velocity (m/s)')
    plt.ylabel('Latitude')
    plt.legend(loc='lower right')
    plt.xlim([-20,20])
    plt.ylim([-34, 35])
    plt.savefig(f'{save_to_path}/zonal_mean_{file_name}.png', format='png', dpi=300)  # saving the figure
    plt.show()

labels_22_11_2021 = ['14:16:52 - 14:26:52', '14:26:52 - 14:36:52', '14:36:52 - 14:46:52', '14:46:52 - 14:56:52']
zonal_mean(civ_22_11_2021_sequence_final, '22_11_2021', path_for_zonal_means, labels_22_11_2021)
labels_04_12_2021 = ['00:31:53 - 00:41:53', '00:41:53 - 00:51:53', '00:51:53 - 01:01:53', '01:01:53 - 01:11:53']
zonal_mean(civ_04_12_2021_sequence_final, '04_12_2021', path_for_zonal_means, labels_04_12_2021)
labels_24_12_2021 = ['10:30:07 - 10:40:06', '10:40:06 - 10:50:06', '10:50:06 - 11:00:06', '11:00:06 - 11:10:06', '11:10:06 - 11:20:06']
zonal_mean(civ_24_12_2021_sequence_final, '24_12_2021', path_for_zonal_means, labels_24_12_2021)
labels_23_10_2023_01 = ['05:40:13 - 06:10:16', '06:10:16 - 06:40:19']
zonal_mean(civ_23_10_2023_01_sequence_final, '23_10_2023_01', path_for_zonal_means, labels_23_10_2023_01)
labels_23_10_2023_02 = ['08:18:33 - 08:48:36', '08:48:36 - 09:18:39']
zonal_mean(civ_23_10_2023_02_sequence_final, '23_10_2023_02', path_for_zonal_means, labels_23_10_2023_02)
labels_23_10_2023_03 = ['10:56:50 - 11:26:53', '11:26:53 - 11:56:56']
zonal_mean(civ_23_10_2023_03_sequence_final, '23_10_2023_03', path_for_zonal_means, labels_23_10_2023_03)

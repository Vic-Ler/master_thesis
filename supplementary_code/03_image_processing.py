#FILE SUMMARY:
### extracting pixel data in 2d
### overview of images on a grid
### histogram matching
### histogram equalization
### adaptive histogram equalization
### sigmoid transformation (function and manually implemented)
### cumulative distribution function & histogram plots
### comparison between different transformations including histograms and CDFs

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
from skimage.util import img_as_ubyte

#LOADING THE IMAGE DATA
def extract_pixel_data_2d(folder_path):
    """
    Input: path to folder containing image files
    Returns: nested list with pixel data of each image in the folder in 2D
    Note: this function was created in 02_data_exploration
    """
    pattern = os.path.join(folder_path, '*.png')  # pattern for identifying png files in folder
    image_files = glob.glob(pattern)  # retrieving png files from folder
    pixel_data = [] # initialising empty array
    for i in image_files: # looping through image files
        image = imread(i, as_gray=True)  # reading the image
        pixel_data.append(image) # appending the 2D data to the initialised array
    print("pixel data saved") # since this codes take a while to run, confirmation after it has been executed
    return pixel_data

#paths to my folders containing respective image sequences
folder_path_22_11_2021 = "C:/Users/Lenovo/Desktop/image_material/03_cropped_images/22_11_2021/10_min"
folder_path_04_12_2021 = "C:/Users/Lenovo/Desktop/image_material/03_cropped_images/04_12_2021/10_min"
folder_path_24_12_2021 = "C:/Users/Lenovo/Desktop/image_material/03_cropped_images/24_12_2021/10_min"
folder_path_23_10_2023_01 = "C:/Users/Lenovo/Desktop/image_material/03_cropped_images/23_10_2023_01/30_min"
folder_path_23_10_2023_02 = "C:/Users/Lenovo/Desktop/image_material/03_cropped_images/23_10_2023_02/30_min"
folder_path_23_10_2023_03 = "C:/Users/Lenovo/Desktop/image_material/03_cropped_images/23_10_2023_03/30_min"

#creating nested lists for all image sequences in 2d
image_data_22_11_2021_2d = extract_pixel_data_2d(folder_path_22_11_2021)
image_data_04_12_2021_2d = extract_pixel_data_2d(folder_path_04_12_2021)
image_data_24_12_2021_2d = extract_pixel_data_2d(folder_path_24_12_2021)
image_data_23_10_2023_01_2d = extract_pixel_data_2d(folder_path_23_10_2023_01)
image_data_23_10_2023_02_2d = extract_pixel_data_2d(folder_path_23_10_2023_02)
image_data_23_10_2023_03_2d = extract_pixel_data_2d(folder_path_23_10_2023_03)

#FUNCTION - DISPLAYING IMAGE SEQUENCES ON A GRID
path_for_image_grid = "C:/Users/Lenovo/Desktop/image_material/14_image_grid"
def display_images_in_grid(images_in_2d, file_name, save_to_path, grid_width = 3):
    """
    Input: 2d image data (also listable) and the amount of columns on the grid
    Returns: a grid displaying all images of a sequence together displayed and saved
    """
    num_images = len(images_in_2d) # number of images in a sequence
    num_cols = min(num_images, grid_width) # number of columns based on input and number of images
    num_rows = (num_images + num_cols - 1) // num_cols # calculate number of rows
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4)) # set-up of grid
    axes = axes.flatten() # easier access of axes
    for ax in axes: # accessing axes depending on image number
        ax.axis('off') # axis are removed first
    for i, image in enumerate(images_in_2d): # looping through images
        axes[i].imshow(image, cmap='gray') # displaying the image
        axes[i].set_title(f'Image {i + 1}') # title of image
        axes[i].axis('off') # removing axis
    plt.tight_layout() # layout option
    plt.savefig(f'{save_to_path}/image_grid_{file_name}.png', format='png', dpi=300)
    # saving the figure
    plt.show() # plotting to console

#applying display_images_in_grid function on all sequences
display_images_in_grid(image_data_22_11_2021_2d, "22_11_2021", path_for_image_grid, 3)
display_images_in_grid(image_data_04_12_2021_2d, "04_12_2021", path_for_image_grid, 3)
display_images_in_grid(image_data_24_12_2021_2d, "24_12_2021", path_for_image_grid, 3)
display_images_in_grid(image_data_23_10_2023_01_2d, "23_10_2023_01", path_for_image_grid, 3)
display_images_in_grid(image_data_23_10_2023_02_2d, "23_10_2023_02", path_for_image_grid, 3)
display_images_in_grid(image_data_23_10_2023_03_2d, "23_10_2023_03", path_for_image_grid, 3)

#FUNCTION - HISTOGRAM MATCHING
def histogram_matching(images_in_2d, index_reference = 0):
    """
    Input: 2d image data including images of the image sequence and index of the image which should be used for reference
    Returns: the first histogram in the sequence unchanged and the remaining images matched to the histogram of the first image
    """
    num_images = len(images_in_2d) # number of images in sequence
    reference = images_in_2d[index_reference] # first image used as a reference for histogram matching
    matched = [] # initialising empty array for storing new images
    for i in range(num_images): # looping through all images
        if i == index_reference: # checks reference image for histogram matching
            matched.append(reference) # appending the reference image unchanged
        else:
            matched_histogram = exposure.match_histograms(images_in_2d[i], reference)
            # applying histogram matching to remaining images
            matched.append(matched_histogram)
            # appending result to the array
    return matched # return array containing the resulting images

#creating new nested arrays for my histogram matched image sequences
image_data_22_11_2021_histo_matched = histogram_matching(image_data_22_11_2021_2d, 0)
image_data_04_12_2021_histo_matched = histogram_matching(image_data_04_12_2021_2d, 0)
image_data_24_12_2021_histo_matched = histogram_matching(image_data_24_12_2021_2d, 2)
# 2 works better for this sequence
image_data_23_10_2023_01_histo_matched = histogram_matching(image_data_23_10_2023_01_2d, 0)
image_data_23_10_2023_02_histo_matched = histogram_matching(image_data_23_10_2023_02_2d, 0)
image_data_23_10_2023_03_histo_matched = histogram_matching(image_data_23_10_2023_03_2d, 0)

#FUNCTION - BOX PLOT FOR COMPARISON (BEFORE AND AFTER)
path_for_box_plot_comparison = "C:/Users/Lenovo/Desktop/image_material/15_box_plot_original_vs_matched"
def plot_boxplot_comparison(image_list_original_2d, image_list_matched_2d, file_name, save_to_path):
    """
    Input: 2d list of the original images and the histogram matched images
    Returns: boxplot for both the original images and the matched images displayed next to each other and figure saved
    Note: the two image arrays need to be of the same length and the execution takes a while
    """
    original_1d = [] # new array for 1d original images
    matched_1d = [] # new array for 1d matched images
    for i in range(len(image_list_original_2d)): # looping through the original image 2d data
        original_1d.append(image_list_original_2d[i].flatten()) # add flatted original images to array
        matched_1d.append(image_list_matched_2d[i].flatten()) # add flattened matched images to array
    fig, axes = plt.subplots(2, 1, figsize=(8, 12)) # creating figure
    # box plot of original images
    axes[0].boxplot(original_1d)  # creating box plot for original image sequence
    axes[0].set_xlabel('Image Numbers')  # number of images in sequence
    axes[0].set_ylabel('Pixel Intensities')  # pixel intensities
    axes[0].set_title('Original Images') # title
    # box plot of histogram matched images
    axes[1].boxplot(matched_1d)  # creating box plot for matched images
    axes[1].set_xlabel('Image Numbers')  # number of images in sequence
    axes[1].set_ylabel('Pixel Intensities')  # pixel intensities
    axes[1].set_title('Histogram-matched Images') # title
    plt.tight_layout() # layout option
    plt.savefig(f'{save_to_path}/box_plot_comparison_{file_name}.png', format='png', dpi=300)
    # saving the figure
    plt.show()  # displaying the figure in console

#applying plot_boxplot_comparison function to all image sequences
plot_boxplot_comparison(image_data_22_11_2021_2d, image_data_22_11_2021_histo_matched, "22_11_2021", path_for_box_plot_comparison)
plot_boxplot_comparison(image_data_04_12_2021_2d, image_data_04_12_2021_histo_matched, "04_12_2021", path_for_box_plot_comparison)
plot_boxplot_comparison(image_data_24_12_2021_2d, image_data_24_12_2021_histo_matched, "24_12_2021", path_for_box_plot_comparison)
plot_boxplot_comparison(image_data_23_10_2023_01_2d, image_data_23_10_2023_01_histo_matched, "23_10_2023_01", path_for_box_plot_comparison)
plot_boxplot_comparison(image_data_23_10_2023_02_2d, image_data_23_10_2023_02_histo_matched, "23_10_2023_02", path_for_box_plot_comparison)
plot_boxplot_comparison(image_data_23_10_2023_03_2d, image_data_23_10_2023_03_histo_matched, "23_10_2023_03", path_for_box_plot_comparison)

#FUNCTION - PLOTTING AND SAVING THE IMAGES OF AN ARRAY
path_for_matched_images = "C:/Users/Lenovo/Desktop/image_material/16_histo_matched_images"
def image_plot_save(image_array_2d, file_name, save_to_path):
    """
    Input: 2d list of images, file name and path where images should be saved
    Returns: image plots of images contained in array and saved images in folder
    """
    for i in range(len(image_array_2d)): # looping through each image data in sequence
        plt.imsave(f'{save_to_path}/matched_image_{file_name}_{i}.png', image_array_2d[i], cmap='gray', format='png', dpi=300)
        # saving the the images
        plt.imshow(image_array_2d[i], cmap = "gray") # plotting plain image data in grayscale
        plt.axis("off") # removing axis
        plt.show() # displaying in console

#saving all of the matched histogram images in my specified folder/ plotting them
image_plot_save(image_data_22_11_2021_histo_matched, "22_11_2021", path_for_matched_images)
image_plot_save(image_data_04_12_2021_histo_matched, "04_12_2021", path_for_matched_images)
image_plot_save(image_data_24_12_2021_histo_matched, "24_12_2021", path_for_matched_images)
image_plot_save(image_data_23_10_2023_01_histo_matched, "23_10_2023_01", path_for_matched_images)
image_plot_save(image_data_23_10_2023_02_histo_matched, "23_10_2023_02", path_for_matched_images)
image_plot_save(image_data_23_10_2023_03_histo_matched, "23_10_2023_03", path_for_matched_images)

#displaying the matched images on a grid aswell
path_for_matched_images_grid = "C:/Users/Lenovo/Desktop/image_material/17_histo_matched_images_grid"
display_images_in_grid(image_data_22_11_2021_histo_matched, "22_11_2021", path_for_matched_images_grid, 3)
display_images_in_grid(image_data_04_12_2021_histo_matched, "04_12_2021", path_for_matched_images_grid, 3)
display_images_in_grid(image_data_24_12_2021_histo_matched, "24_12_2021", path_for_matched_images_grid, 3)
display_images_in_grid(image_data_23_10_2023_01_histo_matched, "23_10_2023_01", path_for_matched_images_grid, 3)
display_images_in_grid(image_data_23_10_2023_02_histo_matched, "23_10_2023_02", path_for_matched_images_grid, 3)
display_images_in_grid(image_data_23_10_2023_03_histo_matched, "23_10_2023_03", path_for_matched_images_grid, 3)

#HISTOGRAM PLOT OVERVIEW AFTER HISTO MATCHING
path_for_saving_histograms_matched = "C:/Users/Lenovo/Desktop/image_material/24_hist_matched_histograms"
def overview_image_histograms(pixel_data_array, file_name, save_to_path, bin_size=10):
    """
    Input: pixel_data_array containing all images in a sequence
    Returns: distribution histogram of pixel intensities per image in overview displayed and saved
    """
    num_images = len(pixel_data_array) # number of images
    num_cols = min(num_images, 3) # max 3 columns
    num_rows = (num_images + num_cols - 1) // num_cols # calculate number of rows
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols * 6, num_rows * 4))  # adjusted figsize
    axes = axes.flatten()  # flatten axes array for easy access
    for i in range(num_images): # looping through image number
        image_pixel_data = pixel_data_array[i]  # access image data
        ax = axes[i] # defining the ax based on image number
        ax.hist(image_pixel_data.flatten(), bins=bin_size, color="black")  # plot histogram
        ax.set_title(f'Image {i + 1}') # title is image number starting from 1
        ax.set_xlabel('Pixel Intensities')
        ax.set_ylabel('Frequency')
    for j in range(num_images, len(axes)): # looping through each plot position
        axes[j].axis('off')  # hide unused subplots
    plt.tight_layout() # layout option
    plt.savefig(f'{save_to_path}/histogram_{file_name}.png', format='png', dpi=300)
    # save figure
    plt.show() # console plot

#applying overview_image_histograms function to all histogram matched image sequences
overview_image_histograms(image_data_22_11_2021_histo_matched, "22_11_2021", path_for_saving_histograms_matched, 100)
overview_image_histograms(image_data_04_12_2021_histo_matched, "04_12_2021", path_for_saving_histograms_matched, 100)
overview_image_histograms(image_data_24_12_2021_histo_matched, "24_12_2021", path_for_saving_histograms_matched, 100)
overview_image_histograms(image_data_23_10_2023_01_histo_matched, "23_10_2023_01", path_for_saving_histograms_matched, 100)
overview_image_histograms(image_data_23_10_2023_02_histo_matched, "23_10_2023_02", path_for_saving_histograms_matched, 100)
overview_image_histograms(image_data_23_10_2023_03_histo_matched, "23_10_2023_03", path_for_saving_histograms_matched, 100)

#CDF PLOT OVERVIEW AFTER HIST MATCHING
path_for_saving_cdf_overview_matched = "C:/Users/Lenovo/Desktop/image_material/25_hist_matched_cdf"
def overview_image_cdf(image_data, file_name, save_to_path, bin_size = 50):
    """
    Input: image array containing the images in a sequence
    Returns: cumulative distribution function of pixel intensities per image displayed and saved
    """
    num_images = len(image_data)  # number of images
    num_cols = min(num_images, 3)  # max 3 columns
    num_rows = (num_images + num_cols - 1) // num_cols  # calculate number of rows
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols * 6, num_rows * 4))  # adjusted fig size
    axes = axes.flatten()  # flatten axes array for easy iteration
    for i in range(num_images): # looping through image quantity
        image_pixel_data = image_data[i]  # access image data
        ax = axes[i] # accessing plot position
        # compute histogram and CDF
        cdf, bins = exposure.cumulative_distribution(image_pixel_data, nbins=bin_size) # computing CDF
        ax.plot(bins, cdf, color='black') # plot CDF
        ax.set_title(f'Image {i + 1}') # title is image number starting from 1
        ax.set_xlabel('Pixel Intensities')
        ax.set_ylabel('Cumulative Frequency')
    # turn off axes for unused subplots
    for j in range(num_images, len(axes)): # looping through each image position
        axes[j].axis('off') # removing axis
    plt.tight_layout() # layout option
    plt.savefig(f'{save_to_path}/cdf_{file_name}.png', format='png', dpi=300)
    # save with specified filename
    plt.show() # display in console

#saving the CDF of each image after histogram matching
overview_image_cdf(image_data_22_11_2021_histo_matched, "22_11_2021", path_for_saving_cdf_overview_matched)
overview_image_cdf(image_data_04_12_2021_histo_matched, "04_12_2021", path_for_saving_cdf_overview_matched)
overview_image_cdf(image_data_24_12_2021_histo_matched, "24_12_2021", path_for_saving_cdf_overview_matched)
overview_image_cdf(image_data_23_10_2023_01_histo_matched, "23_10_2023_01", path_for_saving_cdf_overview_matched)
overview_image_cdf(image_data_23_10_2023_02_histo_matched, "23_10_2023_02", path_for_saving_cdf_overview_matched)
overview_image_cdf(image_data_23_10_2023_03_histo_matched, "23_10_2023_03", path_for_saving_cdf_overview_matched)

#FUNCTION - HISTOGRAM EQUALIZATION
def histogram_equalization(image_array_2d):
    """
    Input: 2d image data
    Returns: histogram equalized image data
    """
    equalized_image = exposure.equalize_hist(image_array_2d) # using skimage function for straight-forward histogram equalization
    return equalized_image

#creating a list of the image sequence arrays
original_data = [image_data_22_11_2021_2d,
                image_data_04_12_2021_2d,
                image_data_24_12_2021_2d,
                image_data_23_10_2023_01_2d,
                image_data_23_10_2023_02_2d,
                image_data_23_10_2023_03_2d]

#creating a list of names for each image sequence and the file path for saving each image
file_names = ["22_11_2021", "04_12_2021", "24_12_2021", "23_10_2023_01", "23_10_2023_02", "23_10_2023_03"]
path_to_save_histo_equal_images = "C:/Users/Lenovo/Desktop/image_material/18_histo_equal_images"

#looping through the image sequences and images respectively, plotting and saving them
for i in range(len(original_data)): # looping through image sequences
    for j in range(len(original_data[i])): # looping through images in each sequence
        new_data = histogram_equalization(original_data[i][j]) # applying histo equalization on each image
        plt.imsave(f'{path_to_save_histo_equal_images}/hist_equal_{file_names[i]}_{j}.png', new_data, cmap='gray', format='png', dpi=300)
        # saving the image
        plt.imshow(new_data, cmap = "gray")
        # plotting the image in grayscale
        plt.axis("off") # removing axis
        plt.show() # display in console

#looping through the image sequences and images respectively, plotting each distribution histogram and saving it
path_to_save_histo_equal_images_histogram = "C:/Users/Lenovo/Desktop/image_material/19_histo_equal_images_histogram"

for i in range(len(original_data)): # looping through the image sequences
    for j in range(len(original_data[i])): # looping through images in each sequence
        new_data = histogram_equalization(original_data[i][j]) # histogram equalization
        plt.hist(new_data.flatten(), 100, color='black')
        # plotting histogram based on flattened data with 100 bins
        plt.xlabel("Pixel Intensities")
        plt.ylabel('Frequency')
        plt.savefig(f'{path_to_save_histo_equal_images_histogram}/hist_equal_histogram_{file_names[i]}_{j}.png', format='png', dpi=300)
        # saving the figure
        plt.show() # console plot

#FUNCTION - ADAPTIVE HISTOGRAM EQUALIZATION
def adaptive_histogram_equalization(image_array_2d, clip_limit=0.01, kernel_size=8):
    """
    Input: 2d image data, clip limit and kernel size
    Returns: adaptive histogram equalized image data
    """
    equalized_image = exposure.equalize_adapthist(image_array_2d, clip_limit=clip_limit, kernel_size=kernel_size)
    # using skimage exposure package
    return equalized_image

#looping through the image sequences and images respectively, plotting and saving them
path_to_save_adaptive_histo_equal_images = "C:/Users/Lenovo/Desktop/image_material/20_adapt_histo_images"

#i used clip limit 0.01 and kernel size 100 based on trial and error
for i in range(len(original_data)): # looping through image sequences
    for j in range(len(original_data[i])): # looping through images in each sequence
        new_data = adaptive_histogram_equalization(original_data[i][j], 0.01, 100)
        # computing CLAHE on each image
        plt.imsave(f'{path_to_save_adaptive_histo_equal_images}/adapt_hist_equal_{file_names[i]}_{j}.png', new_data, cmap='gray', format='png', dpi=300)
        # saving image
        plt.imshow(new_data, cmap = "gray") # plotting in grayscale
        plt.axis("off") # removing axis
        plt.show() # display in console

#looping through the image sequences and images respectively, plotting each distribution histogram and saving it
path_to_save_adaptive_histo_equal_histograms= "C:/Users/Lenovo/Desktop/image_material/21_adapt_histo_histograms"

for i in range(len(original_data)): # looping through the sequences
    for j in range(len(original_data[i])): # looping through the images
        new_data = adaptive_histogram_equalization(original_data[i][j],0.01, 100)
        # computing CLAHE
        plt.hist(new_data.flatten(), 100, color='black')
        # plotting histogram based on flattened data with 100 bins
        plt.xlabel("Pixel Intensities")
        plt.ylabel('Frequency')
        plt.savefig(f'{path_to_save_adaptive_histo_equal_histograms}/adapt_hist_equal_histogram_{file_names[i]}_{j}.png', format='png', dpi=300)
        # saving the figure
        plt.show() # displaying in console

#FUNCTION SIGMOID FILTER (SIGMOID FUNCTION AS MULTIPLICATIVE FACTOR)
def sigmoid_filter(image_array_2d, cutoff_factor, gain_factor, inverse=False):
    """
    Input: 2d image data, cut off factor, fain factor and inverse boolean
    Returns: sigmoid filtered image data
    """
    filtered_data = exposure.adjust_sigmoid(image_array_2d, cutoff=cutoff_factor, gain=gain_factor, inv=inverse)
    # function from skimage package
    return filtered_data

#testing parameters/ comparing different sigmoid filter parameters with original image
#cutoff_factor = 0.3 #shifts the curve horizontally
#gain_factor = 6 #controls the steepness of the curve
#test = sigmoid_filter(image_data_23_10_2023_03_histo_matched[0], cutoff_factor, gain_factor)
#plt.imshow(test, cmap='gray')
#plt.axis("off")
#plt.show()
#plt.imshow(image_data_23_10_2023_03_histo_matched[0], cmap='gray')
#plt.axis("off")
#plt.show()

path_sigmoid_images = "C:/Users/Lenovo/Desktop/image_material/22_sigmoid_filter_images"
#saving image sequence 22_11_2021 - cut-off: 0.3, gain: 8
cutoff_factor = 0.3 # shifts the curve horizontally
gain_factor = 8 # controls the steepness of the curve
for i in range(len(image_data_22_11_2021_2d)): # lopping through image sequence
    new_data = sigmoid_filter(image_data_22_11_2021_2d[i], cutoff_factor, gain_factor) # for each image sigmoid transformation
    plt.imsave(f'{path_sigmoid_images}/sigmoid_22_11_2021_{i}.png', new_data,
               cmap='gray', format='png', dpi=300)
    # saving the image
    plt.imshow(new_data, cmap='gray') # plotting in grayscale
    plt.axis("off") # removing axis
    plt.show() # display in console

# saving image sequence 04_12_2021 - cut-off: 0.3, gain: 6
cutoff_factor = 0.3 # shifts the curve horizontally
gain_factor = 6 # controls the steepness of the curve
for i in range(len(image_data_04_12_2021_2d)):
    new_data = sigmoid_filter(image_data_04_12_2021_2d[i], cutoff_factor, gain_factor)
    plt.imsave(f'{path_sigmoid_images}/sigmoid_04_12_2021_{i}.png', new_data,
                cmap='gray', format='png', dpi=300)
    plt.imshow(new_data, cmap='gray')
    plt.axis("off")
    plt.show()
# saving image sequence 24_12_2021 - cut-off: 0.3, gain: 8
cutoff_factor = 0.3 # shifts the curve horizontally
gain_factor = 8 # controls the steepness of the curve
for i in range(len(image_data_24_12_2021_2d)):
    new_data = sigmoid_filter(image_data_24_12_2021_2d[i], cutoff_factor, gain_factor)
    plt.imsave(f'{path_sigmoid_images}/sigmoid_24_12_2021_{i}.png', new_data,
                cmap='gray', format='png', dpi=300)
    plt.imshow(new_data, cmap='gray')
    plt.axis("off")
    plt.show()
# saving image sequence 23_10_2023_01 - cut-off: 0.4, gain: 7
cutoff_factor = 0.4 # shifts the curve horizontally
gain_factor = 7 # controls the steepness of the curve
for i in range(len(image_data_23_10_2023_01_2d)):
    new_data = sigmoid_filter(image_data_23_10_2023_01_2d[i], cutoff_factor, gain_factor)
    plt.imsave(f'{path_sigmoid_images}/sigmoid_23_10_2023_01_{i}.png', new_data,
                cmap='gray', format='png', dpi=300)
    plt.imshow(new_data, cmap='gray')
    plt.axis("off")
    plt.show()
# saving image sequence 23_10_2023_02 - cut-off: 0.3, gain: 6
cutoff_factor = 0.3 # shifts the curve horizontally
gain_factor = 6 # controls the steepness of the curve
for i in range(len(image_data_23_10_2023_02_2d)):
    new_data = sigmoid_filter(image_data_23_10_2023_02_2d[i], cutoff_factor, gain_factor)
    plt.imsave(f'{path_sigmoid_images}/sigmoid_23_10_2023_02_{i}.png', new_data,
                cmap='gray', format='png', dpi=300)
    plt.imshow(new_data, cmap='gray')
    plt.axis("off")
    plt.show()
# saving image sequence 23_10_2023_03 - cut-off: 0.3, gain: 6
cutoff_factor = 0.3 # shifts the curve horizontally
gain_factor = 6 # controls the steepness of the curve
for i in range(len(image_data_23_10_2023_03_2d)):
    new_data = sigmoid_filter(image_data_23_10_2023_03_2d[i], cutoff_factor, gain_factor)
    plt.imsave(f'{path_sigmoid_images}/sigmoid_23_10_2023_03_{i}.png', new_data,
                cmap='gray', format='png', dpi=300)
    plt.imshow(new_data, cmap='gray')
    plt.axis("off")
    plt.show()

#saving histograms of sigmoid filter
path_to_save_sigmoid_histograms = "C:/Users/Lenovo/Desktop/image_material/23_sigmoid_filter_histograms"

#saving image histograms 22_11_2021 - cut-off: 0.3, gain: 8
cutoff_factor = 0.3 # shifts the curve horizontally
gain_factor = 8 # controls the steepness of the curve
for i in range(len(image_data_22_11_2021_2d)):
    new_data = sigmoid_filter(image_data_22_11_2021_2d[i], cutoff_factor, gain_factor)
    plt.hist(new_data.flatten(), 100, color='black')
    plt.xlabel("Pixel Intensities")
    plt.ylabel('Frequency')
    plt.savefig(f'{path_to_save_sigmoid_histograms}/sigmoid_histo_22_11_2021_{i}.png',
                format='png', dpi=300)  # saving the figure
    plt.show()
# saving image histograms 04_12_2021 - cut-off: 0.3, gain: 6
cutoff_factor = 0.3 # shifts the curve horizontally
gain_factor = 6 # controls the steepness of the curve
for i in range(len(image_data_04_12_2021_2d)):
    new_data = sigmoid_filter(image_data_04_12_2021_2d[i], cutoff_factor, gain_factor)
    plt.hist(new_data.flatten(), 100, color='black')
    plt.xlabel("Pixel Intensities")
    plt.ylabel('Frequency')
    plt.savefig(f'{path_to_save_sigmoid_histograms}/sigmoid_histo_04_12_2021_{i}.png',
                format='png', dpi=300)  # saving the figure
    plt.show()
# saving image histograms 24_12_2021 - cut-off: 0.3, gain: 8
cutoff_factor = 0.3 # shifts the curve horizontally
gain_factor = 8 # controls the steepness of the curve
for i in range(len(image_data_24_12_2021_2d)):
    new_data = sigmoid_filter(image_data_24_12_2021_2d[i], cutoff_factor, gain_factor)
    plt.hist(new_data.flatten(), 100, color='black')
    plt.xlabel("Pixel Intensities")
    plt.ylabel('Frequency')
    plt.savefig(f'{path_to_save_sigmoid_histograms}/sigmoid_histo_24_12_2021_{i}.png',
                format='png', dpi=300)  # saving the figure
    plt.show()
# saving image histograms 23_10_2023_01 - cut-off: 0.4, gain: 7
cutoff_factor = 0.4 # shifts the curve horizontally
gain_factor = 7 # controls the steepness of the curve
for i in range(len(image_data_23_10_2023_01_2d)):
    new_data = sigmoid_filter(image_data_23_10_2023_01_2d[i], cutoff_factor, gain_factor)
    plt.hist(new_data.flatten(), 100, color='black')
    plt.xlabel("Pixel Intensities")
    plt.ylabel('Frequency')
    plt.savefig(f'{path_to_save_sigmoid_histograms}/sigmoid_histo_23_10_2023_01_{i}.png',
                format='png', dpi=300)  # saving the figure
    plt.show()
# saving image histograms 23_10_2023_02 - cut-off: 0.3, gain: 6
cutoff_factor = 0.3 # shifts the curve horizontally
gain_factor = 6 # controls the steepness of the curve
for i in range(len(image_data_23_10_2023_02_2d)):
    new_data = sigmoid_filter(image_data_23_10_2023_02_2d[i], cutoff_factor, gain_factor)
    plt.hist(new_data.flatten(), 100, color='black')
    plt.xlabel("Pixel Intensities")
    plt.ylabel('Frequency')
    plt.savefig(f'{path_to_save_sigmoid_histograms}/sigmoid_histo_23_10_2023_02_{i}.png',
                format='png', dpi=300)  # saving the figure
    plt.show()
# saving image histograms 23_10_2023_03 - cut-off: 0.3, gain: 6
cutoff_factor = 0.3 # shifts the curve horizontally
gain_factor = 6 # controls the steepness of the curve
for i in range(len(image_data_23_10_2023_03_2d)):
    new_data = sigmoid_filter(image_data_23_10_2023_03_2d[i], cutoff_factor, gain_factor)
    plt.hist(new_data.flatten(), 100, color='black')
    plt.xlabel("Pixel Intensities")
    plt.ylabel('Frequency')
    plt.savefig(f'{path_to_save_sigmoid_histograms}/sigmoid_histo_23_10_2023_03_{i}.png',
                format='png', dpi=300)  # saving the figure
    plt.show()

#FUNCTION - SIGMOID FILTER FROM SCRATCH FOR COMPARISON
def sigmoid_filter_scratch(image, cutoff=0.3, gain=7):
    """
    Input: 2d image data, cut off factor, gain factor
    Returns: sigmoid filtered image data
    """
    def sigmoid(x, cutoff=0.3, gain=7):
        return 1 / (1 + np.exp(-gain * (x - cutoff))) # simple sigmoid function
    sigmoid_values = sigmoid(image, cutoff, gain) # applying the function to each pixel
    return sigmoid_values # and that's it

#comparison skimage and scratch filter
skimage_sigmoid_filter = sigmoid_filter(image_data_22_11_2021_2d[0], 0.3, 8) # using skimage function
my_sigmoid_filter = sigmoid_filter_scratch(image_data_22_11_2021_2d[0], 0.3, 8) # using self-made function
plt.imshow(skimage_sigmoid_filter, cmap='gray') # plotting skimage result
plt.axis("off")
plt.show()
plt.imshow(my_sigmoid_filter, cmap='gray') # plotting my result
plt.axis("off")
plt.show()
#(exactly the same)

#FUNCTION - DISPLAY HISTOGRAM EQUALIZATION, AHE AND SIGMOID IN OVERVIEW
path_to_save_overview = "C:/Users/Lenovo/Desktop/image_material/26_overview_of_filters"
def overview_processing(image_array_2d, cut_off_ahe, kernel_ahe, cut_off_sigmoid, gain_sigmoid, nbins, file_name, path):
    """
    Input: 2d image data array including all images of a sequence,
    cut off factor for adaptive histogram filter,
    kernel for adaptive histogram filter,
    cut off factor for sigmoid filter,
    gain factor for sigmoid filter,
    bins for histogram;
    Returns: overview of original image,
    histogram equalized image,
    adaptive histogram equalized image and
    sigmoid transformed image
    including histogram and CDF
    """
    for i in range(len(image_array_2d)): # looping through each image sequence
        image_data = image_array_2d[i] # extracting the individual images from sequence
        # Original Image and CDF
        bins = np.linspace(0, 1, nbins + 1)
        # manually setting up the number of bins based on intensity range
        cdf, _ = exposure.cumulative_distribution(image_data, nbins) # obtaining CDF
        # Histogram Equalization
        plt_01_histogram_equalized = histogram_equalization(image_data)  # HE applied
        plt_01_cdf, _ = exposure.cumulative_distribution(plt_01_histogram_equalized, nbins) # CDF of HE
        # Adaptive Histogram Equalization
        plt_02_adaptive_histo_equal = adaptive_histogram_equalization(image_data, cut_off_ahe, kernel_ahe) # CLAHE applied
        plt_02_cdf, _ = exposure.cumulative_distribution(plt_02_adaptive_histo_equal, nbins) # CDF of CLAHE
        # Sigmoid Filter
        plt_03_sigmoid = sigmoid_filter(image_data, cut_off_sigmoid, gain_sigmoid) # Sigmoid applied
        plt_03_cdf, _ = exposure.cumulative_distribution(plt_03_sigmoid, nbins) # CDF of sigmoid
        # Figure
        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(16, 12)) # setting up figure size
        # Original Image
        axes[0, 0].imshow(image_data, cmap='gray') # original image data in grayscale
        axes[0, 0].set_title('Original Image') # title
        axes[0, 0].axis('off') # removing axis
        # Original Histogram
        counts, _ = np.histogram(image_data.ravel(), bins=bins) # ravel flattens data before creating histogram
        axes[0, 1].hist(image_data.ravel(), bins=bins, histtype='bar', color='black') # histogram with bins defined earlier
        axes[0, 1].set_title('Original Histogram') # title
        axes[0, 1].set_xlabel('Pixel Intensity')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_xlim(0, 1) # x axis limits
        # Original CDF
        ax2 = axes[0, 1].twinx()  # second y-axis
        ax2.plot(bins[:-1], cdf * counts.max(), color='red')  # scaled to fit histogram
        ax2.set_ylabel('CDF', color='red') # y axis label for CDF in red
        # Histogram Equalization
        axes[1, 0].imshow(plt_01_histogram_equalized, cmap='gray') # HE plot in grayscale
        axes[1, 0].set_title('Histogram Equalization') # title
        axes[1, 0].axis('off') # removing axis
        # Histogram Equalized Histogram
        plt_01_counts, _ = np.histogram(plt_01_histogram_equalized.ravel(), bins=bins) # defining the frequencies based on flattened data
        axes[1, 1].hist(plt_01_histogram_equalized.ravel(), bins=bins, histtype='bar', color='black') # CDF histogram
        axes[1, 1].set_title('Equalized Histogram') # title
        axes[1, 1].set_xlabel('Pixel Intensity')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_xlim(0, 1) # x axis limits
        # Histogram Equalized CDF
        ax2 = axes[1, 1].twinx()  # Create a secondary y-axis
        ax2.plot(bins[:-1], plt_01_cdf * plt_01_counts.max(), color='red') # slicing array to exclude last bin edge
        ax2.set_ylabel('CDF', color='red') # y axis for CDF in red
        # Adaptive Histogram Equalization
        axes[2, 0].imshow(plt_02_adaptive_histo_equal, cmap='gray') # CLAHE
        axes[2, 0].set_title('Adaptive Histogram Equalization') # title
        axes[2, 0].axis('off') # remove axis
        # Adaptive Histogram Equalized Histogram
        plt_02_counts, _ = np.histogram(plt_02_adaptive_histo_equal.ravel(), bins=bins) # frequencies for CLAHE histogram
        axes[2, 1].hist(plt_02_adaptive_histo_equal.ravel(), bins=bins, histtype='bar', color='black') # histogram
        axes[2, 1].set_title('Adaptive Equalized Histogram') # title
        axes[2, 1].set_xlabel('Pixel Intensity')
        axes[2, 1].set_ylabel('Frequency')
        axes[2, 1].set_xlim(0, 1) # y limits 0 to 1
        # Adaptive Histogram Equalized CDF
        ax2 = axes[2, 1].twinx() # creating second axis
        ax2.plot(bins[:-1], plt_02_cdf * plt_02_counts.max(), color='red') # accessing left bin edges
        ax2.set_ylabel('CDF', color='red') # y label in red
        # Sigmoid Filter
        axes[3, 0].imshow(plt_03_sigmoid, cmap='gray') # sigmoid plot in grayscale
        axes[3, 0].set_title('Sigmoid Filter') # title
        axes[3, 0].axis('off') # removed axis
        # Sigmoid Filter Histogram
        plt_03_counts, _ = np.histogram(plt_03_sigmoid.ravel(), bins=bins) # flattened frequency data for histogram creation
        axes[3, 1].hist(plt_03_sigmoid.ravel(), bins=bins, histtype='bar', color='black') # histogram
        axes[3, 1].set_title('Sigmoid Histogram') # title
        axes[3, 1].set_xlabel('Pixel Intensity')
        axes[3, 1].set_ylabel('Frequency')
        axes[3, 1].set_xlim(0, 1) # x limits
        # Sigmoid Filter CDF
        ax2 = axes[3, 1].twinx() # create secondary axis
        ax2.plot(bins[:-1], plt_03_cdf * plt_03_counts.max(), color='red') # accessing left bin edges
        ax2.set_ylabel('CDF', color='red') # CDF y axis in red
        # Adjust layout
        plt.tight_layout() # layout option
        plt.savefig(f'{path}/overview_{file_name}_{i}.png',
                    format='png', dpi=300)  # saving the figure
        plt.show() # display in console

# applying the overview_processing function to each image sequence
overview_processing(image_data_22_11_2021_2d, 0.01, 100, 0.3, 8, 300, "22_11_2021", path_to_save_overview)
overview_processing(image_data_04_12_2021_2d, 0.01, 100, 0.3, 6, 300, "04_12_2021", path_to_save_overview)
overview_processing(image_data_24_12_2021_2d, 0.01, 100, 0.3, 8, 300, "24_12_2021", path_to_save_overview)
overview_processing(image_data_23_10_2023_01_2d, 0.01, 100, 0.4, 7, 300, "23_10_2023_01", path_to_save_overview)
overview_processing(image_data_23_10_2023_02_2d, 0.01, 100, 0.3, 6, 300, "23_10_2023_02", path_to_save_overview)
overview_processing(image_data_23_10_2023_03_2d, 0.01, 100, 0.3, 6, 300, "23_10_2023_03", path_to_save_overview)
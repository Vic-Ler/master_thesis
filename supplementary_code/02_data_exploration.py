#FILE SUMMARY:
### overview of general image statistics for each image sequence
### extraction of 1D pixel data
### box plots of pixel intensities for each image sequence
### pixel intensity distribution histograms for each image
### cumulative distribution function plot for each image
### extraction of 2D pixel data
### visualisation of the fourier transform for each image
### visualisation of a high-pass filter for each image
### visualisation of each image in clustered intensities

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

#FUNCTION - GENERAL IMAGE STATISTICS CSV TABLE
#defining the paths to the respective image sequences
folder_path_22_11_2021 = "C:/Users/Lenovo/Desktop/image_material/03_cropped_images/22_11_2021"
folder_path_04_12_2021 = "C:/Users/Lenovo/Desktop/image_material/03_cropped_images/04_12_2021"
folder_path_24_12_2021 = "C:/Users/Lenovo/Desktop/image_material/03_cropped_images/24_12_2021"
folder_path_23_10_2023_01 = "C:/Users/Lenovo/Desktop/image_material/03_cropped_images/23_10_2023_01"
folder_path_23_10_2023_02 = "C:/Users/Lenovo/Desktop/image_material/03_cropped_images/23_10_2023_02"
folder_path_23_10_2023_03 = "C:/Users/Lenovo/Desktop/image_material/03_cropped_images/23_10_2023_03"
# combining the paths so that I can later use loops
folder_list = (folder_path_22_11_2021, folder_path_04_12_2021, folder_path_24_12_2021, folder_path_23_10_2023_01, folder_path_23_10_2023_02, folder_path_23_10_2023_03)
#note: the pixel intensity range has automatically been updated to 0 - 1 due to the way the data was saved during the data extraction step

def image_statistics_overview(folder_path):
    """
    Input: path to folder containing image files
    Returns: dataframe with an overview of general image statistics
    """
    pattern = os.path.join(folder_path, '*.png')  # pattern to identify png files
    image_files = glob.glob(pattern)  # retrieving image file paths in folder
    images_table = []  # initialize empty list for storing image data
    for i in image_files: # looping through the png files in folder
        image = imread(i, as_gray=True) # reading data
        img_data = [
            f'{i}', # name of file
            image.shape[0], # image height
            image.shape[1], # image width
            np.nanmin(image), # min intensity
            np.nanmax(image), # max intensity
            (np.nanmax(image) - np.nanmin(image)), # intensity range
            np.nanmean(image), # mean intensity
            np.nanmedian(image), # median intensity
            np.nanstd(image), # standard deviation
            np.nanvar(image), # variance
            pd.Series(image.flatten()).skew(), # skewness
            pd.Series(image.flatten()).kurtosis() # kurtosis
        ]
        images_table.append(img_data) # appending image data to table
    columns = ['File Name','Pixels-Y', 'Pixels-X', 'Intensity Min', 'Intensity Max', 'Intensity Range', 'Mean Intensity',
               'Median Intensity', 'Standard Deviation', 'Variance', 'Skewness', 'Kurtosis']
    # adding column titles
    images_df = pd.DataFrame(images_table, columns=columns)
    # converting to a dataframe using the data and defined column names
    return images_df # returning dataframe

for i in folder_list: # looping through the paths
    dataframe_overview = image_statistics_overview(i)
    # applying image_statistics_overview function to each path
    output_file = os.path.join(os.path.dirname(i),
                               f'{i}_image_statistics.csv')
    # defining the name of the output CSV
    dataframe_overview.to_csv(output_file, sep=',', index=False, encoding='utf-8') # saving CSV in same folder as image folders

#FUNCTION - EXTRACTING PIXEL DATA
def extract_pixel_data(folder_path):
    """
    Input: path to folder containing image files
    Returns: dataframe with pixel data of each image in the folder in 1D
    """
    pattern = os.path.join(folder_path, '*.png')  # pattern for identifying png files in folder
    image_files = glob.glob(pattern)  # retrieving png files from folder
    pixel_data = [] # initialising empty array
    for i in image_files:
        image = imread(i, as_gray=True)  # reading the image
        pixels = image.flatten()  # flatting the pixel data to 1D array
        pixel_data.append(pixels) # appending the data to the initialised array
    pixel_dataframe = pd.DataFrame(pixel_data).transpose() # converting to a dataframe
    print("pixel data saved to dataframe") # since this codes take a while to run, confirmation after it has been executed
    return pixel_dataframe # returning the dataframe

#creating 1d flattened pixel data dataframes for all image sequences (note: this takes a while)
dataframe_22_11_2021 = extract_pixel_data(folder_path_22_11_2021)
dataframe_04_12_2021 = extract_pixel_data(folder_path_04_12_2021)
dataframe_24_12_2021 = extract_pixel_data(folder_path_24_12_2021)
dataframe_23_10_2023_01 = extract_pixel_data(folder_path_23_10_2023_01)
dataframe_23_10_2023_02 = extract_pixel_data(folder_path_23_10_2023_02)
dataframe_23_10_2023_03 = extract_pixel_data(folder_path_23_10_2023_03)

#FUNCTION - BOX PLOTS
path_for_saving_box_plots = "C:/Users/Lenovo/Desktop/image_material/05_box_plots"
def plot_boxplot(pixel_dataframe, file_name, save_to_path):
    """
    Input: dataframe containing the pixel intensities
    Returns: boxplot of image data displayed and saved
    """
    plt.boxplot(pixel_dataframe) # creating box plot from dataframe
    plt.xlabel('Image Numbers') # depending on quantity of images the image numbers
    plt.ylabel('Pixel Intensity') # pixel intensities
    plt.savefig(f'{save_to_path}/box_plot_{file_name}.png', format='png', dpi=300) # saving the figure
    plt.show() # displaying the figure

#displaying and saving the box plots for all selected image sequences
plot_boxplot(dataframe_22_11_2021, "22_11_2021", path_for_saving_box_plots)
plot_boxplot(dataframe_04_12_2021, "04_12_2021", path_for_saving_box_plots)
plot_boxplot(dataframe_24_12_2021, "24_12_2021", path_for_saving_box_plots)
plot_boxplot(dataframe_23_10_2023_01, "23_10_2023_01", path_for_saving_box_plots)
plot_boxplot(dataframe_23_10_2023_02, "23_10_2023_02", path_for_saving_box_plots)
plot_boxplot(dataframe_23_10_2023_03, "23_10_2023_03", path_for_saving_box_plots)

#FUNCTION - PIXEL INTENSITY DISTRIBUTION HISTOGRAM OVERVIEW
path_for_saving_histograms = "C:/Users/Lenovo/Desktop/image_material/06_distribution_histogram_overview"
def display_image_histograms(pixel_dataframes, file_name, save_to_path, bin_size=10):
    """
    Input: dataframe(s) containing the pixel intensities
    Returns: distribution histogram of pixel intensities per image in overview displayed and saved
    """
    num_images = pixel_dataframes.shape[1] # determining the number of images
    num_cols = min(num_images, 3) # based on number of images max. 3 columns
    num_rows = (num_images + num_cols - 1) // num_cols # based on columns and number of images resulting rows
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols * 3.5, num_rows * 3.5))
    # defining the fig size based on image quantity
    axes = axes.flatten() # makes accessing the axes easier
    for i in range(num_images): # looping through the number of images
        image_pixel_data = pixel_dataframes.iloc[:, i]
        # looping through image pixel data based on image index
        ax = axes[i] # defining axes based on image index
        ax.hist(image_pixel_data, bins=bin_size, color="black")
        # creating histogram
        ax.set_title(f'Image {i + 1}')
        # title based on image index in sequence starting from 1
        ax.set_xlabel('Pixel Intensities')
        ax.set_ylabel('Frequency')
    for j in range(num_images, len(axes)): # looping through image locations
        axes[j].axis('off') # removing unnecessary space on grid
    plt.tight_layout() # layout option for better visualisation
    plt.savefig(f'{save_to_path}/histogram_{file_name}.png', format='png', dpi=300)
    # saving the figure
    plt.show() # displaying figure in console

#displaying and saving the histograms for all selected image sequences
display_image_histograms(dataframe_22_11_2021, "22_11_2021", path_for_saving_histograms, 100)
display_image_histograms(dataframe_04_12_2021, "04_12_2021", path_for_saving_histograms, 100)
display_image_histograms(dataframe_24_12_2021, "24_12_2021", path_for_saving_histograms, 100)
display_image_histograms(dataframe_23_10_2023_01, "23_10_2023_01", path_for_saving_histograms, 100)
display_image_histograms(dataframe_23_10_2023_02, "23_10_2023_02", path_for_saving_histograms, 100)
display_image_histograms(dataframe_23_10_2023_03, "23_10_2023_03", path_for_saving_histograms, 100)

#FUNCTION - PIXEL INTENSITIY DISTRIBUTION HISTOGRAM FOR EACH IMAGE
path_for_saving_histogram = "C:/Users/Lenovo/Desktop/image_material/07_distribution_histogram_individual"
def display_image_histogram(pixel_dataframes, file_name, save_to_path, bin_size=10):
    """
    Input: dataframe(s) containing the pixel intensities for the images
    Returns: distribution histogram of pixel intensities per image displayed and saved
    """
    for i in range(pixel_dataframes.shape[1]): # looping through the number of images based on second dimension layer
        plt.hist(pixel_dataframes[i], bins=bin_size, color="black")  # creating the histogram
        plt.xlabel('Pixel Intensities')  # label for x axis
        plt.ylabel('Frequency')  # label for y axis
        plt.savefig(f'{save_to_path}/histogram_{file_name}_{i}.png', format='png', dpi=300) # saving the plot
        plt.show()  # displaying the plot

# applying display_image_histogram to all image sequences
display_image_histogram(dataframe_22_11_2021, "22_11_2021", path_for_saving_histogram, 100)
display_image_histogram(dataframe_04_12_2021, "04_12_2021", path_for_saving_histogram, 100)
display_image_histogram(dataframe_24_12_2021, "24_12_2021", path_for_saving_histogram, 100)
display_image_histogram(dataframe_23_10_2023_01, "23_10_2023_01", path_for_saving_histogram, 100)
display_image_histogram(dataframe_23_10_2023_02, "23_10_2023_02", path_for_saving_histogram, 100)
display_image_histogram(dataframe_23_10_2023_03, "23_10_2023_03", path_for_saving_histogram, 100)

#FUNCTION - CUMULATIVE DISTRIBUTION FUNCTION OVERVIEW
path_for_saving_cdf_overview = "C:/Users/Lenovo/Desktop/image_material/08_cdf_overview"
def display_image_cdf_overview(pixel_dataframes, file_name, save_to_path, bin_size=10):
    """
    Input: dataframe(s) containing the pixel intensities
    Returns: cumulative distribution function of cumulative pixel intensities per image in overview displayed and saved
    """
    num_images = pixel_dataframes.shape[1]  # determining the number of images
    num_cols = min(num_images, 3)  # based on number of images max. 3 columns
    num_rows = (num_images + num_cols - 1) // num_cols  # based on columns and number of images resulting rows
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols * 3.5, num_rows * 3.5))
    # creating fig size based on image quantity in sequence
    axes = axes.flatten() # makes accessing axes easier in loop that follows
    for i in range(num_images): # looping through each image
        image_pixel_data = pixel_dataframes.iloc[:, i].to_numpy()  # skimages needs numpy format...
        ax = axes[i] # accessing the respective axes for each image
        cdf, bins = exposure.cumulative_distribution(image_pixel_data, nbins=bin_size) # creating CDF
        ax.plot(bins, cdf, color='black') # creating cumulative distribution function
        ax.set_title(f'Image {i + 1}') # image number starting from 1 based on index
        ax.set_xlabel('Pixel Intensities')
        ax.set_ylabel('Cumulative Frequency')
    for j in range(num_images, len(axes)): # looping through each image axes
        axes[j].axis('off') # removing unnecessary space on grid
    plt.tight_layout() # layout option
    plt.savefig(f'{save_to_path}/cdf_{file_name}_{i}.png', format='png', dpi=300)
    # saving the image
    plt.show() # plotting image in console

# applying display_image_cdf_overview to all image sequences
display_image_cdf_overview(dataframe_22_11_2021, "22_11_2021", path_for_saving_cdf_overview, 100)
display_image_cdf_overview(dataframe_04_12_2021, "04_12_2021", path_for_saving_cdf_overview, 100)
display_image_cdf_overview(dataframe_24_12_2021, "24_12_2021", path_for_saving_cdf_overview, 100)
display_image_cdf_overview(dataframe_23_10_2023_01, "23_10_2023_01", path_for_saving_cdf_overview, 100)
display_image_cdf_overview(dataframe_23_10_2023_02, "23_10_2023_02", path_for_saving_cdf_overview, 100)
display_image_cdf_overview(dataframe_23_10_2023_03, "23_10_2023_03", path_for_saving_cdf_overview, 100)

#FUNCTION - CUMULATIVE DISTRIBUTION FUNCTION FOR EACH IMAGE
path_for_saving_cdf = "C:/Users/Lenovo/Desktop/image_material/09_cdf_individual"
def display_image_cdf(pixel_dataframes, file_name, save_to_path, bin_size=10):
    """
    Input: dataframe(s) containing the pixel intensities for the images
    Returns: cumulative distribution function of cumulative pixel intensities per image displayed and saved
    """
    for i in range(pixel_dataframes.shape[1]): # looping through number of images in sequence based on dimension layer
        image_pixel_data = pixel_dataframes.iloc[:, i].to_numpy()  # skimages needs numpy format
        cdf, bins = exposure.cumulative_distribution(image_pixel_data, nbins=bin_size) # creating CDF
        plt.plot(bins, cdf, color='black')  # creating cumulative distribution function plot
        plt.xlabel('Pixel Intensities')
        plt.ylabel('Cumulative Frequency')
        plt.savefig(f'{save_to_path}/cdf_{file_name}_{i}.png', format='png', dpi=300)
        # saving the image
        plt.show() # plotting the image in the console

# applying display_image_cdf to all image sequences
display_image_cdf(dataframe_22_11_2021, "22_11_2021", path_for_saving_cdf, 100)
display_image_cdf(dataframe_04_12_2021, "04_12_2021", path_for_saving_cdf, 100)
display_image_cdf(dataframe_24_12_2021, "24_12_2021", path_for_saving_cdf, 100)
display_image_cdf(dataframe_23_10_2023_01, "23_10_2023_01", path_for_saving_cdf, 100)
display_image_cdf(dataframe_23_10_2023_02, "23_10_2023_02", path_for_saving_cdf, 100)
display_image_cdf(dataframe_23_10_2023_03, "23_10_2023_03", path_for_saving_cdf, 100)

#FUNCTION - HISTO WITH CDF ON TOP FOR INDIVIDUAL IMAGES
path_for_saving_hist_cdf = "C:/Users/Lenovo/Desktop/image_material/33_image_hist_with_cdf"
def display_hist_cdf(pixel_dataframes, file_name, save_to_path, bin_size=10):
    """
    Input: dataframe(s) containing the pixel intensities for the images
    Returns: distribution histogram and cumulative distribution function of pixel intensities per image displayed and saved
    """
    for i in range(pixel_dataframes.shape[1]): # looping through the image quantity in sequence based on dimension layer
        image_pixel_data = pixel_dataframes.iloc[:, i].to_numpy()  # skimages needs numpy format
        plt.hist(image_pixel_data, bins=bin_size, color="black") # creating histogram
        plt.xlabel('Pixel Intensities')
        plt.ylabel('Frequency')
        cdf, bins = exposure.cumulative_distribution(image_pixel_data, nbins=bin_size) # obtaining CDF
        plt.twinx() # introducing separate axis
        plt.plot(bins, cdf, color='red') # plotting CDF
        plt.ylabel('Cumulative Frequency', color='red') # y axis for CDF in red
        plt.xlim(0,1) # x axis limits to 0 - 1
        plt.ylim(0, 1) # y axis limits to 0 - 1
        plt.savefig(f'{save_to_path}/histogram_cdf_{file_name}_{i}.png', format='png', dpi=300)
        # saving the image
        plt.show() # plotting image to console

# applying display_hist_cdf to all image sequences with 100 bins
display_hist_cdf(dataframe_22_11_2021, "22_11_2021", path_for_saving_hist_cdf, 100)
display_hist_cdf(dataframe_04_12_2021, "04_12_2021", path_for_saving_hist_cdf, 100)
display_hist_cdf(dataframe_24_12_2021, "24_12_2021", path_for_saving_hist_cdf, 100)
display_hist_cdf(dataframe_23_10_2023_01, "23_10_2023_01", path_for_saving_hist_cdf, 100)
display_hist_cdf(dataframe_23_10_2023_02, "23_10_2023_02", path_for_saving_hist_cdf, 100)
display_hist_cdf(dataframe_23_10_2023_03, "23_10_2023_03", path_for_saving_hist_cdf, 100)

#FUNCTION - EXTRACT 2D PIXELS (Fourier transform function needs 2D data to work)
def extract_pixel_data_2d(folder_path):
    """
    Input: path to folder containing image files
    Returns: nested list with pixel data of each image in the folder in 2D
    """
    pattern = os.path.join(folder_path, '*.png')  # pattern for identifying png files in folder
    image_files = glob.glob(pattern)  # retrieving png files from folder
    pixel_data = [] # initialising empty array
    for i in image_files: # looping through the identified png file paths
        image = imread(i, as_gray=True)  # reading the image
        pixel_data.append(image) # appending the 2D data to the initialised array
    print("pixel data saved") # since this codes take a while to run, confirmation after it has been executed
    return pixel_data

#creating nested lists for all image sequences in 2d
image_data_22_11_2021_2d = extract_pixel_data_2d(folder_path_22_11_2021)
image_data_04_12_2021_2d = extract_pixel_data_2d(folder_path_04_12_2021)
image_data_24_12_2021_2d = extract_pixel_data_2d(folder_path_24_12_2021)
image_data_23_10_2023_01_2d = extract_pixel_data_2d(folder_path_23_10_2023_01)
image_data_23_10_2023_02_2d = extract_pixel_data_2d(folder_path_23_10_2023_02)
image_data_23_10_2023_03_2d = extract_pixel_data_2d(folder_path_23_10_2023_03)

#FUNCTION - FOURIER TRANSFORM FOR EACH IMAGE
path_for_saving_fourier_transform = "C:/Users/Lenovo/Desktop/image_material/10_fourier_transform"
def image_fourier_transform(image_data_2d, file_name, save_to_path):
    """
    Input: (list of) image data in 2d
    Returns: Plot of fourier transform visualised next to original image and saved
    """
    for i in range(len(image_data_2d)): # looping through 2d image data
        image = image_data_2d[i] # accessing each image data in sequence
        f_transform = np.fft.fft2(image) # obtaining fourier transform using package
        f_transform_shifted = np.fft.fftshift(f_transform)
        # shift zero frequency component to center (more intuitive to analyse and makes it same size as image)
        magnitude_spectrum = np.abs(f_transform_shifted) # calculate magnitude spectrum
        magnitude_spectrum_log = np.log(1 + magnitude_spectrum)  # log scale for enhanced visualization
        fig, ax = plt.subplots(1, 2, figsize=(12, 6)) # creating sub plots or original image and fourier transform
        ax[0].imshow(image, cmap = 'gray') # original image in grayscale
        ax[0].set_title('Original Image') # title of original image
        ax[0].axis('off') # removing x and y axis
        ax[1].imshow(magnitude_spectrum_log, cmap = 'gray') # adding the fourier transform visualisation
        ax[1].set_title('Fourier Transform') # title
        ax[1].axis('off') # removing x and y axis
        plt.savefig(f'{save_to_path}/fourier_transform_{file_name}_{i}.png', format='png', dpi=300)
        # saving the resulting plot
        plt.show() # plotting to console

#applying image_fourier_transform to each sequence
image_fourier_transform(image_data_22_11_2021_2d, "22_11_2021", path_for_saving_fourier_transform)
image_fourier_transform(image_data_04_12_2021_2d, "04_12_2021", path_for_saving_fourier_transform)
image_fourier_transform(image_data_24_12_2021_2d, "24_12_2021", path_for_saving_fourier_transform)
image_fourier_transform(image_data_23_10_2023_01_2d, "23_10_2023_01", path_for_saving_fourier_transform)
image_fourier_transform(image_data_23_10_2023_02_2d, "23_10_2023_02", path_for_saving_fourier_transform)
image_fourier_transform(image_data_23_10_2023_03_2d, "23_10_2023_03", path_for_saving_fourier_transform)

#FUNCTION - HIGH PASS FILTER
path_for_saving_high_pass = "C:/Users/Lenovo/Desktop/image_material/11_high_pass"
def image_high_pass_filter(image_data_2d, file_name, save_to_path, cut_off_radius = 30):
    """
    Input: (list of) image data in 2d
    Returns: Plot of original image, high-pass filter mask and high-pass filter result displayed and saved
    """
    for i in range(len(image_data_2d)): # looping through image 2D data for sequence
        f_transform = np.fft.fft2(image_data_2d[i]) # fourier transform
        f_transform_shifted = np.fft.fftshift(f_transform) # shifting fourier transform to center
        rows, cols = image_data_2d[i].shape # rows and colums based on image shape
        crow, ccol = rows // 2, cols // 2 # shape for mask
        radius = cut_off_radius # radius for cutting off low frequency in frequency domain
        mask = np.ones((rows, cols), dtype=np.float32) # initialising an empty mask
        center = [crow, ccol] # defining the center of the mask based on shape
        y, x = np.ogrid[:rows, :cols] # creating a grid based on dimensions
        distance = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2) # calculating the distance to the center
        mask[distance <= radius] = 0  # all frequencies below or equal to cut off are set to zero
        f_transform_filtered = f_transform_shifted * mask # applying mask
        f_transform_filtered_shifted = np.fft.ifftshift(f_transform_filtered)  # shift back from center
        filtered_image = np.fft.ifft2(f_transform_filtered_shifted) # inverse fourier transform
        filtered_image = np.abs(filtered_image)  # absolute values
        plt.figure(figsize=(12, 6)) # initialising plot
        # original image
        plt.subplot(1, 4, 1) # location of first plot
        plt.title('Original Image') # original image
        plt.imshow(image_data_2d[i], cmap='gray')
        # accessing index of 2d image data and plotting grayscale
        plt.axis('off') # removing x and y axis
        # high-pass filter mask
        plt.subplot(1, 4, 2) # location of first plot
        plt.title('High-Pass Filter Mask') # visualisation of the cut-off radius
        plt.imshow(mask, cmap='gray') # plotting mask in grayscale
        plt.axis('off') # removing x and y axis
        # filtered image
        plt.subplot(1, 4, 3) # location of plot
        plt.title('High-Pass Filtered Image') # final result of high-pass filter
        plt.imshow(filtered_image, cmap='gray') # displaying the high-pass filtered image
        plt.axis('off') # removing axis
        # filtered image log transformed
        plt.subplot(1, 4, 4) # location of image
        plt.title('Square Root Image')  # square root transformed result for enhanced visualisation
        plt.imshow(np.sqrt(filtered_image), cmap='gray') # computing square root directly and display in grayscale
        plt.axis('off') # removing axis
        plt.tight_layout() # layout option
        plt.savefig(f'{save_to_path}/high_pass_{file_name}_{i}.png', format='png', dpi=300)
        # saving the resulting plot
        plt.show() # plotting the visualisation to console

#displaying and saving high pass filter (radius 40 based on trial and error)
image_high_pass_filter(image_data_22_11_2021_2d, "22_11_2021", path_for_saving_high_pass, 40)
image_high_pass_filter(image_data_04_12_2021_2d, "04_12_2021", path_for_saving_high_pass, 40)
image_high_pass_filter(image_data_24_12_2021_2d, "24_12_2021", path_for_saving_high_pass, 40)
image_high_pass_filter(image_data_23_10_2023_01_2d, "23_10_2023_01", path_for_saving_high_pass, 40)
image_high_pass_filter(image_data_23_10_2023_02_2d, "23_10_2023_02", path_for_saving_high_pass, 40)
image_high_pass_filter(image_data_23_10_2023_03_2d, "23_10_2023_03", path_for_saving_high_pass, 40)

#FUNCTION - HIGH-PASS FILTER INDIVIDUAL IMAGES
path_for_saving_high_pass_images = "C:/Users/Lenovo/Desktop/image_material/34_square_root_images_bandpass_filter"
def image_band_pass_filter(image_data_2d, file_name, save_to_path, lower_cutoff_radius=10, upper_cutoff_radius=None):
    """
    Input: (list of) image data in 2d, cutoff radius lower and higher limit (optional)
    Returns: Band-pass or High-pass filter result displayed and saved (square root to brighten image applied)
    """
    for i in range(len(image_data_2d)): # looping through each image data from sequence
        f_transform = np.fft.fft2(image_data_2d[i])  # fourier transform
        f_transform_shifted = np.fft.fftshift(f_transform)  # shift Fourier transform to center
        rows, cols = image_data_2d[i].shape # image dimensions
        crow, ccol = rows // 2, cols // 2  # center coordinates for mask
        mask = np.ones((rows, cols), dtype=np.float32)  # initialize the mask
        center = [crow, ccol]  # center of the mask
        y, x = np.ogrid[:rows, :cols]  # create grid based on dimensions
        distance = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)  # calculate distance from center
        # band-pass filter mask:  frequencies within a specified range
        if upper_cutoff_radius is not None: # if an upper limit is defied in the function
            mask[(distance <= upper_cutoff_radius) & (distance >= lower_cutoff_radius)] = 1
            # if the upper limit radius is greater or equal to the distance to center and
            # the lower cut-off radius is smaller or equal to the distance of the center
            # the mask is activated
            mask[(distance > upper_cutoff_radius) | (distance < lower_cutoff_radius)] = 0
            # but if the upper limit radius is greater than the distance to the center OR
            # the lower cut-off radius is bigger than the distance to the center it is not a match for mask
        else: # case for no upper limit defined
            mask[distance >= lower_cutoff_radius] = 1
            mask[distance < lower_cutoff_radius] = 0
        f_transform_filtered = f_transform_shifted * mask  # apply mask
        f_transform_filtered_shifted = np.fft.ifftshift(f_transform_filtered)  # shift back
        filtered_image = np.fft.ifft2(f_transform_filtered_shifted)  # inverse Fourier transform
        filtered_image = np.abs(filtered_image)  # absolute values
        plt.imshow(np.sqrt(filtered_image), cmap='gray') # square root for enhanced visualisation
        plt.title("Cut-off radius: " + str(lower_cutoff_radius) + " px") # title based on cut-off radius
        plt.axis('off') # deactived axis
        plt.tight_layout() # layout option
        plt.savefig(f'{save_to_path}/band_pass_image_{lower_cutoff_radius}_{file_name}_{i}.png', format='png', dpi=300)
        # saving the image
        plt.show() # dispaying plot in console

#applying image_band_pass_filter with a high-pass filter (no upper limit) to each image sequence
image_band_pass_filter(image_data_22_11_2021_2d, "22_11_2021", path_for_saving_high_pass_images, 40)
image_band_pass_filter(image_data_04_12_2021_2d, "04_12_2021", path_for_saving_high_pass_images, 40)
image_band_pass_filter(image_data_24_12_2021_2d, "24_12_2021", path_for_saving_high_pass_images, 40)
image_band_pass_filter(image_data_23_10_2023_01_2d, "23_10_2023_01", path_for_saving_high_pass_images, 40)
image_band_pass_filter(image_data_23_10_2023_02_2d, "23_10_2023_02", path_for_saving_high_pass_images, 40)
image_band_pass_filter(image_data_23_10_2023_03_2d, "23_10_2023_03", path_for_saving_high_pass_images, 40)

#FUNCTION - TRESHOLD PLOTS
path_for_saving_threshold = "C:/Users/Lenovo/Desktop/image_material/12_threshold"
def threshold_image(image_data_2d, thresholds, gray_scales, file_name, save_to_path):
    """
    Input: List of 2D image data, threshold values, grayscale values, file name, and path to save the results in new folder
    Returns: Plots of original and segmented images for each image and saves them to the specified folder
    """
    for image_index in range(len(image_data_2d)): # looping through 2d image data sequences
        mapped_image = np.zeros_like(image_data_2d[image_index]) # creating a blank image for mapping
        for i, threshold in enumerate(thresholds): # applying thresholds to map grayscale values
            if i == 0: # values less than or equal to threshold in list
                mask = image_data_2d[image_index] <= threshold # boolean mask
            else:
                mask = (image_data_2d[image_index] > thresholds[i - 1]) & (image_data_2d[image_index] <= threshold)
                # mask for pixels in specified range
            mapped_image[mask] = gray_scales[i] # maps the mask to input grayscales
        mapped_image[image_data_2d[image_index] > thresholds[-1]] = gray_scales[-1] # last range remaining
        plt.figure(figsize=(12, 6)) # creating figure
        # Original image
        plt.subplot(1, 2, 1) # location of sub-plot
        plt.title('Original Image') # title
        plt.imshow(image_data_2d[image_index], cmap='gray') # accessing 2d data
        plt.axis('off') # removing axis
        # Threshold image
        plt.subplot(1, 2, 2) # adding second plot
        plt.title(f'Clustered Image - thresholds: {thresholds}') # title with input thresholds
        plt.imshow(mapped_image, cmap='gray') # result of mapping in grayscale
        plt.axis('off') # removed axis
        plt.tight_layout() # layout option
        plt.savefig(f'{save_to_path}/threshold_{file_name}_{image_index}.png', format='png', dpi=300)
        # saving the plot
        plt.show() # displaying in console

#input parameters for function
thresholds = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8] # based on histograms
gray_values = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.] # has to be one more because it "fills the spaces in between thresholds"

#applying threshold_image function to each image sequence
threshold_image(image_data_22_11_2021_2d, thresholds, gray_values, "22_11_2021", path_for_saving_threshold)
threshold_image(image_data_04_12_2021_2d, thresholds, gray_values, "04_12_2021", path_for_saving_threshold)
threshold_image(image_data_24_12_2021_2d, thresholds, gray_values, "24_12_2021", path_for_saving_threshold)
threshold_image(image_data_23_10_2023_01_2d, thresholds, gray_values, "23_10_2023_01", path_for_saving_threshold)
threshold_image(image_data_23_10_2023_02_2d, thresholds, gray_values, "23_10_2023_02", path_for_saving_threshold)
threshold_image(image_data_23_10_2023_03_2d, thresholds, gray_values, "23_10_2023_03", path_for_saving_threshold)

#FUNCTION - THRESHOLD PLOTS INDIVIDUAL
path_for_saving_threshold_individual = "C:/Users/Lenovo/Desktop/image_material/13_threshold_individual"
def threshold_image_individual(image_data_2d, thresholds, gray_scales, file_name, save_to_path):
    """
    Input: List of 2D image data, threshold values, grayscale values, file name, and path to save the results in new folder
    Returns: Plots of segmented images displayed and saved to the specified folder
    """
    for image_index in range(len(image_data_2d)): # looping through each image data in sequence
        mapped_image = np.zeros_like(image_data_2d[image_index]) # Create a blank image for mapping
        for i, threshold in enumerate(thresholds): # Apply thresholds to map grayscale values
            if i == 0: # first threshold
                mask = image_data_2d[image_index] <= threshold # booleans for all that is below first threshold
            else:
                mask = (image_data_2d[image_index] > thresholds[i - 1]) & (image_data_2d[image_index] <= threshold)
                # all the other thresholds
            mapped_image[mask] = gray_scales[i] # mapping the thresholds to grayscale values defined
        mapped_image[image_data_2d[image_index] > thresholds[-1]] = gray_scales[-1] # last range remaining
        #plt.title(f'Clustered Image - thresholds: {thresholds}') # could active title if needed
        plt.imshow(mapped_image, cmap='gray') # plot the image
        plt.axis('off') # removing axis
        plt.tight_layout() # layout option
        plt.savefig(f'{save_to_path}/threshold_{file_name}_{image_index}.png', format='png', dpi=300)
        # saving the figure
        plt.show() # displaying the plot in console

#input parameters for function
thresholds = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8] # based on histograms
gray_values = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.] # has to be one more because it "fills the spaces in between thresholds"

#applying threshold_image_individual function to each image sequence
threshold_image_individual(image_data_22_11_2021_2d, thresholds, gray_values, "22_11_2021", path_for_saving_threshold_individual)
threshold_image_individual(image_data_04_12_2021_2d, thresholds, gray_values, "04_12_2021", path_for_saving_threshold_individual)
threshold_image_individual(image_data_24_12_2021_2d, thresholds, gray_values, "24_12_2021", path_for_saving_threshold_individual)
threshold_image_individual(image_data_23_10_2023_01_2d, thresholds, gray_values, "23_10_2023_01", path_for_saving_threshold_individual)
threshold_image_individual(image_data_23_10_2023_02_2d, thresholds, gray_values, "23_10_2023_02", path_for_saving_threshold_individual)
threshold_image_individual(image_data_23_10_2023_03_2d, thresholds, gray_values, "23_10_2023_03", path_for_saving_threshold_individual)
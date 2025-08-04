#FILE SUMMARY:
### FITS header and data information stats summary
### displaying raw pojected image data
### defining and highlighting a cropping range for images
### cropping the images
### preparing a topography map
### converting cropped areas from pixel to degrees
### displaying cropped areas on topography map

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
from skimage.exposure import cumulative_distribution

#FUNCTION - RETURNING OVERVIEW TABLE
def fits_overview_table(file_path):
    """
    Input: path to FITS file
    Returns: dataframe with header information and data statistics
    (min, max, std, mean, median, skew, curt, variation, range, amount of NaNs) for each image in the data unit
    """
    with fits.open(file_path) as hdul:
        header_unit = hdul[0].header # open header
        data = hdul[0].data # open data
        extracted_data = {
            'Index': list(range(len([header_unit[key] for key in header_unit if str(key).startswith("FILTER")]))),
            # extracting index based on the quantity of images (determined based on number of words starting with "FILTER")
            'Basis': [header_unit[key] for key in header_unit if str(key).startswith("BASEPR")],
            # base product name (basis for image)
            'Filter': [header_unit[key] for key in header_unit if str(key).startswith("FILTER")],
            # filter type of image
            'Time': [header_unit[key] for key in header_unit if str(key).startswith("DATE-O")][1:]
        } # date and time of image
        header_dataframe = pd.DataFrame(extracted_data) # converting to dataframe
        if data is None:
            raise ValueError("no data") # usually there is data though but just in case something goes wrong
        stats_list = [] # creates empty list to store data stats
        for index, image in enumerate(data): # looping through FITS data index to extract image data based on index
            flat_image = image.flatten() # flattening pixel intensities/ dimensions in image
            min_val = np.nanmin(flat_image) # determining the minimum intensity while ignoring NANs
            max_val = np.nanmax(flat_image) # determining the maximum intensity while ignoring NANs
            std_val = np.nanstd(flat_image) # computing the standard deviation
            variation_val = np.nanvar(flat_image) # variance
            range_val = max_val - min_val # pixel intensity range
            mean = np.nanmean(flat_image) # pixel intensity mean
            median = np.nanmedian(flat_image) # pixel intensity median
            skewness = pd.Series(flat_image).skew(), # skewness of intensities
            curtosis = pd.Series(flat_image).kurtosis() # kurtosis of intensities
            nan_count = np.isnan(flat_image).sum() # amount of NANs
            stats_list.append({
                'Index': index,
                'Min': min_val,
                'Max': max_val,
                'Std Dev': std_val,
                'Variation': variation_val,
                'Range': range_val,
                'Mean': mean,
                'Median': median,
                'Kurtosis': curtosis,
                'Skewness': skewness,
                'NaN Count': nan_count
            }) #creating dictionary from the previously computed stats
        stats_dataframe = pd.DataFrame(stats_list) # converting to dataframe
        combined_dataframe = pd.merge(header_dataframe, stats_dataframe, on='Index', how='left')
        # combining header and data unit info (since image quantity is the same, this works seamlessly)
    return combined_dataframe # returning the data

#LOOP - SAVING OVERVIEW TABLE FOR ALL FITS FILES
folder_path = "C:/Users/Lenovo/Desktop/fits_files" # folder path with targeted FITS files
pattern = os.path.join(folder_path, '*.fits') # pattern to identify FITS files in folder
fits_files = glob.glob(pattern) # retrieving all paths of FITS files in folder

for file_path in fits_files: # looping through FITS paths
    dataframe_overview = fits_overview_table(file_path) # applying fits_overview_table function to each file_path
    name = re.search(r'(\d{8}T\d{6}_\d{4})', os.path.basename(file_path))[0] # extracting specific part of the file_path for new file_name
    output_file = os.path.join(os.path.dirname(file_path), f'{name}_primary_hdu.csv') # creating output file path and new file name in same folder as original file
    dataframe_overview.to_csv(output_file, sep=',', index=False, encoding='utf-8') # save dataframe as CSV
    print(dataframe_overview['Filter']) # print filter types of images as confirmation

#FUNCTION - IMAGE PLOTS FROM FITS FILE
def fits_image_plot(file_path, filter_type, save_to_path):
    """
    Input: path to FITS file, desired filter type ('F320') and path for saving the output images
    Returns: image plots of all images in the respective file of the respective filter type are displayed and saved
    (assumes that the image data is located in the primary HDU)
    """
    open_fits = fits.open(file_path)  # opening FITS file
    header_unit = open_fits[0].header # variable for header information
    filter_types = [header_unit[key] for key in header_unit if str(key).startswith("FILTER")]
    # list of the filter types of contained images
    desired_indices = [index for index, value in enumerate(filter_types) if value == filter_type]
    # returns index of desired images based on defined filter type
    image_data = open_fits[0].data # variable for image data
    image_dates = [header_unit[key] for key in header_unit if str(key).startswith("DATE-O")][1:]
    # names for images based on date and time
    for i in desired_indices: # looping through the relevant image indices
        plt.imshow(image_data[i], cmap='gray', origin='lower')
        # plotting the image in grayscale and origin in bottom left corner
        #plt.title(image_dates[i]) # title = date and time of image # can be activated if needed
        #plt.axis('off')  # can be commented out for displaying the images without axis
        plt.xticks(np.linspace(0, image_data[i].shape[1] - 1, 10), [f'{int(tick)}°' for tick in np.linspace(-180, 180, 10)])
        # adding ° ticks to longitude
        plt.yticks(np.linspace(0, image_data[i].shape[0] - 1, 5), [f'{int(tick)}°' for tick in np.linspace(-90, 90, 5)])
        # adding ° ticks latitude
        plt.grid(True) # grid for better visualisation
        plt.savefig(f'{save_to_path}/raw_image_{re.sub(r'[<>:"/\\|?*]', '_', image_dates[i])}.png', format='png', dpi=300)
        # saving images (problematic symbols are replaced)
        plt.show() # displaying image

#LOOP - PLOTTING ALL F320 IMAGES FROM FITS FILES IN FOLDER AND SAVING THEM IN SPECIFIED NEW FOLDER
path_to_save_raw_images = 'C:/Users/Lenovo/Desktop/image_material/01_raw_images'
for file_path in fits_files: # looping through the fits paths
    fits_image_plot(file_path, 'F320',path_to_save_raw_images) # applying fits_image_plot to each FITS file path

#each image is analysed individually to define a suitable cropping area
print(len(fits_files)) # displays the quantity of files = 6
#fits_image_plot(fits_files[0], 'F320')
#fits_image_plot(fits_files[1], 'F320')
#fits_image_plot(fits_files[2], 'F320')
#fits_image_plot(fits_files[3], 'F320')
#fits_image_plot(fits_files[4], 'F320')
#fits_image_plot(fits_files[5], 'F320')

#HISTOGRAM & CDF OF ALL F320 RAW IMAGE DATA
def fits_image_data(file_path, filter_type):
    """
    Input: path to FITS file and desired filter type ('F320')
    Returns: image data of all images in the respective file of the respective filter type are extracted
    (assumes that the image data is located in the primary HDU)
    """
    open_fits = fits.open(file_path)
    # opening FITS file
    header_unit = open_fits[0].header
    # variable for header information
    filter_types = [header_unit[key] for key in header_unit if str(key).startswith("FILTER")]
    # list of the filter types of contained images
    desired_indices = [index for index, value in enumerate(filter_types) if value == filter_type]
    # returns index of desired images based on filter type
    image_data = open_fits[0].data # variable for image data
    for i in desired_indices: # looping through desired image data
        return image_data[i] # extracting previously defined indices for data

all_data = [] # initialising array for all image data
for i in fits_files: # looping through all the FITS files, extracting F320 data
    data = fits_image_data(i, 'F320') # defining filter type
    all_data.append(data) # appending extracted data to the array that was initialised

all_data_array = np.concatenate(all_data) # concatenating all data in numpy array
flattened_data = all_data_array.flatten() # flattening numpy array
#len(flattened_data) testing if it worked
nan_removed = flattened_data[~np.isnan(flattened_data)] # removing NAN values
cdf, bin_centers = cumulative_distribution(nan_removed) # creating CDF using non-Nan data

save_to_path_all_data = 'C:/Users/Lenovo/Desktop/image_material/01_raw_images'
# path for saving histogram and CDF of all concatenated data
#This is for an overview of ALL raw data selected (previously concatenated)
fig, ax1 = plt.subplots() # sub plots because CDF is plotted on top
ax1.hist(nan_removed, bins=100, color='black', edgecolor='black') # histogram with 100 bins
ax1.set_xlabel('Pixel Intensity')
ax1.set_ylabel('Frequency')
ax1.set_xlim(np.min(nan_removed), np.max(nan_removed))
# x axis limits based on the minimum and maximum pixel values
ax2 = ax1.twinx() # creates a new axis object for the second plot
ax2.plot(bin_centers, cdf, 'r') # plotting CDF in red
ax2.set_ylabel('Cumulative Probability', color = 'red') # label for CDF in red
ax2.set_ylim(0, 1) # y axis limits
ax2.tick_params(axis='y', colors='red') # red ticks on y axis for CDF
plt.savefig(f'{save_to_path_all_data}/all_data_overview_.png', format='png',
            dpi=300) # saving figure to specified path
plt.show() # displaying the plot in the console aswell

#FUNCTION - HIGHLIGHTING AREA FOR CROPPING
def fits_image_plot_with_highlight(file_path, filter_type, x_start, x_end, y_start, y_end, save_to_path):
    """
    Input: path to FITS file, desired filter type ('F320'), desired cropping range and path for saving the result
    Returns: image plots of all images in the respective file of the respective filter type
    with highlighted area based on cropping range displayed and saved
    (assumes that the image data is located in the primary HDU)
    """
    open_fits = fits.open(file_path)  # opening FITS file
    header_unit = open_fits[0].header # variable for header information
    filter_types = [header_unit[key] for key in header_unit if str(key).startswith("FILTER")] # list of the filter types of contained images
    desired_indices = [index for index, value in enumerate(filter_types) if value == filter_type] # returns index of desired images
    image_data = open_fits[0].data # variable for image data
    image_dates = [header_unit[key] for key in header_unit if str(key).startswith("DATE-O")][1:] # names for images
    for i in desired_indices: # looping through the relevant image indices
        plt.imshow(image_data[i], cmap='gray', origin='lower') # plotting the image in grayscale and origin in bottom left corner
        #plt.title(image_dates[i]) # title = date and time of image # turned off if image is saved
        #plt.axis('off') # can be commented out for displaying the images without axis
        plt.xticks(np.linspace(0, image_data[i].shape[1] - 1, 10), [f'{int(tick)}°' for tick in np.linspace(-180, 180, 10)])
        # adding ° ticks for longitude
        plt.yticks(np.linspace(0, image_data[i].shape[0] - 1, 5), [f'{int(tick)}°' for tick in np.linspace(-90, 90, 5)])
        # adding ° ticks for latitude
        plt.grid(True) # grid for better visualisation
        frame_highlight = patches.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start, linewidth=2,
                                            edgecolor='r', facecolor='none')  # highlighted area added in red
        plt.gca().add_patch(frame_highlight)  # initialising highlighted area
        plt.savefig(f'{save_to_path}/raw_image_highlighted_{re.sub(r'[<>:"/\\|?*]', '_', image_dates[i])}.png', format='png', dpi=300)
        # saving the image in the specified path
        plt.show() # displaying image

#displaying and saving cropping ranges for all image sequences:
path_to_save_highlighted_images = 'C:/Users/Lenovo/Desktop/image_material/02_raw_images_highlighted_area'
#22.11.2021
x_start, x_end = 500, 2000 # horizontal start and end point of the image
y_start, y_end = 1000, 2000 # vertical start and end point of the image
fits_image_plot_with_highlight(fits_files[0], 'F320', x_start, x_end, y_start, y_end, path_to_save_highlighted_images)
#04.12.2021
x_start, x_end = 200, 1500 # horizontal start and end point of the image
y_start, y_end = 1100, 1900 # vertical start and end point of the image
fits_image_plot_with_highlight(fits_files[1], 'F320', x_start, x_end, y_start, y_end, path_to_save_highlighted_images)
#24.12.2021
x_start, x_end = 400, 1900 # horizontal start and end point of the image
y_start, y_end = 900, 1900 # vertical start and end point of the image
fits_image_plot_with_highlight(fits_files[2], 'F320', x_start, x_end, y_start, y_end, path_to_save_highlighted_images)
#23.10.2023 - 05:40-06:40
x_start, x_end = 1200, 2500 # horizontal start and end point of the image
y_start, y_end = 1000, 2000 # vertical start and end point of the image
fits_image_plot_with_highlight(fits_files[3], 'F320', x_start, x_end, y_start, y_end, path_to_save_highlighted_images)
#23.10.2023 - 08:18-09:18
x_start, x_end = 900, 2000 # horizontal start and end point of the image
y_start, y_end = 1100, 2000 # vertical start and end point of the image
fits_image_plot_with_highlight(fits_files[4], 'F320', x_start, x_end, y_start, y_end, path_to_save_highlighted_images)
#23.10.2023 - 10:56-11:56
x_start, x_end = 500, 1300 # horizontal start and end point of the image
y_start, y_end = 1200, 2000 # vertical start and end point of the image
fits_image_plot_with_highlight(fits_files[5], 'F320', x_start, x_end, y_start, y_end, path_to_save_highlighted_images)

#FUNCTION - PLOTTING THE CROPPED IMAGES
def fits_image_plot_cropped(file_path, filter_type, x_start, x_end, y_start, y_end, save_to_path):
    """
    Input: path to FITS file, the desired filter type ('F320'), cropping range and path for saving the image
    Returns: cropped image plots of all images in the respective file of the respective filter type displayed and saved
    (assumes that the image data is located in the primary HDU)
    """
    open_fits = fits.open(file_path)  # opening FITS file
    header_unit = open_fits[0].header # variable for header information
    filter_types = [header_unit[key] for key in header_unit if str(key).startswith("FILTER")] # list of the filter types of contained images
    desired_indices = [index for index, value in enumerate(filter_types) if value == filter_type] # returns index of desired images
    image_data = open_fits[0].data # variable for image data
    image_dates = [header_unit[key] for key in header_unit if str(key).startswith("DATE-O")][1:] # names for images
    for i in desired_indices: # looping through desired image indices
        plt.imsave(f'{save_to_path}/cropped_image_{re.sub(r'[<>:"/\\|?*]', '_', image_dates[i])}.png', image_data[i][y_start:y_end,x_start:x_end], cmap='gray', origin='lower', format='png', dpi=300)
        # saving image with problematic symbols replaced
        plt.imshow(image_data[i][y_start:y_end,x_start:x_end], cmap='gray', origin='lower')
        # plotting the cropped image in grayscale and origin in bottom left corner
        plt.title(image_dates[i]) # title = date and time of image
        plt.show() # initialising image plot

#viewing and saving the cropped images of each selected FITS file
path_to_save_cropped_images = 'C:/Users/Lenovo/Desktop/image_material/03_cropped_images'
#22.11.2021
fits_image_plot_cropped(fits_files[0], 'F320', 500, 2000, 1000, 2000, path_to_save_cropped_images)
#04.12.2021
fits_image_plot_cropped(fits_files[1], 'F320', 200, 1500, 1100, 1900, path_to_save_cropped_images)
#24.12.2021
fits_image_plot_cropped(fits_files[2], 'F320', 400, 1900, 900, 1900, path_to_save_cropped_images)
#23.10.2023 - 05:40-06:40
fits_image_plot_cropped(fits_files[3], 'F320', 1200, 2500, 1000, 2000, path_to_save_cropped_images)
#23.10.2023 - 08:18-09:18
fits_image_plot_cropped(fits_files[4], 'F320', 900, 2000, 1100, 2000, path_to_save_cropped_images)
#23.10.2023 - 10:56-11:56
fits_image_plot_cropped(fits_files[5], 'F320', 500, 1300, 1200, 2000, path_to_save_cropped_images)

#CONVERSION OF PIXEL RANGE TO LAT/LONG RANGE
cropped_regions = [
    ['22.11.2021', (500, 2000, 1000, 2000)],
    ['04.12.2021', (200, 1500, 1100, 1900)],
    ['24.12.2021', (400, 1900, 900, 1900)],
    ['23.10.2023 - 05:40-06:40', (1200, 2500, 1000, 2000)],
    ['23.10.2023 - 08:18-09:18', (900, 2000, 1100, 2000)],
    ['23.10.2023 - 10:56-11:56', (500, 1300, 1200, 2000)]
] # summarizing the cropped regions in pixels in a nested list (x_start, x_end, y_start, y_end)

reference_fits_file = fits.open(fits_files[0])
#obtaining the original image dimensions from a reference image (all original images are of the same dimensions)
# the dimensions are found in the header information but could also be extracted differently
image_width = [reference_fits_file[0].header[key] for key in reference_fits_file[0].header if str(key).startswith("NAXIS1")][0]
image_height = [reference_fits_file[0].header[key] for key in reference_fits_file[0].header if str(key).startswith("NAXIS2")][0]
#obtaining the scaling factor (pixel to degree) from header unit
image_scaling_factor = [reference_fits_file[0].header[key] for key in reference_fits_file[0].header if str(key).startswith("CDELT1")][0]
# checking if scaling factor in data is correct: (360 - 0) / (5760 - 0) - the factor is the same for x and y

def pixel_range_to_coord(x_start, x_end, y_start, y_end, image_scaling_factor):
    """
    Input: x_start, x_end, y_start, y_end in pixels and scaling factor
    Returns: long_x_start_west, long_x_end_east, lat_y_start_south, lat_y_end_north in degrees
    """
    long_x_start_west = -180 + x_start * image_scaling_factor
    # starting from -180, x_start is converted using the image scaling factor
    long_x_end_east = -180 + x_end * image_scaling_factor
    # starting from -180, x_end is converted using the image scaling factor
    lat_y_start_south = -90 + y_start * image_scaling_factor
    # starting from -90, y_start is converted using the image scaling factor
    lat_y_end_north = -90 + y_end * image_scaling_factor
    # starting from -90, y_end is converted using the image scaling factor
    return (long_x_start_west, long_x_end_east, lat_y_start_south, lat_y_end_north) #returning converted data

#LOOP - CONVERSION OF PIXELS TO DEGREES
for i in range(len(cropped_regions)): # looping through the summary of cropped regions
    range_in_degrees = pixel_range_to_coord(cropped_regions[i][1][0],cropped_regions[i][1][1],cropped_regions[i][1][2],cropped_regions[i][1][3],image_scaling_factor)
    # i is the cropped region index defined in the summary
    # [][] specifies the component (e.g. x_start) within the entry in the summary
    cropped_regions[i].append(range_in_degrees)
    # appending results to my nested list that previously contains the cropping ranges only
print(cropped_regions) # checking if correctly appended

#DISPLAYING CROPPED REGIONS ON A TOPOGRAPHY MAP
topography_path = "C:/Users/Lenovo/Desktop/image_material/mola32.nc" # path to topography .nc file
topography_dataset = nc.Dataset(topography_path) # retrieving the topography map dataset
print(topography_dataset) # viewing the variables contained
map_latitudes = topography_dataset.variables['latitude'][:] # extracting the latitude variable
map_longitudes = topography_dataset.variables['longitude'][:] # extracting the longitude variable
map_altitudes = topography_dataset.variables['alt'][:] # extracting the altitude variable

#slightly shifting the map data so that it aligns with my longitude range (-180 - 180)
map_center = len(map_longitudes) // 2 # horizontal center of the map
left_half = map_altitudes[:, :map_center] # extracting all altitude data from left to center
right_half = map_altitudes[:, map_center:] # extracting all altitude data from center to right
map_altitudes_shifted = np.hstack((right_half, left_half)) # switch left and right altitude data
map_longitudes_shifted = map_longitudes - 180 # subtracting 180 to shift the range

plt.contourf(map_longitudes_shifted, map_latitudes, map_altitudes_shifted, cmap='OrRd')
# initialising contour plot with cmap 'OrRd' (because it looked nice)
plt.colorbar(label='Altitude (meters)', orientation='horizontal')
# putting my color bar at the bottom
plt.xlabel('Longitude')
plt.ylabel('Latitude')
#adding all the rectangles to highlight my cropped regions
set_01 = patches.Rectangle((cropped_regions[0][2][0], cropped_regions[0][2][2]),
                           # extracting degree left bottom starting point
                           cropped_regions[0][2][1] - cropped_regions[0][2][0],
                           # that is the range/ difference that it goes to the left
                           cropped_regions[0][2][3] - cropped_regions[0][2][2],
                           # that is the range/ difference that it goes up
                           linewidth=1, edgecolor='#000000', facecolor='none', linestyle='-') # style
plt.gca().add_patch(set_01) # adding the highlighted area
set_02 = patches.Rectangle((cropped_regions[1][2][0], cropped_regions[1][2][2]),
                           cropped_regions[1][2][1] - cropped_regions[1][2][0],
                           cropped_regions[1][2][3] - cropped_regions[1][2][2],
                           linewidth=1, edgecolor='#7F7F7F', facecolor='none', linestyle='-')
plt.gca().add_patch(set_02)
set_03 = patches.Rectangle((cropped_regions[2][2][0], cropped_regions[2][2][2]),
                           cropped_regions[2][2][1] - cropped_regions[2][2][0],
                           cropped_regions[2][2][3] - cropped_regions[2][2][2],
                           linewidth=1, edgecolor='#C4C4C4', facecolor='none', linestyle='-')
plt.gca().add_patch(set_03)
set_04 = patches.Rectangle((cropped_regions[3][2][0], cropped_regions[3][2][2]),
                           cropped_regions[3][2][1] - cropped_regions[3][2][0],
                           cropped_regions[3][2][3] - cropped_regions[3][2][2],
                           linewidth=1, edgecolor='#00A44A', facecolor='none', linestyle='-')
plt.gca().add_patch(set_04)
set_05 = patches.Rectangle((cropped_regions[4][2][0], cropped_regions[4][2][2]),
                           cropped_regions[4][2][1] - cropped_regions[4][2][0],
                           cropped_regions[4][2][3] - cropped_regions[4][2][2],
                           linewidth=1, edgecolor='#8ED973', facecolor='none', linestyle='-')
plt.gca().add_patch(set_05)
set_06 = patches.Rectangle((cropped_regions[5][2][0], cropped_regions[5][2][2]),
                           cropped_regions[5][2][1] - cropped_regions[5][2][0],
                           cropped_regions[5][2][3] - cropped_regions[5][2][2],
                           linewidth=1, edgecolor='#C2F1C8', facecolor='none', linestyle='-')
plt.gca().add_patch(set_06)
# adding a legend for my rectangles
frame_legend = [
    patches.Patch(edgecolor='#000000', linestyle='-', facecolor='none', linewidth=1, label='Set 1'),
    patches.Patch(edgecolor='#7F7F7F', linestyle='-', facecolor='none', linewidth=1, label='Set 2'),
    patches.Patch(edgecolor='#C4C4C4', linestyle='-', facecolor='none', linewidth=1, label='Set 3'),
    patches.Patch(edgecolor='#00A44A', linestyle='-', facecolor='none', linewidth=1, label='Set 4'),
    patches.Patch(edgecolor='#8ED973', linestyle='-', facecolor='none', linewidth=1, label='Set 5'),
    patches.Patch(edgecolor='#C2F1C8', linestyle='-', facecolor='none', linewidth=1, label='Set 6'),
] # by manually replacing the line type, I saved the file in multiple variations
plt.legend(handles=frame_legend, loc='upper right') # defining where my legend is shown on the plot
plt.gca().set_aspect('equal', adjustable='box') # defining aspect ratio
plt.savefig("C:/Users/Lenovo/Desktop/image_material/04_cropped_region_overview/topography_map_cropped_regions_line.png", format='png', dpi=300)
# saving the image to my folder
plt.show() # displaying plot in console




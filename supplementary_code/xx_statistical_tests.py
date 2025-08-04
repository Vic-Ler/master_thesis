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
from scipy import stats
from scipy.stats import shapiro

def extract_correlation_coefficients(folder_path):
    '''
    input: folder path of all the nc files of a sequence
    output: correlation coefficient list of all sequences flattened in an array
    '''
    pattern = os.path.join(folder_path, '*.nc')  # pattern to identify nc files in folder
    nc_files = glob.glob(pattern)  # retrieving all paths of nc files in folder
    correlation_coefficients = [] # empty array for storing all correlation coefficients of the sequence
    for file_path in nc_files:
        data = nc.Dataset(file_path) # extracting the data
        corr = np.array(data.variables["Civ2_C"][:]) # extract correlation from data as numpy array
        correlation_coefficients.append(corr)
    return np.array(correlation_coefficients).flatten()

# ORIGINAL IMAGES
c_original_22112021 = extract_correlation_coefficients("C:/Users/Lenovo/Desktop/MSC DATA SCIENCE/MSC_PROJECT/THESIS/image_material/03_cropped_images/22_11_2021/10_min.civ")
c_original_04122021 = extract_correlation_coefficients("C:/Users/Lenovo/Desktop/MSC DATA SCIENCE/MSC_PROJECT/THESIS/image_material/03_cropped_images/04_12_2021/10_min.civ")
c_original_24122021 = extract_correlation_coefficients("C:/Users/Lenovo/Desktop/MSC DATA SCIENCE/MSC_PROJECT/THESIS/image_material/03_cropped_images/24_12_2021/10_min.civ")
c_original_2310202301 = extract_correlation_coefficients("C:/Users/Lenovo/Desktop/MSC DATA SCIENCE/MSC_PROJECT/THESIS/image_material/03_cropped_images/23_10_2023_01/30_min.civ")
c_original_2310202302 = extract_correlation_coefficients("C:/Users/Lenovo/Desktop/MSC DATA SCIENCE/MSC_PROJECT/THESIS/image_material/03_cropped_images/23_10_2023_02/30_min.civ")
c_original_2310202303 = extract_correlation_coefficients("C:/Users/Lenovo/Desktop/MSC DATA SCIENCE/MSC_PROJECT/THESIS/image_material/03_cropped_images/23_10_2023_03/30_min.civ")
# HISTO MATCHED IMAGES
c_matched_22112021 = extract_correlation_coefficients("C:/Users/Lenovo/Desktop/MSC DATA SCIENCE/MSC_PROJECT/THESIS/image_material/16_histo_matched_images/22_11_2021/10_min.civ")
c_matched_04122021 = extract_correlation_coefficients("C:/Users/Lenovo/Desktop/MSC DATA SCIENCE/MSC_PROJECT/THESIS/image_material/16_histo_matched_images/04_12_2021/10_min.civ")
c_matched_24122021 = extract_correlation_coefficients("C:/Users/Lenovo/Desktop/MSC DATA SCIENCE/MSC_PROJECT/THESIS/image_material/16_histo_matched_images/24_12_2021/10_min.civ")
c_matched_2310202301 = extract_correlation_coefficients("C:/Users/Lenovo/Desktop/MSC DATA SCIENCE/MSC_PROJECT/THESIS/image_material/16_histo_matched_images/23_10_2023_01/30_min.civ")
c_matched_2310202302 = extract_correlation_coefficients("C:/Users/Lenovo/Desktop/MSC DATA SCIENCE/MSC_PROJECT/THESIS/image_material/16_histo_matched_images/23_10_2023_02/30_min.civ")
c_matched_2310202303 = extract_correlation_coefficients("C:/Users/Lenovo/Desktop/MSC DATA SCIENCE/MSC_PROJECT/THESIS/image_material/16_histo_matched_images/23_10_2023_3/30_min.civ")
# HISTO EQUAL IMAGES
c_he_22112021 = extract_correlation_coefficients("C:/Users/Lenovo/Desktop/MSC DATA SCIENCE/MSC_PROJECT/THESIS/image_material/18_histo_equal_images/22_11_2021/10_min.civ")
c_he_04122021 = extract_correlation_coefficients("C:/Users/Lenovo/Desktop/MSC DATA SCIENCE/MSC_PROJECT/THESIS/image_material/18_histo_equal_images/04_12_2021/10_min.civ")
c_he_24122021 = extract_correlation_coefficients("C:/Users/Lenovo/Desktop/MSC DATA SCIENCE/MSC_PROJECT/THESIS/image_material/18_histo_equal_images/24_12_2021/10_min.civ")
c_he_2310202301 = extract_correlation_coefficients("C:/Users/Lenovo/Desktop/MSC DATA SCIENCE/MSC_PROJECT/THESIS/image_material/18_histo_equal_images/23_10_2023_01/30_min.civ")
c_he_2310202302 = extract_correlation_coefficients("C:/Users/Lenovo/Desktop/MSC DATA SCIENCE/MSC_PROJECT/THESIS/image_material/18_histo_equal_images/23_10_2023_02/30_min.civ")
c_he_2310202303 = extract_correlation_coefficients("C:/Users/Lenovo/Desktop/MSC DATA SCIENCE/MSC_PROJECT/THESIS/image_material/18_histo_equal_images/23_10_2023_03/30_min.civ")
# CLAHE IMAGES
c_clahe_22112021 = extract_correlation_coefficients("C:/Users/Lenovo/Desktop/MSC DATA SCIENCE/MSC_PROJECT/THESIS/image_material/20_adapt_histo_images/22_11_2021/10_min.civ")
c_clahe_04122021 = extract_correlation_coefficients("C:/Users/Lenovo/Desktop/MSC DATA SCIENCE/MSC_PROJECT/THESIS/image_material/20_adapt_histo_images/04_12_2021/10_min.civ")
c_clahe_24122021 = extract_correlation_coefficients("C:/Users/Lenovo/Desktop/MSC DATA SCIENCE/MSC_PROJECT/THESIS/image_material/20_adapt_histo_images/24_12_2021/10_min.civ")
c_clahe_2310202301 = extract_correlation_coefficients("C:/Users/Lenovo/Desktop/MSC DATA SCIENCE/MSC_PROJECT/THESIS/image_material/20_adapt_histo_images/23_10_2023_01/30_min.civ")
c_clahe_2310202302 = extract_correlation_coefficients("C:/Users/Lenovo/Desktop/MSC DATA SCIENCE/MSC_PROJECT/THESIS/image_material/20_adapt_histo_images/23_10_2023_02/30_min.civ")
c_clahe_2310202303 = extract_correlation_coefficients("C:/Users/Lenovo/Desktop/MSC DATA SCIENCE/MSC_PROJECT/THESIS/image_material/20_adapt_histo_images/23_10_2023_03/30_min.civ")
# SIGMOID IMAGES
c_sig_22112021 = extract_correlation_coefficients("C:/Users/Lenovo/Desktop/MSC DATA SCIENCE/MSC_PROJECT/THESIS/image_material/22_sigmoid_filter_images/22_11_2021/10_min.civ")
c_sig_04122021 = extract_correlation_coefficients("C:/Users/Lenovo/Desktop/MSC DATA SCIENCE/MSC_PROJECT/THESIS/image_material/22_sigmoid_filter_images/04_12_2021/10_min.civ")
c_sig_24122021 = extract_correlation_coefficients("C:/Users/Lenovo/Desktop/MSC DATA SCIENCE/MSC_PROJECT/THESIS/image_material/22_sigmoid_filter_images/24_12_2021/10_min.civ")
c_sig_2310202301 = extract_correlation_coefficients("C:/Users/Lenovo/Desktop/MSC DATA SCIENCE/MSC_PROJECT/THESIS/image_material/22_sigmoid_filter_images/23_10_2023_01/30_min.civ")
c_sig_2310202302 = extract_correlation_coefficients("C:/Users/Lenovo/Desktop/MSC DATA SCIENCE/MSC_PROJECT/THESIS/image_material/22_sigmoid_filter_images/23_10_2023_02/30_min.civ")
c_sig_2310202303 = extract_correlation_coefficients("C:/Users/Lenovo/Desktop/MSC DATA SCIENCE/MSC_PROJECT/THESIS/image_material/22_sigmoid_filter_images/23_10_2023_03/30_min.civ")

def qq_plot_shapiro(array, sig_level=0.05):
    """
    Input: array of data
    Output: QQ Plot and Shapiro-Wilk test result
    """
    stat, p_value = shapiro(array)
    print("Shapiro Stats: " + str(stat) + " " + str(p_value))
    if p_value > sig_level:
        title_var = f"P-value: {p_value:.4f} - Data is normally distributed"
    else:
        title_var = f"P-value: {p_value:.4f} - Data is not normally distributed"
    stats.probplot(array, dist="norm", plot=plt, color = "black")
    plt.title(title_var)
    plt.show()
#theoretically, strictly speaking the data is not normally distributed, however,
#the sample size is quite large and therefore we can assume that the central limit theorem applies
#which means I think it's still reasonable to do the t-test. (the results also seem quite reasonable)

def t_test(original_01, array_02, significance = 0.05):
    """
    array 01 = original data
    array 02 = different data
    output = t test results
    """
    t_statistic, p_value = stats.ttest_ind(original_01, array_02) # apply t-test
    mean_01 = original_01.mean()
    mean_02 = array_02.mean()
    if mean_02 > mean_01:
        direction = " Array 2 has a higher mean than Array 1 (original data)."
    else:
        direction = " Array 1 (original data) has a higher mean than Array 2."
    print(f"T-statistic: {t_statistic}")
    print(f"P-value: {np.round(p_value, 4)}")
    alpha = significance  # significance level
    if p_value < alpha:
        print("There is a significant difference between the two arrays." + direction)
    else:
        print("There is no significant difference between the two arrays." + direction)

t_test(c_original_22112021, c_matched_22112021)
t_test(c_original_22112021, c_he_22112021)
t_test(c_original_22112021, c_clahe_22112021)
t_test(c_original_22112021, c_sig_22112021)
np.mean(c_original_22112021)
np.mean(c_clahe_22112021)
#plt.hist(c_original_22112021, 100, color = "black")
#plt.show()

#ALL RESULTS
results_22112021 = (stats.ttest_ind(c_original_22112021, c_matched_22112021)[1],
                    stats.ttest_ind(c_original_22112021, c_he_22112021)[1],
                    stats.ttest_ind(c_original_22112021, c_clahe_22112021)[1],
                    stats.ttest_ind(c_original_22112021, c_sig_22112021)[1])
results_04122021 = (stats.ttest_ind(c_original_04122021, c_matched_04122021)[1],
                    stats.ttest_ind(c_original_04122021, c_he_04122021)[1],
                    stats.ttest_ind(c_original_04122021, c_clahe_04122021)[1],
                    stats.ttest_ind(c_original_04122021, c_sig_04122021)[1])
results_24122021 = (stats.ttest_ind(c_original_24122021, c_matched_24122021)[1],
                    stats.ttest_ind(c_original_24122021, c_he_24122021)[1],
                    stats.ttest_ind(c_original_24122021, c_clahe_24122021)[1],
                    stats.ttest_ind(c_original_24122021, c_sig_24122021)[1])
results_2310202301 = (stats.ttest_ind(c_original_2310202301, c_matched_2310202301)[1],
                    stats.ttest_ind(c_original_2310202301, c_he_2310202301)[1],
                    stats.ttest_ind(c_original_2310202301, c_clahe_2310202301)[1],
                    stats.ttest_ind(c_original_2310202301, c_sig_2310202301)[1])
results_2310202302 = (stats.ttest_ind(c_original_2310202302, c_matched_2310202302)[1],
                    stats.ttest_ind(c_original_2310202302, c_he_2310202302)[1],
                    stats.ttest_ind(c_original_2310202302, c_clahe_2310202302)[1],
                    stats.ttest_ind(c_original_2310202302, c_sig_2310202302)[1])
results_2310202303 = (stats.ttest_ind(c_original_2310202303, c_matched_2310202303)[1],
                    stats.ttest_ind(c_original_2310202303, c_he_2310202303)[1],
                    stats.ttest_ind(c_original_2310202303, c_clahe_2310202303)[1],
                    stats.ttest_ind(c_original_2310202303, c_sig_2310202303)[1])

results_df = pd.DataFrame({'Methods': ["HISTOGRAM-SPECIFICATION","HE","CLAHE","SIGMOID"],
                           '22.11.2021': np.array(np.round(results_22112021,4)),
                           '04.12.2021': np.array(np.round(results_04122021, 4)),
                           '24.12.2021': np.array(np.round(results_24122021,4)),
                           '23.10.2023 - 01': np.array(np.round(results_2310202301, 4)),
                           '23.10.2023 - 02': np.array(np.round(results_2310202302, 4)),
                           '23.10.2023 - 03': np.array(np.round(results_2310202303, 4))})
results_df.to_csv("C:/Users/Public/Desktop/correlation_test_summary.csv", index=False)

original = np.concatenate((c_original_22112021, c_original_04122021, c_original_24122021, c_original_2310202301, c_original_2310202302, c_original_2310202303))
matched = np.concatenate((c_matched_22112021, c_matched_04122021, c_matched_24122021, c_matched_2310202301, c_matched_2310202302, c_matched_2310202303))
he = np.concatenate((c_he_22112021, c_he_04122021, c_he_24122021, c_he_2310202301, c_he_2310202302, c_he_2310202303))
clahe = np.concatenate((c_clahe_22112021, c_clahe_04122021, c_clahe_24122021, c_clahe_2310202301, c_clahe_2310202302, c_clahe_2310202303))
sigmoid = np.concatenate((c_sig_22112021, c_sig_04122021, c_sig_24122021, c_sig_2310202301, c_sig_2310202302, c_sig_2310202303))

t_test(original, matched)
t_test(original, he)
t_test(original, clahe)
t_test(original, sigmoid)

path = "C:/Users/Lenovo/Desktop/MSC DATA SCIENCE/MSC_PROJECT/THESIS/image_material/03_cropped_images/22_11_2021/10_min"
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
extracted_test = extract_pixel_data(path)
extracted_test[0]
len(np.unique(extracted_test[0]))
#206

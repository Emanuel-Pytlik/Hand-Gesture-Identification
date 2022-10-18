#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##################################
###### FUNCTION DEFINITIONS ######
##################################

# Definition of a function replacing zero-entries with the last valid entry (only used for the arrays containing the number of measurements)
def modify_array(array):
    value = 0
    for i in range(len(array)):
        if (array[i] > value):
            value = array[i]
        else:
            array[i] = value


# Definition of a function that linearly resamples the given array            
def resample_linear(array, desired_size):
    array_length = len(array)
    difference = desired_size - array_length
    if (difference > 0):
        # Insert evenly spaced NaNs into the arrays
        for j in range(1, difference + 1):
            index = (array_length//difference) * j           
            array = np.insert(array, index, 'nan')
        # Perform a linear interpolation
        return ((pd.Series(array)).interpolate()).to_numpy()
    else:
        return array


# Definition of a function that calculates and prints out the most important metrics of a dataset
def print_metrics(measurement_count, sampling_time_sec):    
    # Calculate the metrics of the dataset
    maximum_measurement_difference = np.amax(measurement_count) - np.amin(measurement_count)
    maximum_frequency = np.amax(measurement_count) / (sampling_time_sec)
    minimum_frequency = np.amin(measurement_count) / (sampling_time_sec)
    average_number_measurements = np.mean([measurement_count])
    ratio_missing_measurements = maximum_measurement_difference/average_number_measurements

    # Print out the metrics of the dataset
    print("Maximum difference of measurements: ", "%.2f" % maximum_measurement_difference)
    print("Maximum frequency: ", "%.2f" % maximum_frequency)
    print("Minimum frequency: ", "%.2f" % minimum_frequency)
    print("Average number of measurements: ", "%.2f" % average_number_measurements)
    print("Ration between maximum measurement difference and average number of measurements: " "%.2f" % ratio_missing_measurements)


# Definition of a plot function to standardize plots
def plot(title, xlabel, ylabel, array, dimension):
    
    # Do some settings for the interactive plots  
    fig = plt.figure()
    fig.canvas.toolbar_visible = True
    fig.canvas.header_visible = False    
        
    # Do the actual plot
    plt.title(title)  
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Distinguish depending on the dimension of the array
    if (dimension == 1):
        plt.plot(array)
    elif (dimension == 2):
        for i in range(len(array)):
            plt.plot(array[i], label=label[i])    
        plt.legend()
    else:
        return


# Definition of a function that plots confusion matrices as heatmaps
def plot_heatmap(confusion_matrix, xlabel, ylabel):
    plt.clf()    
    heatmap = seaborn.heatmap(confusion_matrix, vmin=0, vmax=1, cmap=seaborn.color_palette("Blues", as_cmap=True), annot=True, cbar=False, square=True, xticklabels=['relaxed', 'flat', 'spreaded', 'fist', 'rot. fist', 'tilt r.', 'tilt l.', 'thumb up'], yticklabels=['relaxed', 'flat', 'spreaded', 'fist', 'rot. fist', 'tilt r.', 'tilt l.', 'thumb up'])    
    heatmap.set(xlabel = xlabel, ylabel = ylabel)
    heatmap.figure.tight_layout()
    plt.show()


# Definition of a function that calculates the mean pressure of each sensor and saves it as a numpy array
def calculate_mean(array, number_of_sensors, path):
    average = np.zeros(number_of_sensors)
    for i in range(number_of_sensors):
        average[i] = np.mean(array[:,i:i+1,:])
    np.save(path, average)


# Definition of a function that outputs outliers according to the absolute values of the measurements    
def print_outliers_absolute(array):
    outliers = [np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)]    
    for i in range(len(array)):       
        for j in range(len(array[i])):
            # Check whether the measurement is within the predefined bounds
            if (array[i][j] < LOWER_THRESHOLD_ABSOLUTE_OUTLIER_DETECTION or array[i][j] > UPPER_THRESHOLD_ABSOLUTE_OUTLIER_DETECTION):
                outliers[i] = np.append(outliers[i], array[i][j])               
    return outliers


# Definition of a function that removes outliers (with linear interpolation) according to the absolute values of the measurements           
def remove_outliers_absolute(array):
    for i in range(len(array)):
        for j in range(len(array[i])):
            # Check whether the measurement is within the predefined bounds
            if (array[i][j] < LOWER_THRESHOLD_ABSOLUTE_OUTLIER_DETECTION or array[i][j] > UPPER_THRESHOLD_ABSOLUTE_OUTLIER_DETECTION):
                # The first measurement is an outlier
                if (j == 0):
                    start = j
                    end = j
                    while (array[i][end] < LOWER_THRESHOLD_ABSOLUTE_OUTLIER_DETECTION or array[i][end] > UPPER_THRESHOLD_ABSOLUTE_OUTLIER_DETECTION):
                        end += 1         
                    array[i][start:end] =  array[i][end]
                # The last measurement is an outlier
                elif (j == len(array[i])-1):                    
                    array[i][j] = array[i][j-1]
                # There are outliers which are neither the first nor the last measurment
                else:
                    start = j-1
                    end = j                                                                 
                    while (array[i][end] < LOWER_THRESHOLD_ABSOLUTE_OUTLIER_DETECTION or array[i][end] > UPPER_THRESHOLD_ABSOLUTE_OUTLIER_DETECTION):
                        end += 1
                        # Ckeck whether the end of the array is reached and whether the last measurement is an outlier
                        if (end == len(array[i])-1 and (array[i][end] < LOWER_THRESHOLD_ABSOLUTE_OUTLIER_DETECTION or array[i][end] > UPPER_THRESHOLD_ABSOLUTE_OUTLIER_DETECTION)):
                            array[i][start:end+1] = array[i][start]
                            return                                
                    array[i][start:end+1] = np.linspace(array[i][start], array[i][end], end+1-start)


# Definition of a function that calculates the hand-designed features for a dataset
def get_features(array, path, filename, number_of_sensors, number_of_features_per_sensor, number_of_samples):
    # Initialize the arrays
    features = np.zeros(number_of_sensors*number_of_features_per_sensor)
    maximas = np.zeros(number_of_sensors)
    minimas = np.zeros(number_of_sensors)
    variances = np.zeros(number_of_sensors)
    means = np.zeros(number_of_sensors)
    means_minus_references = np.zeros(number_of_sensors)
    # Load the reference values
    references = np.load(path + filename + '_references.npy')
    for i in range(number_of_sensors):
        # Calculate the maximas, minimas, variances, means and the means minus the references
        maximas [i] = np.amax(array[i])
        minimas [i] = np.amin(array[i])
        variances[i] = np.var(array[i])
        means[i] = np.mean(array[i])
        means_minus_references[i] = means[i] - references[i]
    for i in range(number_of_sensors):
        # Calculate the features given in the paper for every sensor
        features[i] = means[i] / np.max(means)
        features[number_of_sensors+i] = (means_minus_references[i] - np.min(means_minus_references)) / (np.max(means_minus_references) - np.min(means_minus_references))
        # Calculate additional features for every sensor
        features[2*number_of_sensors+i] = variances[i] / np.max(variances)
        features[3*number_of_sensors+i] = (maximas[i] - minimas[i]) / np.amax(maximas) ### Try: np.amax(maximas-minimas) 
        features[4*number_of_sensors+i] = (np.polyfit(range(number_of_samples), array[i], 1))[0]
        features[5*number_of_sensors+i] = feature_extraction.feature_calculators.cid_ce(array[i], False)    
    return features


# Definition of a function that sets up and saves the feature vector
def make_features(path_1, path_2, filename, number_of_gestures, number_of_repetitions, number_of_sensors, number_of_features_per_sensor, number_of_samples):
    features = np.zeros(shape=(number_of_gestures, number_of_repetitions, number_of_sensors*number_of_features_per_sensor))
    for gesture in range(number_of_gestures):
        array = np.load(path_1 + filename + '_gesture' + str(gesture) + '.npy')
        for repetition in range(number_of_repetitions):
            features[gesture][repetition] = get_features(array[repetition], path_2, filename, number_of_sensors, number_of_features_per_sensor, number_of_samples)
    np.save(path_2 + filename + '_features.npy', features)


# Definition of a function that calculates the indices, which are in the middle of each segment of the study
def get_sample_indices(timing, measurement_count, sampling_time_sec, offset=0):
    # Initialize the array of the sample indices and calculate the maximum sampling frequency
    sample_indices = np.zeros(len(timing))    
    sampling_frequency = np.amax(measurement_count) / (sampling_time_sec)
    for i in range(len(timing)):
        sample_indices[i] = np.round(timing[i] * sampling_frequency) + offset
    return sample_indices


# Definition of a function that determines the intervalls of the sample segments, extracts the samples and stores them into an numpy array
def get_samples(array, gesture_indices, sample_indices, interval_samples, path):
    # Calculate the width of the half interval and initialize the array which saves the results    
    half_interval_samples = interval_samples // 2
    result = np.full((len(gesture_indices), len(array), interval_samples), 0, dtype=float)
    index = 0
    for i in gesture_indices:
        middle_index = sample_indices[i-1]
        start_index = middle_index - half_interval_samples
        end_index = middle_index + half_interval_samples
        result[index] = array[0:int(len(array)),int(start_index):int(end_index)]
        index += 1
    np.save(path, result)
    

# Definition of a function that changes the format of the original samples to a format with is compatible with the tsfresh library
def format_array(path_1, path_2, filename, number_of_sensors, number_of_gestures, number_of_repetitions, number_of_samples):
    # Initialize the resulting array with the proper dimensions   
    result = np.zeros((number_of_samples*number_of_repetitions*number_of_gestures, number_of_sensors+1))
    # Format the new array according to the input option 1 of the tsfresh extract_features() method
    for gesture in range(number_of_gestures):
        # Load the array, which should be formated differently
        array = np.load(path_1 + filename + '_gesture' + str(gesture) + '.npy')        
        for repetition in range(number_of_repetitions):
            for sample in range(number_of_samples):
                # Consider the index offsets which is caused by the gestures and repetitions, since the new array is only 2d
                result[sample+repetition*number_of_samples+gesture*number_of_repetitions*number_of_samples][0] = repetition+gesture*number_of_repetitions
                for sensor in range(number_of_sensors):
                    # Consider the index offsets which is caused by the gestures and repetitions, since the new array is only 2d
                    result[sample+repetition*number_of_samples+gesture*number_of_repetitions*number_of_samples][sensor+1] = array[repetition][sensor][sample]
    np.save(path_2 + filename + '.npy', result)


# In[ ]:


#########################################
###### 1.IMPORTS AND CONFIGURATION ######
#########################################

# Enable interactive plots
get_ipython().run_line_magic('matplotlib', 'widget')

# # Install these libraries when running for the first time
# !pip install tsfresh

# Import all necessary libraries
import seaborn
import numpy as np
import pandas as pd
import xgboost as xgb
import scipy.signal as signal
import matplotlib.pyplot as plt
from tqdm import tqdm
from tsfresh import extract_features
from tsfresh import feature_extraction
from tsfresh.utilities.dataframe_functions import impute
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Define the configuration parameters
LOWER_THRESHOLD_ABSOLUTE_OUTLIER_DETECTION = 900
UPPER_THRESHOLD_ABSOLUTE_OUTLIER_DETECTION = 2300


# In[ ]:


###########################
###### 2.DATA IMPORT ######
###########################

# Load the raw data from a .txt file into an array of lists
with open('Data/Study/Data Recordings/Simon_3.txt',"r") as file:           
        all_data = (map(float, line.split()) for line in file)  
        raw_data_unsorted = [list(line) for line in all_data]
file.close()

# Determine the maximum iteration of the sampling process
maximum_iteration = int(raw_data_unsorted[len(raw_data_unsorted)-1][1])

# Array initializations
measurement_count = np.zeros(6)
label = np.array(["Sensor 1", "Sensor 2", "Sensor 3", "Sensor 4", "Sensor 5", "Sensor 6"])
raw_data = [np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)]
array_elements = [np.zeros(maximum_iteration+1, dtype=int), np.zeros(maximum_iteration+1, dtype=int), np.zeros(maximum_iteration+1, dtype=int), np.zeros(maximum_iteration+1, dtype=int), np.zeros(maximum_iteration+1, dtype=int), np.zeros(maximum_iteration+1, dtype=int)]

# Seperate the data
for i in range(len(raw_data_unsorted)):
    sensor_identifier = raw_data_unsorted[i][0]
    iteration = int(raw_data_unsorted[i][1])
    
    # Store all measurements of each sensor in a seperate array
    if (sensor_identifier == 1):
        raw_data[0] = np.append(raw_data[0], raw_data_unsorted[i][2:len(raw_data_unsorted[i])])
        array_elements[0][iteration] = len(raw_data[0])
    elif (sensor_identifier == 2):
        raw_data[1] = np.append(raw_data[1], raw_data_unsorted[i][2:len(raw_data_unsorted[i])])
        array_elements[1][iteration] = len(raw_data[1])
    elif (sensor_identifier == 3):
        raw_data[2] = np.append(raw_data[2], raw_data_unsorted[i][2:len(raw_data_unsorted[i])])
        array_elements[2][iteration] = len(raw_data[2])
    elif (sensor_identifier == 4):
        raw_data[3] = np.append(raw_data[3], raw_data_unsorted[i][2:len(raw_data_unsorted[i])])
        array_elements[3][iteration] = len(raw_data[3])
    elif (sensor_identifier == 5):
        raw_data[4] = np.append(raw_data[4], raw_data_unsorted[i][2:len(raw_data_unsorted[i])])
        array_elements[4][iteration] = len(raw_data[4])
    elif (sensor_identifier == 6):
        raw_data[5] = np.append(raw_data[5], raw_data_unsorted[i][2:len(raw_data_unsorted[i])])
        array_elements[5][iteration] = len(raw_data[5])


# In[ ]:


########################################
###### 3.RESAMPLING AND FILTERING ######
########################################

# Determine the maximum number of measurements across all sensors
maximum_array_size = max(len(raw_data[0]), len(raw_data[1]), len(raw_data[2]), len(raw_data[3]), len(raw_data[4]), len(raw_data[5]))

# Array initializations
resampled_data = np.array([np.zeros(maximum_array_size), np.zeros(maximum_array_size), np.zeros(maximum_array_size), np.zeros(maximum_array_size), np.zeros(maximum_array_size), np.zeros(maximum_array_size)])
filtered_data = np.array([np.zeros(maximum_array_size), np.zeros(maximum_array_size), np.zeros(maximum_array_size), np.zeros(maximum_array_size), np.zeros(maximum_array_size), np.zeros(maximum_array_size)])
segmented_data = np.array([np.zeros(maximum_array_size), np.zeros(maximum_array_size), np.zeros(maximum_array_size), np.zeros(maximum_array_size), np.zeros(maximum_array_size), np.zeros(maximum_array_size)])
final_data = np.array([np.zeros(maximum_array_size), np.zeros(maximum_array_size), np.zeros(maximum_array_size), np.zeros(maximum_array_size), np.zeros(maximum_array_size)])

# Smooth out all of the zeros of the arrays which contain the number of array elements
for i in range(len(array_elements)):
    modify_array(array_elements[i])

# Indicate and remove outliers from the data
print(print_outliers_absolute(raw_data))
remove_outliers_absolute(raw_data)
   
# Perform linear resampling such that all arrays contain the same number of measurements
for i in range(len(resampled_data)):
    resampled_data[i] = resample_linear(raw_data[i], maximum_array_size)
    
# Apply a lowpass filter to the resampled data
sos = signal.butter(1, 0.3, analog=False, btype='lowpass', output='sos')
for i in range(len(filtered_data)):
    filtered_data[i] = signal.sosfiltfilt(sos,  resampled_data[i])
    
# Delete the array which corresponds to sensor 2, since this sensor did not collect any data
for i in range(len(final_data)):
    if i == 0:
        index = i
    else:
        index = i+1
    final_data[i] = filtered_data[index]


# In[ ]:


#############################
###### 4.DATA ANALYSIS ######
#############################

### CONFIGURATION ###
time_recording = 550

# Count the number of valid measurements in each array
for i in range(len(measurement_count)):
    measurement_count[i] = len(raw_data[i])

# Calculate some metrics of the dataset and print them
print_metrics(measurement_count, 550)

# Plot the number of array elements for each sensor in an interactive plot
plot("Number of array elements", "Iteration", "Number of elements", array_elements, 2)

# Plot the raw measurements of each sensor in an interactive plot
plot("Raw Measurements", 'Sample number', 'Pressure in hPa', raw_data, 2)

# Plot the resampled and filtered measurements of each sensor in an interactive plot
plot('Filtered Measurements', 'Sample number', 'Pressure in hPa', filtered_data, 2)


# In[ ]:


############################
###### 5.SEGMENTATION ######
############################

### CONFIGURATION ###
filename = 'Camila_1'
path = 'Data/Study/Samples (5 Seconds)/'
time_recording = 550
interval_samples = 1000

# Define the times in seconds, which lie in the middle of each segment of the study (according to the study video)
timing = np.array([5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155.5, 165.5, 175.5, 185.5, 195.5, 205.5, 215.5, 225.5, 235.5, 245.5, 255.5, 265.5, 275.5, 285.5, 295.5, 305.5, 315.5, 326, 336, 346, 356, 366, 376, 386, 396, 406, 416, 426, 436, 446, 456, 466, 476])
timing_camila_1 = np.array([5, 15, 25.5, 36, 46.5, 57, 67.5, 78, 88.5, 99.5, 109, 119, 129, 142, 151, 161, 171, 181, 190.5, 200.5, 210.5, 220.5, 231.5, 241.5, 251.5, 261.5, 270.5, 280.5, 290.5, 300.5, 309.5, 320, 328.5, 337.5, 347, 357, 367, 377, 388, 398, 408, 416.5, 426.5, 439, 447.5, 457, 466, 477])

# Define the occurences of each gesture within the study (according to the study video)
gesture0 = np.array([2, 4, 6, 8, 10, 12])
gesture1 = np.array([5, 18, 22, 29, 36, 47])
gesture2 = np.array([7, 19, 27, 31, 35, 43])
gesture3 = np.array([1, 15, 21, 30, 41, 45])
gesture4 = np.array([3, 17, 25, 34, 38, 48])
gesture5 = np.array([9, 14, 26, 32, 39, 44])
gesture6 = np.array([11, 16, 23, 33, 37, 46])
gesture7 = np.array([13, 20, 24, 28, 40, 42])

# Combine all arrays in one 2D array
gesture = np.array([gesture0, gesture1, gesture2, gesture3, gesture4, gesture5, gesture6, gesture7])

# Count the number of valid measurements in each array
for i in range(len(measurement_count)):
    measurement_count[i] = len(raw_data[i])

# Calculate the sample indices according to the sampling frequency (dependent on the measurement count and the duration of the sampling in seconds)
sample_indices = get_sample_indices(timing, measurement_count, time_recording)

# Extract all sample segments and save them as a numpy array
for i in range(8):
    get_samples(final_data, gesture[i], sample_indices, interval_samples, path + filename + '_gesture' + str(i) + '.npy')

# For inspection only: Indicate the intervals in the array which represents sensor 2
segmented_data[5] = np.full(len(segmented_data[0]), 800)
index = 0
for i in range(len(segmented_data[0])):
    if (index < len(sample_indices) and i == sample_indices[index]):
        segmented_data[5][i-(interval_samples//2)] = 2000        
        segmented_data[5][i+((interval_samples//2)-1)] = 2000
        index += 1


# In[ ]:


#################################
##### 6.FEATURE EXTRACTION ######
#################################

### CONFIGURATION ###
names = np.array(['Camila', 'David', 'Emanuel', 'Kornelia', 'Martin', 'Simon'])
path_source = 'Data/Study/Samples (5 Seconds)/'
path_destination = 'Data/Study/Features (5 Seconds)/'
number_of_files = 18
number_of_gestures = 8
number_of_repetitions = 6
number_of_sensors = 5
number_of_features_per_sensor = 6
number_of_samples = 1000

# Iterate over the names
for name in names:
    
    # Indices for 'Emanuel' start at 5 instead of 1
    if name == 'Emanuel':
        offset = 4
    else:
        offset = 0
    
    # Iterate over the sessions
    for session in range(1,4):
        
        # Calculate and save the mean from the data recorded with gesture0 and use it as the reference (needs to be done for every filename)
        calculate_mean(np.load(path_source + name + '_' + str(session+offset) + '_gesture0.npy'), number_of_sensors, path_destination + name + '_' + str(session+offset) + '_references.npy')

        # Create and save the feature vector
        make_features(path_source, path_destination, name + '_' + str(session+offset), number_of_gestures, number_of_repetitions, number_of_sensors, number_of_features_per_sensor, number_of_samples)

# Create and save the label vector for one session and for all sessions (number of files)
labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7])
labels_all = (np.array([labels, labels, labels, labels, labels, labels, labels, labels, labels, labels, labels, labels, labels, labels, labels, labels, labels, labels])).reshape(len(labels)*number_of_files)
np.save('Data/Study/Labels/labels.npy', labels)
np.save('Data/Study/Labels/labels_all.npy', labels_all)


# In[ ]:


##################################################
###### 7.FORMAT ARRAYS ACCORDING TO TSFRESH ######
##################################################

### CONFIGURATION ###
names = np.array(['Camila', 'David', 'Emanuel', 'Kornelia', 'Martin', 'Simon'])
path_source = 'Data/Study/Samples (5 Seconds)/'
path_destination = 'Data/Study/Formatted Arrays (5 Seconds)/'
number_of_files = 18
number_of_sensors = 5
number_of_gestures = 8
number_of_repetitions = 6
number_of_samples = 1000

# Iterate over the names
for name in names:    
    # Indices for 'Emanuel' start at 5 instead of 1
    if name == 'Emanuel':
        offset = 4
    else:
        offset = 0
        
    # Iterate over the sessions
    for session in range(1,4):
        # Format the samples of every session of every participant
        format_array(path_source, path_destination, name + '_' + str(session+offset), number_of_sensors, number_of_gestures, number_of_repetitions, number_of_samples)


# In[ ]:


###############################################################
###### 8.FEATURE EXTRACTION OF ALL FEATURES WITH TSFRESH ######
###############################################################

### CONFIGURATION ###
names = np.array(['Camila', 'David', 'Emanuel', 'Kornelia', 'Martin', 'Simon'])
path_source = 'Data/Study/Formatted Arrays (5 Seconds)/'
path_destination = 'Data/Study/Features (5 Seconds, tsfresh)/'
number_of_files = 18
number_of_sessions = 3
number_of_sensors = 5
number_of_gestures = 8
number_of_repetitions = 6
number_of_samples = 1000

# Define auxiliary variables and an array
name_index = 0
array_length = number_of_samples*number_of_repetitions*number_of_gestures
formatted_array_all = np.zeros((array_length*number_of_files, number_of_sensors+1))

# Iterate over the names
for name in names:    
    # Indices for 'Emanuel' start at 5 instead of 1
    if name == 'Emanuel':
        offset = 4
    else:
        offset = 0
        
    # Iterate over the sessions
    session_index = 0
    for session in range(1,4):
        # Load the feature vector for the test session
        formatted_array = np.load(path_source + name + '_' + str(session+offset) + '.npy')
        # Calculate the index according to the sessions and names
        index = session_index+name_index*number_of_sessions
        # Concatenate the formatted arrays into one array
        formatted_array_all[index*array_length:index*array_length+array_length] = formatted_array
        #Increment the session index
        session_index += 1
    #Increment the name index
    name_index += 1

# Correct the first column, where the id`s are saved
for i in range(number_of_files*number_of_gestures*number_of_repetitions):
    formatted_array_all[i*number_of_samples:i*number_of_samples+number_of_samples,0:1] = i

# Convert the numpy array into a pandas dataframe
formatted_dataframe_all = pd.DataFrame(data=formatted_array_all, columns=['id', 'sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5' ])
    
# Extract and impute the features
extracted_features_all = extract_features(formatted_dataframe_all, column_id="id", column_kind=None, column_value=None)
impute(extracted_features_all)

# Save the extracted features as a numpy array
np.save(path_destination + 'features_all.npy', extracted_features_all)


# In[ ]:


#######################################################################
###### 9.DETERMINE THE FEATURE IMPORTANCES USING A RANDOM FOREST ######
#######################################################################

### CONFIGURATION ###
path = 'Data/Study/Features (5 Seconds, tsfresh impurity)/'
number_of_files = 18
number_of_sessions = 3
number_of_sensors = 5
number_of_gestures = 8
number_of_repetitions = 6
number_of_samples = 1000

# Print the program progress
for i in tqdm (range (100), desc="Loading..."):
    
    # Load the concatenated feature vector and the concatenated label vector
    features_all = np.load(path + 'features_all.npy')   
    labels_all = np.load('Data/Study/Labels/labels_all.npy')
    
    # Fitting the random forest classifier
    forest = RandomForestClassifier(random_state=0)
    forest.fit(features_all, labels_all)
    
    # Determining the feature importances
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

# Save the array which stores the feature importances
np.save(path + 'feature_importances.npy', importances)


# In[ ]:


###################################################
###### 10.SELECT THE MOST IMPORTANT FEATURES ######
###################################################

### CONFIGURATION ###
path = 'Data/Study/Features (5 Seconds, tsfresh impurity + own features)/'
number_of_files = 18
number_of_gestures = 8
number_of_repetitions = 6
threshold = 0.0005   # Experimental value

# Define auxiliary variables
important_features = 0
rows = number_of_files*number_of_gestures*number_of_repetitions

# Load the arrays
features_all = np.load(path + 'features_all.npy')
feature_importances = np.load(path + 'feature_importances.npy')

# Count the number of features which have an importance above the threshold
for i in range(len(feature_importances)):
    if feature_importances[i] > threshold:
        important_features += 1
        
# Print the number of important features
print(important_features)

# Initialize the array which saves the selected features
features_all_selected = np.zeros((rows,important_features))

# Select the features which have an importance above the threshold
for i in range(rows):
    features_all_selected[i] = [features_all[i][j] for j in range(len(features_all[i])) if feature_importances[j] > threshold]
    
# Save the selected features as a numpy array
np.save(path + 'features_all_selected.npy', features_all_selected)


# In[ ]:


########################################################################
###### 11.CONCATENATE THE SELECTED AND THE HAND-DESIGNED FEATURES ######
########################################################################

### CONFIGURATION ###
path_1 = 'Data/Study/Features (5 Seconds, tsfresh impurity)/'
path_2 = 'Data/Study/Features (5 Seconds)/'
number_of_sensors = 5
number_of_features_per_sensor = 6
number_of_added_features = 25

# Load the feature vectors
features_1 = np.load(path_1 + 'features_all_selected.npy')
features_2 = (np.array([np.load(path_2 + 'Camila_features.npy'), np.load(path_2 + 'David_features.npy'), np.load(path_2 + 'Emanuel_features.npy'), np.load(path_2 + 'Kornelia_features.npy'), np.load(path_2 + 'Martin_features.npy'), np.load(path_2 + 'Simon_features.npy')])).reshape(864, number_of_sensors*number_of_features_per_sensor)

# Define an array, which saves the combined feature vectors
features_combined = np.zeros((len(features_1), len(features_1[0]) + number_of_added_features))

# Combine the feature vectors
for i in range(len(features_1)):
    features_combined[i:i+1, 0:len(features_1[i])] = features_1[i]
    features_combined[i:i+1, len(features_1[i]):len(features_1[i])+number_of_added_features] = features_2[i:i+1, 0:number_of_added_features]
    
# Save the combined feature vectors
np.save(path_1 + 'features_all_merged.npy', features_combined)


# In[ ]:


##################################################################################################
###### 12.SPLIT THE CONCATENATED FEATURE VECTOS INTO FEATURE VECTORS PER PERSON PER SESSION ######
##################################################################################################

### CONFIGURATION ###
names = np.array(['Camila', 'David', 'Emanuel', 'Kornelia', 'Martin', 'Simon'])
path = 'Data/Study/Features (5 Seconds, tsfresh impurity + own features)/'
number_of_files = 18
number_of_sessions = 3
number_of_sensors = 5
number_of_gestures = 8
number_of_repetitions = 6

# Load the feature vector, which contains all features of the dataset and define an auxiliary variable
name_index = 0
features_all = np.load(path + 'features_all_merged.npy')

# Iterate over the names
for name in names:    
    # Indices for 'Emanuel' start at 5 instead of 1
    if name == 'Emanuel':
        offset = 4
    else:
        offset = 0
        
    # Iterate over the sessions
    session_index = 0
    for session in range(1,4):        
        # Calculate the index according to the sessions and names
        index = session_index+name_index*number_of_sessions
        # Split the combined feature vector by participant and session and save it as a numpy array
        np.save(path + name + '_' + str(session+offset) + '_features.npy', features_all[index*(number_of_repetitions*number_of_gestures):index*(number_of_repetitions*number_of_gestures)+(number_of_repetitions*number_of_gestures)])
        #Increment the session index
        session_index += 1
    #Increment the name index
    name_index += 1


# In[ ]:


##################################################################
###### 13.CLASSIFICATION AND CROSS-SESSION CONFUSION MATRIX ######
##################################################################

### CONFIGURATION ###
names = np.array(['Camila', 'David', 'Emanuel', 'Kornelia', 'Martin', 'Simon'])
path = 'Data/Study/Features (5 Seconds, tsfresh impurity + own features)/'
number_of_features = 469

# Load the labels and initialize the confusion matrices
labels_1_session = np.load('Data/Study/Labels/labels.npy')
average_confusion_matrix = np.full((8,8), 0)

# Create a cross-session confusion matrix
for name in names:    
    # Indices for 'Emanuel' start at 5 instead of 1
    if name == 'Emanuel':
        offset = 4
    else:
        offset = 0
        
    # Iterate over the sessions
    for session in range(1,4):
        # Load the feature vector for the test session
        features_test = (np.load(path + name + '_' + str(session+offset) + '_features.npy')).reshape(48, number_of_features)
        
        # Concatenate 2 feature vectors and reshape it to 2 dimensions for the 2 train sessions
        sessions_train = [i+offset for i in range(1,4) if i != session]
        features_train = (np.array([np.load(path + name + '_' + str(sessions_train[0]) + '_features.npy'), np.load(path + name + '_' + str(sessions_train[1]) + '_features.npy')])).reshape(96, number_of_features)        
        
        # Concatenate 2 label vectors and reshape it to 1 dimension
        labels_train = (np.array([labels_1_session, labels_1_session])).reshape(96)
        
        # Use a SVM classifier
        classifier = make_pipeline(StandardScaler(), SVC(kernel='linear', C=0.009, tol=0.0001))
        # Use a XGBoost classifier
        #classifier = xgb.XGBClassifier(n_estimators = 1000, n_jobs = -1, use_label_encoder = False, verbosity = 0, random_state = 0)
        # Use a random forest classifier        
        #classifier = RandomForestClassifier(random_state=0)        
        
        classifier.fit(features_train, labels_train)        
        
        # Predict the features on the left out session
        labels_predict = classifier.predict(features_test)
        
        # Calculate the confusion matrix
        new_confusion_matrix = confusion_matrix(labels_1_session, labels_predict, labels=[0, 1, 2, 3, 4, 5, 6, 7])  
        
        # Add up the confusion matrices        
        average_confusion_matrix += new_confusion_matrix
        
# Calculate the accuracies of the cross-session gesture classification for each gesture
accuracies = np.zeros(len(average_confusion_matrix))
for i in range(len(average_confusion_matrix)):
    accuracies[i] = average_confusion_matrix[i][i] / np.sum(average_confusion_matrix[i:i+1, :])
    
# Calculate the average accuracy and the standard deviation of the accuracies and print them
average_accuracy = np.mean(accuracies)
standard_deviation_of_accuracies = np.std(accuracies)
print('Average accuracy: ', '%.2f' % average_accuracy)
print('Standard deviation of accuracies: ', '%.2f' % standard_deviation_of_accuracies)
        
# Normalize the average confusion matrix
average_confusion_matrix = np.around(average_confusion_matrix / np.full((8,8), 108), decimals=2)
        
# Plot the confusion matrix as a heatmap
plot_heatmap(average_confusion_matrix, 'Predicted label', 'True label')


# In[ ]:


#################################################################
###### 14.CLASSIFICATION AND CROSS-PERSON CONFUSION MATRIX ######
#################################################################

### CONFIGURATION ###
names = np.array(['Camila', 'David', 'Emanuel', 'Kornelia', 'Martin', 'Simon'])
path = 'Data/Study/Features (5 Seconds, tsfresh impurity + own features)/'
number_of_features = 469

# Load the labels and initialize the confusion matrices
labels_1_person = (np.array([np.load('Data/Study/Labels/labels.npy'), np.load('Data/Study/Labels/labels.npy'), np.load('Data/Study/Labels/labels.npy')])).reshape(144)
average_confusion_matrix = np.full((8,8), 0)

# Create one single feature vector for every person
for name in names:    
    # Indices for 'Emanuel' start at 5 instead of 1
    if name == 'Emanuel':
        offset = 4
    else:
        offset = 0        
    
    # Concatenate 3 feature vectors for every person and save it as a numpy array
    np.save(path + name + '_features.npy', np.array([np.load(path + name + '_' + str(1+offset) + '_features.npy'), np.load(path + name + '_' + str(2+offset) + '_features.npy'), np.load(path + name + '_' + str(3+offset) + '_features.npy')]))

# Create a cross-person confusion matrix
for name in names:
    # Load the feature vector for the test person
    features_test = (np.load(path + name + '_features.npy')).reshape(144, number_of_features)    
    
    # Concatenate the combined feature vector of every person and reshape it to 2 dimensions for the 5 train persons
    names_train = [i for i in names if i != name]
    features_train = (np.array([np.load(path + name_train + '_features.npy') for name_train in names_train])).reshape(720, number_of_features)

    # Concatenate the label vectors of the 5 train persons and reshape it to 1 dimension
    labels_train = (np.array([labels_1_person, labels_1_person, labels_1_person, labels_1_person, labels_1_person])).reshape(720)

    # Use a SVM classifier
    classifier = make_pipeline(StandardScaler(), SVC(kernel='linear', C=0.009, tol=0.0001))
    # Use a XGBoost classifier
    #classifier = xgb.XGBClassifier(n_estimators = 1000, n_jobs = -1, use_label_encoder = False, verbosity = 0)
    # Use a random forest classifier        
    #classifier = RandomForestClassifier(random_state=0)    
    
    classifier.fit(features_train, labels_train)    

    # Predict the features on the left out person
    labels_predict = classifier.predict(features_test)

    # Calculate the confusion matrix
    new_confusion_matrix = confusion_matrix(labels_1_person, labels_predict, labels=[0, 1, 2, 3, 4, 5, 6, 7])

    # Add up the confusion matrices        
    average_confusion_matrix += new_confusion_matrix
    
# Calculate the accuracies of the cross-person gesture classification for each gesture
accuracies = np.zeros(len(average_confusion_matrix))
for i in range(len(average_confusion_matrix)):
    accuracies[i] = average_confusion_matrix[i][i] / np.sum(average_confusion_matrix[i:i+1, :])
    
# Calculate the average accuracy and the standard deviation of the accuracies and print them
average_accuracy = np.mean(accuracies)
standard_deviation_of_accuracies = np.std(accuracies)
print('Average accuracy: ', '%.2f' % average_accuracy)
print('Standard deviation of accuracies: ', '%.2f' % standard_deviation_of_accuracies)

# Normalize the average confusion matrix
average_confusion_matrix = np.around(average_confusion_matrix / np.full((8,8), 108), decimals=2)
        
# Plot the confusion matrix as a heatmap
plot_heatmap(average_confusion_matrix, 'Predicted label', 'True label')


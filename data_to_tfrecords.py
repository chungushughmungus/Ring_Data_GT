import algotom.io.loadersaver as losa
import numpy as np
import tensorflow as tf
import keras
from keras import layers, models, utils
import glob
import matplotlib.pyplot as plt
import pandas as pd
import math
import random

# Note: Turns out I formatted the entire setup incorrectly:
# TFRecords streams in PER FILE! Each image should technically be a tfrecord on its own, not in an aggregate!
# I completely screwed up the format. RIP.
# THIS ONLY PREVENTS RAM FROM GROWING - no memory leaks (it grows a bit every time it saves model weights though but drops back down)

# Note: If this runs and a RuntimeWarning pops up about invalid value encountered in divide:
# It is because the X-Center values, when processing, have 0 standard deviation so equation (arr-mean)/std divides by 0
# Because only alpha angle is changing, not beta angle and so x-center coordinates don't change

gpus = tf.config.list_physical_devices('GPU')
# DO THIS BEFORE GPU PHYSICALLY INITIALIZES!!! - Can't modify memory growth after initialization
# Requires monitoring to determine GPU memory required for training
# This grows memory - starts small, then grows as needed
# There's only 1 GPU on WS1/3 but can use for future idk
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
# Runs - CUDA GPU initialized and works

# Filepath to folder
main_file_path = "Your file path"


# Number of images to load in
# d_set_size = 5000
# Takes folder holding ring imgs/data csv files and sorts filepaths for ring imgs/csv files into specific tf.datasets
def sort_filepaths(main_file_path, img_list_ds, data_list_ds):
    # Separate into Tif, CSV files
    material_imgs = sorted(list(glob.glob(main_file_path + "/*.tif")))  # [:d_set_size]
    materials_data = sorted(list(glob.glob(main_file_path + "/*.csv")))  # [:d_set_size]
    return material_imgs, materials_data


img_list_ds = []
data_list_ds = []
img_list_ds, data_list_ds = sort_filepaths(main_file_path, img_list_ds, data_list_ds)


# Splits data into train/test sets for both image and ring data
def data_splitter(material_imgs, materials_data, split_ratio=0.2):
    # Select a random list of indices to use as test set
    split_num = split_ratio * float(len(material_imgs))
    split_num = int(math.ceil(split_num))
    # Generate a list of random indices to use for testing
    np.random.seed(0)
    random_indices = random.sample(range(0, len(material_imgs)), split_num)
    random_index_list = np.sort(
        random_indices)  # sort indices , but also shows doesn't sample with replacement - so each index is unique
    # now have random list of file path indices, get corresponding file paths!
    # Populate new test arrays by taking the random indices of the full dataset
    test_imgs_filepaths = np.take(material_imgs, random_index_list)
    test_data_filepaths = np.take(materials_data, random_index_list)
    # Delete the random 20% taken from the dataset - this is the training set
    train_imgs_filepaths = np.delete(material_imgs, random_index_list)
    train_data_filepaths = np.delete(materials_data, random_index_list)

    return test_imgs_filepaths, test_data_filepaths, train_imgs_filepaths, train_data_filepaths


# Splits the set of filepaths of images/data to train/test datasets
test_imgs_filepaths, test_data_filepaths, train_imgs_filepaths, train_data_filepaths = data_splitter(img_list_ds,
                                                                                                     data_list_ds,
                                                                                                     split_ratio=0.2)
max_rings = 3  # fill for dataset

img_height = 256
img_width = 256
ring_in_img_train = []
ring_in_img_test = []
img_train = np.zeros(((len(train_imgs_filepaths)), img_height, img_width, 1))
img_test = np.zeros((len(test_imgs_filepaths), img_height, img_width, 1))


# For looking at img filenames
# x = 19
# split_str = os.path.split(img_list_ds[x])
# split_str = split_str[1].split(".")[0]
# print(split_str)

# Helper functions to make your feature definition more readable
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# initialize lists to hold data + padding zeros for data
data_train = []
data_test = []
padding_2d = np.reshape(np.zeros(5), (1, 5))


# Pads data if they do not have 3 rings.
# Takes in image information, # of rings, padding array (1D array of length 5)
# Returns padded version of image information
def data_padder(intermediate_data, max_rings, padding_2d):
    padding_2d = np.reshape(np.zeros(5), (1, 5))  # this is padding array used for this length 5 dataset
    if len(intermediate_data) != max_rings:
        for x in range(0, (max_rings - len(intermediate_data))):
            intermediate_data = np.append(intermediate_data, padding_2d, axis=0)
    return intermediate_data


# Loads in the reshaped images and datasets of each image of each file name/url, pads data if needed, and loads data into array
# Takes in list of image/dataset urls/filepaths, initialized list to store ring number and images/corresponding datasets, max amount of rings, padding array, and image heigh/width to reshape
# Returns array of images, datasets, and # of rings in each image
def data_aggregator(img_files, data_files, ring_in_img, max_rings, padding_2d, img_load, data_load):
    for x in range(0, len(img_files)):
        data_pd = pd.read_csv(str(data_files[x]), encoding='utf-8')
        # Convert to ndarray
        intermediate_data = data_pd[
            ['X-center', 'Y-center', 'Major Axis', 'Minor Axis', 'Alpha Rotation Angle']].to_numpy()
        ring_in_img.append(len(intermediate_data))
        # convert variable dimension ndarray to list (due to dif # of rings)
        intermediate_data = data_padder(intermediate_data, max_rings, padding_2d)
        data_load_list = intermediate_data.tolist()
        data_load.append(data_load_list)
    ring_in_img = np.array(ring_in_img)
    data_load = np.array(data_load)
    return img_load, data_load, ring_in_img


# scales data to obtain mean of 0, unit variance of 1 (STD 1)
# Takes in data (array) to scale, initialized mean/std variables
# Returns the scaled array data along with data specific mean and std
def scaler(arr, mean, std):
    arr_flat = arr.flatten()
    arr_flat = arr_flat[arr_flat != 0]
    mean = np.mean(arr_flat, axis=0)
    std = np.std(arr_flat, axis=0)
    arr = (arr - mean) / std
    arr = np.array(arr)
    return arr, mean, std


# Undos scaler function
# Takes in scaled data (array) and dataset specific mean/std values
# Returns unscaled data
def unscaler(arr, mean, std):
    arr = (arr * std) + mean
    return arr


# Helper function for Scaler Function: honestly could get rid of this - pulls dupliate duty as original scaler function
# Can just feed 3 variables in, would load out correctly in data type
def data_standardizer(dataset, data_mean, data_std):
    standardized_data = scaler(dataset, data_mean, data_std)
    data_mean = standardized_data[1]
    data_std = standardized_data[2]
    standardized_data = np.array(standardized_data[0])
    return standardized_data, data_mean, data_std


# Takes the test or training dataset and extracts ring characteristics of interest into separate list entries
# Takes in unprocessed dataset (list), initialized list, and "splits" clumped dataset into more easily manageable separate entries of ring characteristics
# Returns list with separated ring characteristics for each list index
def data_loader(full_dataset, extracted_dataset):
    i = 0
    full_dataset_t = full_dataset.transpose()
    for x in enumerate(full_dataset_t):
        extracted_dataset.append(full_dataset_t[i].transpose())
        i += 1
    return extracted_dataset


# Takes list of all extracted datasets and standardizes them all, returns dataset specific means/stds
# Takes in list of separated ring characteristics, 3 initialized lists (dataset, means, stds) to hold data once standardized
# Returns list of standardized data for separated ring characteristics, as well as lists for their dataset specific means/stds
def loop_data_standardizer(extracted_dataset, processed_dataset, data_means, data_stds):
    i = 0
    for x in enumerate(extracted_dataset):
        standardized_data = []
        data_mean = 0.0
        data_std = 0.0
        standardized_data, data_mean, data_std = data_standardizer(extracted_dataset[i], data_mean, data_std)
        processed_dataset.append(standardized_data)
        data_means.append(data_mean)
        data_stds.append(data_std)
        i += 1
    return processed_dataset, data_means, data_stds


img_test, data_test, ring_in_img_test = data_aggregator(test_imgs_filepaths, test_data_filepaths, ring_in_img_test,
                                                        max_rings, padding_2d, img_test, data_test)
img_train, data_train, ring_in_img_train = data_aggregator(train_imgs_filepaths, train_data_filepaths,
                                                           ring_in_img_train, max_rings, padding_2d, img_train,
                                                           data_train)
img_train, img_test = img_train / 255.0, img_test / 255.0

# Training Dataset
extracted_train_dataset = []
extracted_train_dataset = data_loader(data_train, extracted_train_dataset)

# Standardizing Training Dataset
processed_train_dataset = []
data_train_means = []
data_train_stds = []
processed_train_dataset, data_train_means, data_train_stds = loop_data_standardizer(extracted_train_dataset,
                                                                                    processed_train_dataset,
                                                                                    data_train_means, data_train_stds)

# Testing Dataset
extracted_test_dataset = []
extracted_test_dataset = data_loader(data_test, extracted_test_dataset)

# Standardizing Testing Dataset
processed_test_dataset = []
data_test_means = []
data_test_stds = []
processed_test_dataset, data_test_means, data_test_stds = loop_data_standardizer(extracted_test_dataset,
                                                                                 processed_test_dataset,
                                                                                 data_test_means, data_test_stds)

# For updated ALL DATASET TRAINING/TESTING
# 0 =Xc, 1 = Yc, 2 = MajorAx, 3 = MinorAx, 4 = AlphaAngle
chosen_dset = 2

dataset_test = processed_test_dataset[chosen_dset]
dataset_test_mean = data_test_means[chosen_dset]
dataset_test_std = data_test_stds[chosen_dset]

dataset_train = processed_train_dataset[chosen_dset]
dataset_train_mean = data_train_means[chosen_dset]
dataset_train_std = data_train_stds[chosen_dset]

# Load in size of dataset, as well as dataset mean and std to process later
file_path_test = 'test_' + str(int(len(data_test))) + '_' + str(data_test_means[chosen_dset]) + '_' + str(
    data_test_stds[chosen_dset]) + '_' + "major_ax.tfrecord"
file_path_train = 'train_' + str(len(data_train)) + '_' + str(data_train_means[chosen_dset]) + '_' + str(
    data_train_stds[chosen_dset]) + '_' + "major_ax.tfrecord"


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


# Takes in specific set of images and image data to convert to image-label paired tfrecord
def tf_records_converter(img_list_ds, img_load, img_height, img_width, filepath, dataset):
    file_path = filepath
    writer = tf.io.TFRecordWriter(str(file_path))
    for z in range(0, len(img_list_ds)):
        img_load[z] = np.asarray(losa.load_image(str(img_list_ds[z])), dtype=np.float32).reshape(img_height, img_width,
                                                                                                 1)
        feature = {'image': _bytes_feature(tf.compat.as_bytes(img_load[z].tobytes())),
                   'label1': _float_feature(float(dataset[z][0])),
                   'label2': _float_feature(float(dataset[z][1])),
                   'label3': _float_feature(float(dataset[z][2]))
                   # Will need to change to accomodate for eventual 3+ ring images
                   }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())


tf_records_converter(test_imgs_filepaths, img_test, img_height, img_width, file_path_test, dataset_test)
tf_records_converter(train_imgs_filepaths, img_train, img_height, img_width, file_path_train, dataset_train)

print("Finished outputting desired dataset labels and images to tfrecords")
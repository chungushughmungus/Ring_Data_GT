import tensorflow as tf
import keras
from keras import datasets, layers, models, utils
import numpy as np
import algotom.io.loadersaver as losa
import matplotlib.pyplot as plt
import pandas as pd
import glob
import random
import math

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

# major axes of each 256x256 is 10 pixels apart at minimum since they blur together

# This is the combined dataset
# 1st feed in data, plot to make sure coordinate systems look good (origin loaded in should match origin loaded out + how it's processed in here)
main_file_path = "your file path"
material_imgs = sorted(list(glob.glob(main_file_path + "/*.tif")))[:5000]
materials_data = sorted(list(glob.glob(main_file_path + "/*.csv")))[:5000]


# Loads abt 13 MB in (Just files)


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
    # Populate new ndarray by taking the random indices of the full dataset
    test_imgs = np.take(material_imgs, random_index_list)
    test_data = np.take(materials_data, random_index_list)
    # Delete the 20% taken from the dataset - this is the training set
    train_imgs = np.delete(material_imgs, random_index_list)
    train_data = np.delete(materials_data, random_index_list)

    return test_imgs, test_data, train_imgs, train_data


test_imgs, test_data, train_imgs, train_data = data_splitter(material_imgs, materials_data, split_ratio=0.2)
# Loads about 46 MB in for splitting filepaths

# File paths in, now structure loading in data
seed = 42  # keep randomization down
np.random.seed = seed
max_rings = 3  # This is the max amount of rings in an image
IMG_WIDTH = 256
IMG_HEIGHT = 256

# Create 4D Ndarray for img storage - # imgs, width/height
img_train = np.zeros((len(train_imgs), IMG_HEIGHT, IMG_WIDTH, 1))
img_test = np.zeros((len(test_imgs), IMG_HEIGHT, IMG_WIDTH, 1))
# Assigns 15.7GB ALONE HERE for 30k imgs!

# check model ,input/output sizes per epoch to see which one's growing
# use pycharm for documentation for libraries, for

# initialize lists to hold data + padding zeros for data
data_train = []
data_test = []
padding_2d = np.reshape(np.zeros(5), (1, 5))
print(padding_2d)


# Pads data if they do not have 3 rings.
# Takes in image information, # of rings, padding array (1D array of length 5)
# Returns padded version of image information
def data_padder(intermediate_data, max_rings, padding_2d):
    padding_2d = np.reshape(np.zeros(5), (1, 5))  # this is padding array used for this length 5 dataset
    if len(intermediate_data) != max_rings:
        for x in range(0, (max_rings - len(intermediate_data))):
            intermediate_data = np.append(intermediate_data, padding_2d, axis=0)
    return intermediate_data


# initialize lists for ring # per image
ring_in_img_train = []
ring_in_img_test = []


# Loads in the reshaped images and datasets of each image of each file name/url, pads data if needed, and loads data into array
# Takes in list of image/dataset urls/filepaths, initialized list to store ring number and images/corresponding datasets, max amount of rings, padding array, and image heigh/width to reshape
# Returns array of images, datasets, and # of rings in each image
def data_aggregator(img_files, data_files, ring_in_img, max_rings, padding_2d, img_load, data_load, IMG_HEIGHT,
                    IMG_WIDTH):
    for x in range(0, len(img_files)):
        img_load[x] = np.asarray(losa.load_image(str(img_files[x])), dtype=np.float32).reshape(IMG_HEIGHT, IMG_WIDTH, 1)
        # represent data as ndarrays, pass in relevant csv file data - since encoded into csv as utf-16, decode back out as utf-16 (pandas weird)
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


img_test, data_test, ring_in_img_test = data_aggregator(test_imgs, test_data, ring_in_img_test, max_rings, padding_2d,
                                                        img_test, data_test, IMG_HEIGHT, IMG_WIDTH)
img_train, data_train, ring_in_img = data_aggregator(train_imgs, train_data, ring_in_img_train, max_rings, padding_2d,
                                                     img_train, data_train, IMG_HEIGHT, IMG_WIDTH)


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
    print("Mean: %f and Standard Deviation: %f " % (mean, std))
    return arr, mean, std


# Normalizes data; takes min and max of dataset to squeeze values from 0 to 1
# Takes in data (array), and initialized variables to hold min/max dataset values
# Returns normalized data and dataset specific min and max values
def normalizer(arr, min, max):  # Get data between 0 and 1
    arr_flat = arr.flatten()
    arr_flat = arr_flat[arr_flat != 0]
    min = arr_flat[np.argmin(arr_flat)]
    max = arr_flat[np.argmax(arr_flat)]
    arr = (arr - min) / (max - min)
    arr = np.array(arr)
    print("Minimum: %f and Maximum: %f" % (min, max))
    return arr, min, max


# Undos normalizer function
# Takes in normalized data (array) and dataset specific min/max values
# Returns unnormalized data
def unnormalizer(arr, min, max):
    arr = (arr) * (max - min) + min
    arr = np.array(arr)
    return arr


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


img_train, img_test = img_train / 255.0, img_test / 255.0

# Section to process data

# Training Dataset
extracted_train_dataset = []
extracted_train_dataset = data_loader(data_train, extracted_train_dataset)
print(extracted_train_dataset[2][12])  # 0 =x center, 1 = Y center
print(len(extracted_train_dataset))
print(len(data_train))

# Standardizing Training Dataset
processed_train_dataset = []
data_train_means = []
data_train_stds = []
processed_train_dataset, data_train_means, data_train_stds = loop_data_standardizer(extracted_train_dataset,
                                                                                    processed_train_dataset,
                                                                                    data_train_means, data_train_stds)
print(processed_train_dataset[2][12])
print(len(processed_train_dataset[0]))
print(len(data_train_means))

# Testing Dataset
extracted_test_dataset = []
extracted_test_dataset = data_loader(data_test, extracted_test_dataset)
print(extracted_test_dataset[0][120])  # 0 =x center

# Standardizing Testing Dataset
processed_test_dataset = []
data_test_means = []
data_test_stds = []
processed_test_dataset, data_test_means, data_test_stds = loop_data_standardizer(extracted_test_dataset,
                                                                                 processed_test_dataset,
                                                                                 data_test_means, data_test_stds)
print(len(processed_test_dataset[4]))
print(len(data_test_means))

# This is list of all datasets, standardized (5 total, 0=Xcenter, 1 = Ycenter, 2= MajorAx, 3= MinorAx, 4 = AlphaAngle)

# Model - Sequential Version
model = tf.keras.models.Sequential([
    layers.Conv2D(filters=16, kernel_size=(15, 15), strides=1, padding="same", activation='relu',
                  input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    layers.Conv2D(filters=32, kernel_size=(15, 15), strides=1, padding="same", activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation="linear")  # for ring axis testing
    # layers.Dense(4, activation ="softmax") # for ring quantity testing
])
model.summary()

# test = utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# List to hold files for the weights for each dataset trained on model (so far only 2 since others were on old model)
# This Y-center weight set has been overwritten - NO LONGER USABLE WITHOUT TRAINING
weights = ["your file paths"
           ]

best_model = keras.callbacks.ModelCheckpoint(weights[5])

# For updated ALL DATASET TRAINING/TESTING
# 0 =Xc, 1 = Yc, 2 = MajorAx, 3 = MinorAx, 4 = AlphaAngle
s = 1  # Var to select which dataset to train - still same even if using noisy data

# Training Dataset
model_train_data = processed_train_dataset[s]
data_train_mean = data_train_means[s]
data_train_std = data_train_stds[s]

# Testing Dataset
model_test_data = processed_test_dataset[s]
data_test_mean = data_test_means[s]
data_test_std = data_test_stds[s]

# # For classification work:
# model_train_data = np.array(ring_in_img_train)
# model_test_data = ring_in_img_test

# Optimizer - for quick change in optimizer function and learning rate
Opti = keras.optimizers.Adam(learning_rate=0.000005)  # 10^-5 optimal for quite a bit but 0.5x10^-6 currently

# Compile model with loss metrics and optimizer
model.compile(optimizer=Opti,
              loss=["mse"],  # tf.keras.losses.SparseCategoricalCrossentropy,
              # loss_weights = {"Ring_Pred": 1.0, "Major_Ax_Pred": 1.0},
              metrics=["mae"])  # "root_mean_squared_error"])#tf.keras.metrics.SparseCategoricalAccuracy()])

# # For Ring classification
# model.compile(optimizer=Opti,
#                 loss=["SparseCategoricalCrossentropy"],#tf.keras.losses.SparseCategoricalCrossentropy,
#                 metrics=['sparse_categorical_accuracy'])
#                 # loss= [tf.keras.losses.SparseCategoricalCrossentropy],
#                 # metrics= [tf.keras.metrics.SparseCategoricalAccuracy()])

# don't play with batch size right now since small dataset (1k imgs per label is not much given the large amt of parameters)
history = model.fit(img_train,
                    # ring_in_img, # for classification
                    # epochs=60, # for ring classification
                    model_train_data,
                    epochs=400,
                    # validation_data =[img_test, ring_in_img_test],# for classification
                    validation_data=[img_test, model_test_data],
                    callbacks=best_model
                    )

# Total RAM use seems to be about 16GB for image data allocation and then another 12-14GB for training memory storage

# Seems minor easier than major axis training - 100 still not in e-04 yet even with padding and adjusted lr

# blows up on ws-1 - use ws3
print("Best model epoch had val MSE loss of: %f and a val MAE of : %f" % (
min(history.history['val_loss']), min(history.history['val_mae'])))

# For 200E MajorAx: Best model epoch had val MSE loss of: 0.001140 and a val MAE of : 0.018468
# 400E MajorAx: Best model epoch had val MSE loss of: 0.000841 and a val MAE of : 0.013374 8/14 - need to verify
# 400E Y center: best model epoch had val MSE loss of: 0.156141 and a val MAE of : 0.267095 8/14 - need to verify
# 400E Alpha Angles: best model epoch had val MSE loss of: 0.008006 and val MAE of: 0.060736
# 310E Minor Axis: loss: 2.9878e-04 - mae: 0.0113 - val_loss: 7.1638e-04 - val_mae: 0.0108
# Alpha Ring #s 99.97% classification accuracy (Weights[4]) - need to verify
# DO 1 BIG VERIFIY TMR! FOR ALL DSETS!

# For 5k imgs, 0,3 Noise, 100E major axis: Best model epoch had val MSE loss of: 0.047745 and a val MAE of : 0.107351
# Alpha angles version of noise: Best model epoch had val MSE loss of: 0.110973 and a val MAE of : 0.216249
# 400E: for alpha angles - noised: Best model epoch had val MSE loss of: 0.082603 and a val MAE of : 0.174519


# Load in model weights from best epoch, val-loss wise, to evaluate with - Holds 800E 15x15 RRRL
model.load_weights(weights[6])

# Training/Validation Loss Chart to plot progress in training
plt.plot(history.history['loss'], label='mse')
plt.plot(history.history['val_loss'], label='val_mse')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error Loss')
plt.ylim([0.00001, 0.9])
plt.legend(loc='lower right')
plt.show()

# Best model epoch had val MSE loss of: 0.000662 and a val MAE of : 0.010955 - 400E 5 layers, 2 5x5 filters, MajorAx
# val MSE loss of: 0.000616 and a val MAE of : 0.011931 for Major Ax 400E Alpha rings
# basically with 3 5x5 layers or 1 15x15 it's same MSE/MAE validation - not much change

# Yc stats:

test_loss, test_acc = model.evaluate(img_test, model_test_data, verbose=2)
test = model.predict(img_test)
print(type(test))
print(test.shape)


# Load in specific weights for dataset and evaluate against test dataset
def load_and_evaluate(model_weights, img_test, model_test_data, test):
    model.load_weights(model_weights)
    test_loss, test_acc = model.evaluate(img_test, model_test_data, verbose=2)
    test = model.predict(img_test)
    return test, test_loss, test_acc


# unscales data to see how model performed
test = unscaler(test, data_test_mean, data_test_std)
model_test_data = unscaler(model_test_data, data_test_mean, data_test_std)
model_train_data = unscaler(model_train_data, data_train_mean, data_train_std)

tax = (float(min(history.history['val_loss'])))
tax_root = np.sqrt(tax) * data_test_std
print(tax_root)

print("This is the MSE converted into coordinates (I think): ", tax)


# Getting very close! I think I should write custom loss function (large incorrect values in individual differences would be weighted 0.4 and punished more)

# add functionality to load in single image, and then have model evaluate - return output printed
# plots 16 imgs and their P/GTs for each dataset desired, index range as well
def mad_plotter(index1, index2, test_imgs, test, model_test_data):
    # Loads plot for 16 rings/ground truth values and model predicted values
    plt.figure(figsize=(15, 15))
    plt.title("Test Data:")
    ax = plt.figure().gca()
    ax.get_yaxis().set_visible(False)
    p = 0
    for i in range(index1, index2):
        plt.subplot(4, 4, p + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(test_imgs[i], cmap='gray')
        plt.xlabel("P: " + str(test[i]) + "\n  GT: " + str(model_test_data[i]))  # for coordinate regression
        # plt.xlabel("P: " + str(np.argmax(test[i])) + "\n  GT: " + str(ring_in_img_test[i])) # for ring classification
        plt.subplots_adjust(wspace=1.6, hspace=0.5)
        p += 1
    plt.show()


mad_plotter(69, 85, img_test, test, model_test_data)

# Appears that when noise is significant relative to ring intensity, model tends to underestimate. When noise is lower, to non-significatn lvl,
# model tends to overestimate (from 400E noised 0.3 alpha angles testing of 5k imgs)


# For On-Demand Evaluation
# For On-Demand Evaluation


# Testing Dataset
# 0 =Xc, 1 = Yc, 2 = MajorAx, 3 = MinorAx, 4 = AlphaAngle
s = 2
# Training Dataset
model_train_data = processed_train_dataset[s]
data_train_mean = data_train_means[s]
data_train_std = data_train_stds[s]

# Testing Dataset
model_test_data = processed_test_dataset[s]
data_test_mean = data_test_means[s]
data_test_std = data_test_stds[s]

print(len(model_train_data))
print(len(model_test_data))

# # For classification work:
# model_train_data = np.array(ring_in_img_train)
# model_test_data = np.array(ring_in_img_test)

test = []

# For weights:
# 0 = Yc, 1 = MajorAx, 2 = MinorAx, 3 = AlphaAngle, 4 = Ring #
test, test_loss, test_acc = load_and_evaluate(weights[1], img_test, model_test_data, test)

# reverts data to see how model performed
test = unscaler(test, data_test_mean, data_test_std)
model_test_data = unscaler(model_test_data, data_test_mean, data_test_std)
model_train_data = unscaler(model_train_data, data_train_mean, data_train_std)

mad_plotter(0, 16, img_test, test, model_test_data)

import tensorflow as tf

tf.keras.backend.clear_session()

# For On-Demand Evaluation
# For On-Demand Evaluation


# To dump gpu memory after a run - seems like gpu stores prior model weights in memory even with killed/restarted kernels
import numba

gpus = numba.cuda.list_devices()
print(gpus)

gpu_use = numba.cuda.get_current_device()
print(gpu_use)
import gc

gc.collect()

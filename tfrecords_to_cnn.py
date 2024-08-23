import numpy as np
import tensorflow as tf
import keras
from keras import layers, models, utils
import matplotlib.pyplot as plt
import data_to_tfrecords as processor

img_height = 256
img_width = 256

# Note: Turns out I formatted the entire setup incorrectly:
# TFRecords streams in PER FILE! Each image should technically be a tfrecord on its own, not in an aggregate!
# I completely screwed up the format. RIP.
# THIS ONLY PREVENTS RAM FROM GROWING - no memory leaks (it grows a bit every time it saves model weights though but drops back down)

# THESE ARE THE TFRECORDS TO LOAD IN!!! - choose the processed files/datasets you want to load in
# # 5k image dataset here
file_path_train = "your file path"
file_path_test = "your file path"


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
main_file_path = "main file path to folder"


# CNN File (load in + create dataset + cnn)
def get_image_and_label(features):
    image, label = features['image'], features['label1', 'label2', 'label3']
    return image, label


# Parses the tf record to find image values and labels
def _parse_function(proto):
    keys_to_features = {'image': tf.io.FixedLenFeature([], tf.string),
                        'label1': tf.io.FixedLenFeature([], tf.float32),
                        'label2': tf.io.FixedLenFeature([], tf.float32),
                        'label3': tf.io.FixedLenFeature([], tf.float32)
                        }
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    # Decodes the bytestring formatted image data into float 64 values
    parsed_features['image'] = tf.io.decode_raw(parsed_features['image'], tf.float64)
    label = [parsed_features['label1'], parsed_features['label2'], parsed_features['label3']]
    return parsed_features['image'], label


# Prior operations used 32/batch (24k/750 steps)
NUM_CLASSES = 4
epochs = 200


# because 1 batch per epoch right now - it fits within size (so 1 batch in 1 epoch)
def create_dataset(filepath, BATCH_SIZE):
    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filepath)
    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(_parse_function, num_parallel_calls=8)
    # Set the number of datapoints you want to load and shuffle
    dataset = dataset.shuffle(buffer_size=320, seed=None).batch(BATCH_SIZE)
    # This dataset will go on for # of epochs specified
    dataset = dataset.repeat(epochs)
    # # Set the batchsize
    # dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    # Create an iterator
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    # Create your tf representation of the iterator
    image, label = iterator.get_next()
    # Bring your picture back in shape
    image = tf.reshape(image, [-1, 256, 256, 1])
    # Create a one hot array for your labels - for ring classification
    # label = tf.one_hot(label, NUM_CLASSES)
    return image, label


# Split tf record file name to obtain size, mean, and std
str_test = file_path_test.split('_')
str_train = file_path_train.split('_')

# 0 is filepath, 1 is # imgs in dataset, 2 is mean, 3 is std
test_size = int(str_test[1])
test_mean = float(str_test[2])
test_std = float(str_test[3])
print(test_size, test_mean, test_std)

train_size = int(str_train[1])
train_mean = float(str_train[2])
train_std = float(str_train[3])

# Keep full dataset in 1 batch to use all - buffer in 320 images per shuffle since each step used by tensorflow is 32
test_image, test_label = create_dataset(file_path_test, BATCH_SIZE=test_size)
train_image, train_label = create_dataset(file_path_train, BATCH_SIZE=int(train_size / 5))

# File paths in, now structure loading in data
seed = 42  # keep randomization down
np.random.seed = seed

# # Plots filestream-batch read in image! (Batch size is 32 so up to index 31 here)
# d=12
# plt.imshow(test_image[d])
# plt.xlabel("L: " + str(test_label[d])) #unordered so ring_in_img doesn't match up with label/img pairs
# plt.show()

# Model - Sequential Version
model = tf.keras.models.Sequential([
    layers.Input(shape=(img_height, img_width, 1)),  # input_shape=(img_height, img_width, 1)),
    layers.Conv2D(filters=16, kernel_size=(15, 15), strides=1, padding="same", activation='relu'),
    # , input_shape=(img_height, img_width, 1)),#kernel_initializer= 'he_normal', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)))
    layers.Conv2D(filters=32, kernel_size=(15, 15), strides=1, padding="same", activation='relu'),
    # layers.LeakyReLU(alpha=0.2),kernel_regularizer=keras.regularizers.l2(l2=0.1)))
    # extra layer for x,y center coordinate regression not v good
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation="linear")  # for ring axis testing
    # layers.Dense(NUM_CLASSES, activation ="softmax") # for ring quantity testing
])

# output shapes of label and image are dif - need them to line up, so get on it@(255 and 4,)
model.summary()

weight_path = "path to store weights"
best_model = keras.callbacks.ModelCheckpoint(weight_path)

# Optimizer - for quick change in optimizer function and learning rate
Opti = keras.optimizers.Adam(
    learning_rate=0.000005)  # adjust lr if it's bouncing - so far not - less/more would so 10^-5 optimal currently
# use loss function specific accuracy metrics - here is sparse categorical

# Compile model with loss metrics and optimizer
model.compile(optimizer=Opti,
              loss=["mse"],  # tf.keras.losses.SparseCategoricalCrossentropy,
              # loss_weights = {"Ring_Pred": 1.0, "Major_Ax_Pred": 1.0},
              metrics=["mae"])  # "root_mean_squared_error"])#tf.keras.metrics.SparseCategoricalAccuracy()])

# # For Ring classification
# model.compile(optimizer=Opti,
#                 loss=["SparseCategoricalCrossentropy"],#tf.keras.losses.SparseCategoricalCrossentropy,
#                 metrics=['sparse_categorical_accuracy']
#                 )

history = model.fit(train_image,
                    # ring_in_img, # for classification
                    # epochs=60, # for ring classification
                    train_label,
                    epochs=epochs,
                    # validation_data =[img_test, ring_in_img_test],# for classification
                    validation_data=[test_image, test_label],
                    callbacks=best_model
                    )
# Loads in best weights to evaluate test set on
model.load_weights(weight_path)

test_loss, test_acc = model.evaluate(test_image, test_label, verbose=2)
test = model.predict(test_image)

# unscales data to see how model performed
test = processor.unscaler(test, test_mean, test_std)
test_label = processor.unscaler(test_label, test_mean, test_std)

# Plots 15 images and their predictions and ground truths
plt.figure(figsize=(15, 15))
plt.title("Test Data:")
ax = plt.figure().gca()
ax.get_yaxis().set_visible(False)
p = 0
for i in range(0, 15):
    plt.subplot(4, 4, p + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_image[i], cmap='gray')
    plt.xlabel("GT: " + str(test_label[i]) + "\n P: " + str(test[i]))  # for coordinate regression, makes it less messy
    # plt.xlabel("P: " + str(np.argmax(test[i])) + "\n  GT: " + str(test_label[i])) # for ring classification
    plt.subplots_adjust(wspace=1.6, hspace=0.5)
    p += 1
plt.show()

# Clear out tensorflow gpu ram
tf.keras.backend.clear_session()

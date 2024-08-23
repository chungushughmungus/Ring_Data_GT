import os
import numpy as np
import sim_lib1 as sim
import algotom.io.loadersaver as losa
import matplotlib.pyplot as plt
import random
import pandas as pd
import random_str
import itertools
from itertools import combinations, chain
import math

# have function that can pull up random img, eval + print out results - should be simple. Load weights in, etc.
# function to generate data - pass in values, or leave blank to use default.
# probably have another to specify default path - set to where this program runs + generate new folder

# Directories to pull/load files from/to

# for combined testing
output_base = "/nsls2/data/hex/proposals/2024-2/pass-316730/Frank_data/Perkin_Elmer_Test_Imgs/Testing_Alpha_Ring_Images"  # NOT GENERATING ON ACTUAL USE DATASET!

# Center Values - You can change these to change ring centers
x0 = 128
y0 = 128

# rings you want
rings_wanted = 1

# Alpha angle - desired detector tilt
alpha_angles = np.array([10, 20, 30, 40, 45])  # generated these sets in both 1,2,3 and combined folders (2k imgs each)

# Something broke and I don't have time to debug why it's limited to 250+ images
# Here choose minimum 250 random images - not all possible combinations - tho can do that - just input len(combos) or something instead of img_quantity
img_quantity = 250  # how many images + datafiles you want to generate

# Add noise to image
noise_level = 0.0

# Detector Pixel resolution
height = 256
width = 256
pixel_size = 0.2 * (2048 / height)  # mm resolution of each pixel

energy = 60.0  # keV of beam

# Beam Center Coordinates:
Xt = width / 2
Yt = height / 2


def xy_shift_distance(x, y, Xt, Yt):
    shifted_distance = math.sqrt((abs(x0 - Xt)) ** 2 + (abs(y0 - Yt)) ** 2)
    return shifted_distance


shifted_distance = xy_shift_distance(x0, y0, Xt, Yt)
print("Distance from center", shifted_distance)
# Multiply by d-spacing correlation to each pixel
ring_resizing = shifted_distance * 0.00105  # extra 05 behind to account for extreme edge cases for rings near edges

# Get ring size limit with resizing
ring_size_limit = 0.0850 + ring_resizing
print("Ring Size Limit:", ring_size_limit)

# Smaller d spacing means larger ring
ring_size_center = 1  # upper bound of "d-spacing" for ring - small center ring
ring_step_size = 0.0022  # just 2.2 pixels shift per ring - probably more overlap in rings closer to the center, less towards edges but still shift

# Choose subset to sample from combinatorics (to evade combinatorical explosion) and determine if combo valid/not
subset_size = 2500000

# generate set of intensities to randomly sample from; d_spacing to cover/generate at minimum 400 unique rings
intensities_set = np.arange(6, 255, 1)

# Normalizes intensities to 0-1
intensities_set = intensities_set / 255
# This sets max bounds via size limit. I will probably have to do a check to see if coordinate shift or angle shift is greater - then use that to plug in
random_d_spacing = np.arange(ring_size_limit, ring_size_center, ring_step_size)

print(ring_size_limit)
print(ring_size_center)

# Equation to cut off rings from consideration via d-spacing
angle_cutoff = 0.00065 * abs(alpha_angles) + 0.08

# If center coordinates are changed: use to determine which cutoff is more strict - altered xy center or alpha angle for ring cutoffs
if x0 != 128 or y0 != 128:
    angle_cutoff = np.max(angle_cutoff, ring_size_limit)

# look at indices to remove based on ring cutoffs from differing alpha angles
i = 0
cutoff_indices = []
for x in enumerate(alpha_angles):
    cutoff_index = np.argmax(random_d_spacing > angle_cutoff[i])
    cutoff_indices.append(cutoff_index)
    i += 1

print(cutoff_indices)

if len(random_d_spacing) == 0:
    print(
        "Sorry, no valid rings exist for this coordinate ring set. Please change the X,Y center coordinates and try again.")
    exit()

# splice array for each angle used.
cutoff_modified_d_spacing = []

# get spliced d_spacing values into separate arr, store all in jagged list
a = 0
for x in enumerate(cutoff_indices):
    splice = random_d_spacing[cutoff_indices[a]:]
    cutoff_modified_d_spacing.append(splice)
    a += 1

# Perkin Elmer detector info
# Keep 1 distance, no tilt for testing first - looks good v original 800mm w tilt
list_distance = np.asarray([800])
distance_used = np.asarray(list_distance[0])  # mm
# list_alpha = np.asarray([50]) # Degree - detector pitch
list_beta = np.asarray([0.0])  # Degree - detector roll

# Gen Random String for ID - each img/csv data file
data_id = random_str.random_string(4)

# Initialize lists to store angle cutoff values
b = 0  # counter var for alpha angle for loop
angle_cutoff_results = []  # for angles but kinda don't need it
angle_cutoff_coefs = []  # dump coefs for each angle/ring combo
m_a = []  # list of all the major axes of the angle rings used

print(type(cutoff_modified_d_spacing[b]))
# run a big ol for loop w angles as 1st, then inside is d_spacing to get da values

# Get angle coefs and major axes for each ring that is to be processed later
for x in enumerate(alpha_angles):
    cutoff_results = []
    angles = []
    m_a_1 = []
    for i, dspace in enumerate(np.array(cutoff_modified_d_spacing[b])):
        theta = sim.energy_kev_to_theta(energy, dspace * 10.0)  # d space multiply by 10 why?
        # Each img - not each ring - has different detector tilt (alpha, beta) in list_alpha/beta
        coefs = sim.calc_ellipse_coefficients(x0, y0, alpha_angles[b],
                                              0, theta,
                                              800 / pixel_size)
        # print(alpha_angles[b])
        results = sim.get_ellipse_parameters(coefs)
        cutoff_results.append(
            coefs)  # dump each ring coefficient into corresponding index, then backcalc once get combo - might not even be much faster. Memory access def but computationally drawing is intensive.
        major_axis, minor_axis, x_center, y_center, tilt_angle = results
        m_a_1.append(results[0])
        angles.append(alpha_angles[b])

    angle_cutoff_coefs.append(cutoff_results)
    angle_cutoff_results.append(
        angles)  # kinda don't really need to add angles in here, but it's fine to keep it so I have the same # of angles in each jagged arr
    m_a.append(m_a_1)
    # print(alpha_angles[b])
    b += 1


######################################################
######################################################

# for each increase in ring, add 1 extra index and add 1 to second value, double it for 3rd value, so on to gen index deletion list
def indexer(rings_wanted, t_list):
    for x in range(rings_wanted):
        d = (rings_wanted + 1) * x
        t_list.append(d)
    return (t_list)


# Selects a subset of the total combinations possible to avoid combinatorical explosion and increased processing time
def subset_selector(nd_arrs, subset_size):
    # Here select random indices from nd_arr such that will only obtain random subset of desired size
    if len(nd_arrs) > subset_size:
        deletion_subset = len(
            nd_arrs) - subset_size  # if subset_size same as nd_arrs size then 0, nothing to choose to delete out lol
        # get list of random indices to remove
        random_indices = np.random.choice(len(nd_arrs), size=deletion_subset, replace=False)
        nd_arrs = np.delete(nd_arrs, random_indices, axis=0)
    return nd_arrs


# get tuples of ring distancing combos (indices) that don't work - their distances based on major axes too close to each other
def ring_distancing(nd_arrs, major_ax, t_list, fake_indices):
    i = 0
    for x in enumerate(nd_arrs):
        if i == len(nd_arrs):
            break
        dnot = ([abs(major_ax[nd_arrs[i]] - major_ax[c]) < 10 for c in nd_arrs[i]])
        # print(arrX[nd_arrs[i]])
        # want to set up list comprehension for each combination - if any false, add that index to list to later dump out - or could only add those with True - but multiple true values so not easy
        dnot = np.delete(dnot, t_list)
        # print(dnot)
        dump_or_nah = any(dnot)
        if dump_or_nah == True:
            fake_indices.append(i)
        i += 1
    return (fake_indices)


# Filters amount of images generated - stops if more than possible and continues if possible.
def img_quantity_filter(img_quantity, nd_arrs):
    if img_quantity > len(nd_arrs):
        print("Too many images: current valid ring set is only %d rings" % len(nd_arrs))
        print("The number of images generated will be equal to the amount of valid rings available")
    # Here select random indices from nd_arr such that will only obtain x imgs
    elif img_quantity < len(nd_arrs):
        # get list of random indices to remove
        random_indices = np.random.choice(len(nd_arrs), size=(len(nd_arrs) - img_quantity), replace=False)
        nd_arrs = np.delete(nd_arrs, random_indices, axis=0)
        print("contents of random_indices:", random_indices)
    return nd_arrs


t_list = []
indexer(rings_wanted, t_list)  # this can be done 1x - since each time we are selecting for 1,2 or 3 rings
processed_usable_combos = []
final_random_intensities_list = []
i = 0

# Undergoes ring selection dependent on total number of rings in each image
for x in enumerate(angle_cutoff_coefs):
    m_a_load = np.array(m_a[
                            i])  # turn major axis list into np.array - initialize new one for each to use later since this is an intermediate processing step
    arrX = np.arange(0, len(angle_cutoff_coefs[i]),
                     1)  # gen array to see # combos possible - this will act as indices for arr
    # gen all possible z length combos for values in arrX - indices
    combos = list(itertools.combinations(arrX, rings_wanted))
    # combos = list(np.random.choice(len(arrX), size = img_quantity, replace=True))
    if rings_wanted == 1:
        doggo = []
        d = 0
        # 1 ring - use np.unique after stacking to determine which index values have duplicates
        random_intensities = np.random.choice(intensities_set, size=2 * img_quantity, replace=True)
        random_d_spacing = np.random.choice(range(0, len(angle_cutoff_coefs[i])), size=2 * img_quantity, replace=True)
        d_and_i = np.stack((random_d_spacing, random_intensities), axis=1)
        removed_dups = np.unique(d_and_i, axis=0)
        random_d_spacingx, random_intensitiesx = np.hsplit(removed_dups, 2)
        # remove shape (1k,1) -> (1k,)
        random_intensitiesx = np.squeeze(random_intensitiesx)
        combos = random_d_spacingx
        random_intensities1 = random_intensitiesx[:img_quantity]
        final_random_intensities_list.append(random_intensities1)

    # Convert list of tuples to ndarray of fixed length ndarrays (# of rings)
    nd_arrs1 = np.array(combos)
    nd_arrs1 = np.round(nd_arrs1.astype(int))
    if rings_wanted != 1:
        # if true after dumping same index comparisons then means not separated enough - take this index and remove
        fake_indices = []
        # functions are defined below - move the functions up once successful
        nd_arrs1 = subset_selector(nd_arrs1,
                                   subset_size)  # select and fill in the subset we want to use for random combo verification
        fake_indices = ring_distancing(nd_arrs1, m_a_load, t_list, fake_indices)
        # turn list to arr
        fake_indices = np.array(fake_indices)
        # If only printing 1 ring images - disregard index searches
        nd_arrs1 = np.delete(nd_arrs1, fake_indices, 0)
        nd_arrs1 = img_quantity_filter(img_quantity, nd_arrs1)
        random_intensities1 = np.random.choice(intensities_set, size=img_quantity, replace=True)
        final_random_intensities_list.append(random_intensities1)
    processed_usable_combos.append(nd_arrs1)
    i += 1
    # now just need corresponding d_spacings, which are stored above as sliced indices, and intensities, can get that also
    # store each unique nd_arr as own part in list, and then iterate thru list in below stuff


# Obtains ellipse image and loads ellipse characteristics into initialized lists
def image_and_data_generator_intermediate(combo_indices, random_intensities, iterator, ring_coefs, alpha_angles,
                                          noise_level):
    ellipse_params_img = np.zeros((height, width), dtype=np.float32)
    i = 0
    for x in enumerate(combo_indices):
        # Gen Random String for ID - each img/csv data file
        ellipse_params = sim.get_ellipse_parameters(ring_coefs[combo_indices[i]])
        major_axis, minor_axis, x_center, y_center, tilt_angle = ellipse_params
        ellipse_params_img = ellipse_params_img + sim.generate_ellipse_image(ring_coefs[combo_indices[i]], height,
                                                                             width,
                                                                             tolerance=0.02,
                                                                             size=1, max_val=random_intensities[
                iterator])  # combo_indices[i]])
        major_ax1.append(ellipse_params[0])
        minor_ax1.append(ellipse_params[1])
        Xc1.append(ellipse_params[2])
        Yc1.append(ellipse_params[3])
        # alpha_angle1.append(ellipse_params[4]) # NEED TO CHANGE THIS TO RESULT IN ACTUAL ALPHA ANGLE NOT ALPHA/BETA COMBINED ANGLE
        alpha_angle1.append(alpha_angles)
        i += 1
        # print("Noise:", noise_level)
    ellipse_params_img = ellipse_params_img + noise_level * np.random.rand(height, width)
    return ellipse_params_img


# Convert set lists of desired datapoints to ndarrays
def converter(major_ax1, minor_ax1, Xc1, Yc1, alpha_angle1):
    major_ax1 = np.asarray(major_ax1)
    minor_ax1 = np.asarray(minor_ax1)
    Xc1 = np.asarray(Xc1)
    Yc1 = np.asarray(Yc1)
    alpha_angle1 = np.asarray(alpha_angle1)


# Builds dataframe in format wanted to output as csv file.
def build_format_dataframe(ring_index1, Xc1, Yc1, major_ax1, minor_ax1, alpha_angle1, random_d_spacing,
                           random_intensities, d_spacing_iterator, intensity_iterator):
    build_dataframe = np.array(
        [ring_index1, Xc1, Yc1, major_ax1, minor_ax1, alpha_angle1, random_d_spacing[d_spacing_iterator],
         random_intensities[intensity_iterator]])  # , random_index_list+1])
    # Need to transpose da data since dataframe takes in ndarrays as row inputs normally. Want separate columns here since each represent dif. set of data
    df1 = pd.DataFrame(build_dataframe).T
    df1.columns = ["No.", "X-center", "Y-center", "Major Axis", "Minor Axis", "Alpha Rotation Angle", "d-spacing [nm]",
                   "Intensity"]  # , "Original Ring No."]
    df1["No."] = df1["No."].astype(int)
    # df["Original Ring No."] = df["Original Ring No."].astype(int)
    df1["Intensity"] = df1["Intensity"].apply(lambda x: round(x, 4))
    df1["Major Axis"] = df1["Major Axis"].apply(lambda x: round(x, 2))
    df1["Minor Axis"] = df1["Minor Axis"].apply(lambda x: round(x, 2))
    return df1


# Generate an image + csv file containing ring information after using helper functions.
def img_data_generator(nd_arrs, combo_indices, i, j, Xc1, Yc1, major_ax1, minor_ax1, alpha_angle1, random_d_spacing,
                       random_intensities, d_spacing_iterator, intensity_iterator, ring_coefs, alpha_angles,
                       noise_level):
    # print("Noise", noise_level)
    ellipse_params_img = image_and_data_generator_intermediate(combo_indices, random_intensities, j, ring_coefs,
                                                               alpha_angles, noise_level)
    data_id = random_str.random_string(4)
    name = str(j) + "_" + str(data_id) + "_" + str(rings_wanted) + "_(" + str(alpha_angle1[0]) + ")_TEST_" + str(
        height) + ".tif"
    losa.save_image(output_base + "/" + name, ellipse_params_img)

    converter(major_ax1, minor_ax1, Xc1, Yc1, alpha_angle1)

    # for indexing current ring numbers in this img
    ring_index1 = np.arange(1, len(combo_indices) + 1, 1)  # random_index_list)+1,1)
    df1 = build_format_dataframe(ring_index1, Xc1, Yc1, major_ax1, minor_ax1, alpha_angle1, random_d_spacing,
                                 random_intensities, d_spacing_iterator, intensity_iterator)

    # Check if directory exists, make if doesn't
    os.makedirs(output_base, exist_ok=True)
    # Dump dataframe to csv in specified file directory + add name, remove dataframe specific index (0,1,2...)
    # dataframe outputs some random unicode/ascii characters in csv that can't be read so encode specifically as utf-16 and use csv_to_pd to read in since other methods don't work
    df1.to_csv(output_base + "/" + name.strip(".tif") + '.csv', index=False, encoding='utf-8')


j = 0
k = 0
# Big for loop to initialize new set of lists and load in data per trial to go through all images desired.
g = 0
for x in enumerate(alpha_angles):
    random_d_spacing = cutoff_modified_d_spacing[g]
    random_intensities = final_random_intensities_list[g]
    nd_arrs = processed_usable_combos[g][:img_quantity]
    ring_coefs = angle_cutoff_coefs[g]
    alpha_angles1 = alpha_angles[g]
    j = 0
    for x in enumerate(nd_arrs):
        # run loop
        ellipse_params = []
        # Initialize lists (since write faster than arrays) for major/minor axes, X/Y center points, alpha angle
        major_ax1 = []
        minor_ax1 = []
        Xc1 = []
        Yc1 = []
        alpha_angle1 = []
        combo_indices = np.array(nd_arrs[j][:img_quantity])
        # print("Size of the combo - should be equal to # of rings desired:", len(combo_indices))
        ellipse_params_img = np.zeros((height, width), dtype=np.float32)
        i = 0
        img_data_generator(nd_arrs, combo_indices, i, j, Xc1, Yc1, major_ax1, minor_ax1, alpha_angle1, random_d_spacing,
                           random_intensities, combo_indices, combo_indices, ring_coefs, alpha_angles1, noise_level)
        j += 1
        k += 1
        print(j)
        if j == img_quantity:
            break
    g += 1

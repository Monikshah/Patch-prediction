#!/usr/bin/env python
# coding: utf-8

# ## Plot colunmer structure from natural images

# #### Import package

import glob
import os
import numpy as np
import pickle
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from skimage.filters import gabor_kernel
from skimage import io
from skimage.transform import resize
from scipy import stats
# from numpy.linalg import inv
import math
import random

# #### Gabor function
# This function plots 2D gabor filter

def genGabor(sz, omega, theta, func=np.cos, K=np.pi):
    radius = (int(sz[0]/2.0), int(sz[1]/2.0))
    [x, y] = np.meshgrid(range(-radius[0], radius[0]+1), range(-radius[1], radius[1]+1))

    x1 = x * np.cos(theta) + y * np.sin(theta)
    y1 = -x * np.sin(theta) + y * np.cos(theta)
    
    gauss = omega**2 / (4*np.pi * K**2) * np.exp(- omega**2 / (8*K**2) * ( 4 * x1**2 + y1**2))
    sinusoid = func(omega * x1) * np.exp(K**2 / 2)
    gabor = gauss * sinusoid
    return gabor

g = genGabor((100,100), 0.3, 105*(np.pi/180.0), func=np.cos) 



# #### Filter bank for 12 orientations

# Parameters for generating gabor filters
phase_offset = 90
patch_size = 21
#theta_list = np.linspace(0, 165, 12)
#theta_list = np.linspace(0, 160, 9)
theta_list = np.linspace(0, 175, 36)
print("Orientation to consider {}: {}".format(len(theta_list), theta_list))

# 2D filter bank using Gabor function
gabor_bank = []
for ii in range(0, len(theta_list)):
    g_kernel = genGabor((100,100), 0.3, theta_list[ii]*(np.pi/180.0), func=np.cos) 
    g_kernel = resize(g_kernel, (patch_size, patch_size))
    gabor_bank.append(np.array(g_kernel))

image_filter = np.concatenate(gabor_bank, axis=1)
print(image_filter.shape)
# 1D (flatten) filter bank for normalized dot product
filt_flatten = []
for i in range(len(theta_list)):
    patch = gabor_bank[i]
    patch = np.ndarray.flatten(patch)
    #patch = (patch-np.mean(patch))
    patch = patch / np.linalg.norm(patch)
    filt_flatten.append(patch)

filt_flatten = np.vstack(filt_flatten)


# #### Filter response map generated from images 
# Normalized dot product is calculated between each patch and gabor filter

def get_filter_map(imm, filt_flatten):
    f_map = np.zeros((imm.shape[0]-patch_size+1, imm.shape[1]-patch_size+1, len(gabor_bank)))
    for i in range(imm.shape[0]-patch_size+1):
        for j in range(imm.shape[1]-patch_size+1):
            patch = imm[i:i+patch_size, j:j+patch_size]
            patch = np.ndarray.flatten(patch)
            #patch = patch - np.min(patch)            ###making the feature vector positive
            patch = patch / np.linalg.norm(patch)
            for k in range(len(gabor_bank)):
                f_map[i,j,k] = np.dot(patch, filt_flatten[k])
    return f_map

imm_list = glob.glob("./NaturalImages/*.jpg")
print("Number of images: {}".format(len(imm_list)))

# Calculate feature maps for each image and save in a file
# Feature map with be of size = [ih-kh+1, iw-kw+1, 12]
# ih = image height, kh = gabor kernel height
# iw = image width, kw = gabor kernel width

#if not os.path.exists("./FeatMapsUpdatedAllpositive9/"):
#    print("Creating directory to store feature map")
#    os.makedirs("./FeatMapsUpdatedAllpositive9/")

#for kk in range(0, len(imm_list)):
#    print("Working on {}/{}".format(kk+1, len(imm_list)))
#    imm = io.imread(imm_list[kk], as_gray=True)
#    size_img = imm.shape
#    f_map = get_filter_map(imm, filt_flatten)
#    print(kk, size_img, f_map.shape)
#    fname = "./FeatMapsUpdatedAllpositive9/" + os.path.basename(imm_list[kk]).split(".")[0] + ".pkl"
#    with open(fname,'wb') as f:
#        pickle.dump(f_map, f)
#    print("")

# #### Generate neighbor connection list 
# Function get_connection_list() determines the strongly connected orientation for each neighbour

# ##### Plot neighborhood

# helper functions

def p_dump(filename, data):
    print("saving data to file: {}".format(filename))
    with open(filename,'wb') as f:
        pickle.dump(data, f)

def load_pickle(pickle_file):
    print("starting function..")
    with open(pickle_file, 'rb') as f:
        pickle_data = pickle.load(f)
    return pickle_data

'''
this function computer the correlation matrix using outer product of central patch feature (data[0:feat_num]) and 
neighbor node feature
and coputer the sum of outer product matrices for each interation
data: combined flattened features of all the nodes together
'''
def get_iterative_correlation(data):
    global summ
    outer = np.outer(data[0:feat_num], data)
    summ = np.add(summ, outer)

# ##### Generate connection strength for each radius
'''
f_map: features of an image
XY_coord: coordinates of the neighbor patches
'''
def get_connection_list(f_map, XY_coord):
    global counter
    row = random.sample(range(r, f_map.shape[0] - r), 10)
    col = random.sample(range(r, f_map.shape[1] - r), 100)
    for i in row:
        for j in col:
            vall_list = []
            c_f = f_map[i, j]
            c_f = c_f - np.min(c_f)
            c_f = c_f / np.linalg.norm(c_f)
            if np.isnan(c_f).any():
                continue
            vall_list = vall_list + list(c_f)
            for k in range(XY_coord.shape[0]):
                n_f = f_map[i + XY_coord[k, 0], j + XY_coord[k, 1]]
                n_f = n_f - np.min(n_f)
                n_f = n_f / np.linalg.norm(n_f)
                if np.isnan(n_f).any():
                    break
                vall_list = vall_list + list(n_f)
            if not np.isnan(n_f).any():
                counter = counter + 1
                get_iterative_covariance(vall_list)
'''
computes n k*k outer product matrices from summ
'''

def get_expectation(summ):
    global exp_sum, mat_cov
    for idx in range(n_node):
        mat_cov[idx,:,:] = summ[:,(idx) * feat_num: (idx+1) * feat_num]
    exp_sum = np.add(exp_sum, mat_cov)

'''
f_map: features of patches in an image
node_feat_arr : expected correlation matrix
XY_coord : coodinates of central and neighbor patches  
'''
def prediction(f_map, node_feat_arr, XY_coord):
    global N, sumError1, sumSquare1, sumError2, sumSquare2
    inverse = np.linalg.inv(node_feat_arr[0,:,:])
    row = random.sample(range(r, f_map.shape[0] - r), 10)
    col = random.sample(range(r, f_map.shape[1] - r), 100)
    for i in row:
        for j in col:
            temp1 = []
            temp2 = []
            temp3 = []
            c_f = f_map[i, j]
            c_f = c_f - np.min(c_f)
            c_f = c_f / np.linalg.norm(c_f)
            if np.isnan(c_f).any(): # doesnot compute if there is any nan in feature vector
                continue
            temp1.append(c_f)
            for k in range(XY_coord.shape[0]):
                n_f = f_map[i + XY_coord[k, 0], j + XY_coord[k, 1]]
                n_f = n_f - np.min(n_f)
                n_f = n_f / np.linalg.norm(n_f)
                if np.isnan(n_f).any():
                    break
                temp1.append(n_f)
            if not np.isnan(n_f).any():
                for l in range(len(node_feat_arr)):
                    W = np.matmul(inverse, node_feat_arr[l,:,:]) 
                    result = np.matmul(W, np.transpose(c_f))
                    result = result.tolist()
                    result = result/np.linalg.norm(result)
                    if np.linalg.norm(temp1[l]) < 1e-5:
                        break
                    error1 = np.sum(np.square(temp1[l] - result))
                    error2 = np.square(np.dot(temp1[l], result)) 
                    temp2.append(error1)
                    temp3.append(error2)
                sumError1 = sumError1 + np.array(temp2)
                sumSquare1 = sumSquare1 + (np.array(temp2))**2
                sumError2 = sumError2 + np.array(temp3)
                sumSquare2 = sumSquare2 + (np.array(temp3))**2
                N = N + 1     

# Calculate and save connection strength statistics

if not os.path.exists("./multipl_xy_36_full_1k_allpos_nomean/"):
    print("Creating directory to store winner feature node arrangement matrix for each radius")
    os.makedirs("./multipl_xy_36_full_1k_allpos_nomean/")

list_rad = [1, 2, 4, 7, 10, 13, 16, 20, 25, 30, 35, 40]
map_list = glob.glob("./FeatMapsUpdated36/*.pkl")
feat_num = 36
n_node = 73
print("Number of feature maps to process: {}".format(len(map_list)))

for r in list_rad:
    counter = 0
    summ = np.zeros([feat_num, n_node * feat_num])
    exp_sum = np.zeros(shape=(n_node, feat_num, feat_num))
    mat_cov = np.zeros(shape=(n_node, feat_num, feat_num))
    print("="*70)
    print("Working for radius = {}".format(r))
    print("="*70)
    theta = np.pi / feat_num
    n = int(2*np.pi / theta)
    X_coord = np.round(r*np.cos(np.arange(n)*theta))
    Y_coord = np.round(r*np.sin(np.arange(n)*theta))
    XY_coord = np.vstack((X_coord, Y_coord)).astype(int).T

    for kk in range(len(map_list)): 
        f_name = map_list[kk]
        f_val = f_name.split("/")[-1].split(".")[0]
        print("Computing val stack for: {}".format(f_val))
        with open(f_name,'rb') as f:
            f_map = pickle.load(f)
        if kk is 55:
            continue
        print("Working on {}/{} file: {}".format(kk+1, len(map_list), map_list[kk]))
        get_connection_list(f_map, XY_coord)
    get_expectation(summ)
    exp_sum = exp_sum / counter
    print("total patch used", counter)
    p_dump("./multipl_xy_36_full_1k_allpos_nomean/rad" + str(r) + "xy_full.pkl", exp_sum)

m1 = []
m2 = []
s1 = []
s2 = []
for r in list_rad:
    N = 0
    sumError1 = 0
    sumSquare1 = 0
    sumError2 = 0
    sumSquare2 = 0
    
    fname = "./multipl_xy_9_full_1k_allpos_nomean/rad" + str(r) + "xy_full.pkl"
    with open(fname,'rb') as f:
        node_feat_arr = pickle.load(f)

    theta = np.pi / feat_num
    n = int(2*np.pi / theta)
    X_coord = np.round(r*np.cos(np.arange(n)*theta))
    Y_coord = np.round(r*np.sin(np.arange(n)*theta))
    XY_coord = np.vstack((X_coord, Y_coord)).astype(int).T
    for kk in range(len(map_list)):
        print(kk)
        f_name = map_list[kk]
        f_val = f_name.split("/")[-1].split(".")[0]
        print("Computing val stack for: {}".format(f_val))
        with open(f_name,'rb') as f:
            f_map = pickle.load(f)
        if kk is 55:
            continue
        print("Working on {}/{} file: {}".format(kk+1, len(map_list), map_list[kk]))
        prediction(f_map, node_feat_arr, XY_coord)
    mean1 = sumError1/N
    std1 = np.sqrt(sumSquare1/N - (mean1**2))

    print("Working for radius:", r)
    print("Euclidean Error1: ************")

    mean1 = np.mean(mean1[1:]) #comute mean and st of prediction error of features of neighbor patch
    std1 = np.mean(std1[1:])
    
    m1.append(mean1)
    s1.append(std1)

    print("------------------------")
    try:
        mean2 = sumError2/N
        dev = sumSquare2/N - (mean2**2)
        std2 = np.sqrt(dev)

    except Exception as e:
        print("I came here", e)

    mean2 = np.mean(mean2[1:])
    std2 = np.mean(std2[1:])
    
    m2.append(mean2)
    s2.append(std2)

    print("number of patches:", N)

print("Total prediction Error mean1:", m1)
print("Total prediction Error standard deviation1:", s1)
print("Total prediction Error mean2:", m2)
print("Total prediction Error standard deviation2:", s2)

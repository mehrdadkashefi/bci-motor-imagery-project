################################################################
# Programmer : Mehrdad Kashefi i8
# Date: Jan 17th 2019
# Version : Initial Version
################################################################
# Objective
# This program calculates CSP transform for given Data
# Input Shape: (Trials,time_samples, channels)
################################################################
import numpy as np


def csp(class_1, class_2, m):
    class_1_cov = 0
    class_2_cov = 0

    # Making Channels zero-mean and calculating covariance matrices
    for i in range(len(class_1)): # Operation for class I
        class_1[i, :, :] - np.mean(class_1[i, :, :], 0)
        cov = np.dot(class_1[i, :, :].T, class_1[i, :, :])
        class_1_cov = class_1_cov + cov/np.trace(cov)

    for i in range(len(class_2)): # Operation for class II
        class_2[i, :, :] - np.mean(class_2[i, :, :], 0)
        cov = np.dot(class_2[i, :, :].T, class_2[i, :, :])
        class_2_cov = class_2_cov + cov/np.trace(cov)

    class_1_cov /= len(class_1)
    class_2_cov /= len(class_2)

    # Eigen Value decomposition for class 1
    eig_val_1, eig_vec_1 = np.linalg.eig(np.dot(np.linalg.inv(class_2_cov), class_1_cov))
    eig_vec_1 = eig_vec_1[:, np.argsort(-eig_val_1)]

    # Eigen Value decomposition for class 2
    eig_val_2, eig_vec_2 = np.linalg.eig(np.dot(np.linalg.inv(class_1_cov), class_2_cov))
    eig_vec_2 = eig_vec_2[:, np.argsort(-eig_val_2)]
    # Pickup first m eigen vector of first and decompositions
    w = np.concatenate((eig_vec_1[:, 0:m], eig_vec_2[:, 0:m]), 1)

    return w

################################################################
# Objective
# This program calculates CSP transform with Tikhonov regularization for given Data
# Input Shape: (Trials,time_samples, channels)
################################################################


def csp_tikhonov(class_1, class_2, m, alpha, k):
    class_1_cov = 0
    class_2_cov = 0

    # Making Channels zero-mean and calculating covariance matrices
    for i in range(len(class_1)): # Operation for class I
        class_1[i, :, :] - np.mean(class_1[i, :, :], 0)
        cov = np.dot(class_1[i, :, :].T, class_1[i, :, :])
        class_1_cov = class_1_cov + cov/np.trace(cov)

    for i in range(len(class_2)): # Operation for class II
        class_2[i, :, :] - np.mean(class_2[i, :, :], 0)
        cov = np.dot(class_2[i, :, :].T, class_2[i, :, :])
        class_2_cov = class_2_cov + cov/np.trace(cov)

    class_1_cov /= len(class_1)
    class_2_cov /= len(class_2)

    # Eigen Value decomposition for class 1
    eig_val_1, eig_vec_1 = np.linalg.eig(np.dot(np.linalg.inv(class_2_cov + alpha*k), class_1_cov))
    eig_vec_1 = eig_vec_1[:, np.argsort(-eig_val_1)]

    # Eigen Value decomposition for class 2
    eig_val_2, eig_vec_2 = np.linalg.eig(np.dot(np.linalg.inv(class_1_cov + alpha*k), class_2_cov))
    eig_vec_2 = eig_vec_2[:, np.argsort(-eig_val_2)]
    # Pickup first m eigen vector of first and decompositions
    w = np.concatenate((eig_vec_1[:, 0:m], eig_vec_2[:, 0:m]), 1)

    return w

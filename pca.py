#!/usr/bin/python
# -*- coding: utf-8 -*-
########################################################################
# Name:
# 		pca
# Description:
# 		pca
# Author:
# 		wesley wu
# Python:
#       3.5
# Version:
#		1.0
########################################################################
import numpy as np
import matplotlib.pyplot as plt


def subtract_mean(matrix):
    for col in matrix.columns:
        mean = np.mean(matrix[col])
        sub_mean = lambda x: x - mean
        matrix[col] = matrix[col].apply(sub_mean)
    return matrix


def covariance(matrix):
    cov_list = np.array([])
    cols = matrix.columns
    c_header1, c_header2 = 'x', 'y'

    combinations = [[col, f] for f in cols for col in cols]
    for c1, c2 in combinations:
        comb_data = matrix[[c1,c2]]
        comb_data.columns = [c_header1, c_header2]
        multiplied_columns = [np.multiply(i, j) for i, j in comb_data.itertuples(index=False)]
        n = comb_data.shape[0]
        covariance = np.sum(multiplied_columns) / (n-1)
        cov_list = np.append(cov_list, covariance)

    cov_matrix = np.matrix(cov_list).reshape([len(cols), len(cols)])
    return cov_matrix


def get_principle_comp(eig_vals, eig_vecs, dimensions):
    # We sort the eigvals and return the indices for the
    # ones we want to include (specified by "dimensions" paramater)

    eigval_max = np.argsort(-eig_vals)[:dimensions]

    eigvec_max = eig_vecs[:, eigval_max]

    return eigvec_max



# Random seed
np.random.seed(139)
# Use two dimensions data
d_mean = np.array([6, 6])
d_cov = np.array([[5, 0],[20, 20]])
d_samples = np.random.multivariate_normal(d_mean, d_cov, 100).T
assert d_samples.shape == (2, 100) , "the dimension is not 2X100"


# Do the z-score normalization x*=(x-mean)/std_var(x)
def normalize_data(ndarray):
    ndarray = np.copy(ndarray)
    for row in range(len(ndarray)):
        sub_mean = np.mean(ndarray[row, :])
        sub_std_var = np.std(ndarray[row, :])
        for col in range(len(ndarray[row])):
            ndarray[row][col] = (ndarray[row][col] - sub_mean) / sub_std_var

    return ndarray

# Compute the Covariance/Correlation of the normalized matrix
def get_cov_matrix(nda, dimensions):
    nda_t = nda.T
    result = np.matmul(nda, nda_t)
    # divide the elements by n-1
    for row in range(dimensions):
        result[row] = np.divide(result[row],len(nda_t)-1)

    return result


# Get the normalized sample datas
d_samples_n = normalize_data(d_samples)
# Get the covariance matrix
cov_matrix = get_cov_matrix(d_samples_n, len(d_samples_n))

# Get the eigenvalue and eigenvector, eigenvector is display by columns
eig_val, eig_vec = np.linalg.eig(cov_matrix)


# Validate the result by the equation: Matrix *eigvector = eigvalue* eigenvector
for i in range(len(eig_val)):
    eigv = eig_vec[:,i].reshape(1,2).T
    np.testing.assert_array_almost_equal(cov_matrix.dot(eigv), eig_val[i] * eigv, decimal=6, err_msg='Not Equal!', verbose=True)

# Validate the length of the eigenvector equal one
for i in range(len(eig_val)):
    eigv = eig_vec[:,i].reshape(1,2).T
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(eigv))

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# change the standard basis to the basis given by the eigenvectorrs, x' = p'x
x_quote = np.dot(eig_vec, d_samples_n)

# draw the data in the graph, we will use multiple graph to visualise the data
fig = plt.figure()

# 1st sub graph
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_title('Original Data Set')
# plot original point, x direction (min, max)  (0,0)
ax1.plot([min(d_samples[0,:]),max(d_samples[0,:])], [0,0], color='red', alpha=0.7, linewidth= 2)
# plot original point, x direction (0,0) (min, max)
ax1.plot([0,0], [min(d_samples[1,:]),max(d_samples[1,:])], color='red', alpha=0.7, linewidth= 2)
ax1.scatter(d_samples[0], d_samples[1], s= np.pi * (3)**2, color='b', alpha = 0.5)


# 2nd sub graph
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_title('Normalized Data Set')
# plot original point, x direction (min, max)  (0,0)
ax2.plot([min(d_samples_n[0,:]),max(d_samples_n[0,:])], [0,0], color='red', alpha=0.7, linewidth= 2)
# plot original point, x direction (0,0) (min, max)
ax2.plot([0,0], [min(d_samples_n[1,:]),max(d_samples_n[1,:])], color='red', alpha=0.7, linewidth= 2)
ax2.scatter(d_samples_n[0], d_samples_n[1], s= np.pi * (3)**2, color='b', alpha = 0.5)

# 3re sub graph
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_title('PCA With The Original Basis')
ax3.plot([min(d_samples_n[0,:]),max(d_samples_n[0,:])], [0,0], color='red', alpha=0.7, linewidth= 2)
# plot original point, x direction (0,0) (min, max)
ax3.plot([0,0], [min(d_samples_n[1,:]),max(d_samples_n[1,:])], color='red', alpha=0.7, linewidth= 2)
# plot the max variance direction
for e_, v_ in zip(eig_val, eig_vec.T):
    ax3.plot([0, 3*e_*v_[0]], [0, 3*e_*v_[1]], 'g-', alpha=0.7, lw=2)
ax3.scatter(d_samples_n[0], d_samples_n[1], s= np.pi * (3)**2, color='b', alpha = 0.5)

#4th sub graph
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_title('The Basis Of Eigenvectors')
ax4.plot([min(x_quote[0,:]),max(x_quote[0,:])], [0,0], color='green', alpha=0.7, linewidth= 2)
ax4.plot([0,0], [min(x_quote[1,:]),max(x_quote[1,:])], color='green', alpha=0.7, linewidth= 2)
ax4.scatter(x_quote[0,:], x_quote[1,:], s= np.pi * (3)**2, color='b', alpha = 0.5)
ax4.axis([-5,5,-5,5])

plt.show()

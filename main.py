#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
Created on Wed Nov 15 10:21:42 2017

@author: Sarunas Nejus and Arturo May Alvarez
'''

import numpy as np
import os

def myPCA_eig(X):
    '''
    Performs PCA using eigenvalue decomposition of the data's covariance matrix
    Arguments: 
        X     -- A numpy array of size m x n.
                 m is the number of obervations
                 n is the number of variables
                 
    Returns:
        model -- A python dictionary containing our parameters:
                 mu     -- A numpy array of size n x 1: The average of rows
                           in matrix X
                 sigma  -- A numpy array of size n x 1: The standard deviation
                           of the observed data along each principal axis
                 W      -- A numpy array of size n x min(m, n): The eigenvectors
                           of the covariance matrix
     '''
     
    import numpy as np
    
    # find the array size
    m, n = X.shape 
         
    # compute the mean
    mu = np.mean(X, axis=0)

    # centre the data around the mean 
    X_centred = X - np.repeat(np.column_stack(mu), m, axis=0)
    
    # compute the covariance matrix
    S = np.dot(X_centred.T, X_centred) / (m-1)
    
    #variance = np.sum(X_centred**2, axis=0) / m # just in case it's needed
    
    # compute the eigenvalue decomposition
    [eigenValues, W] = np.linalg.eig(S)
    sigma = np.sqrt(eigenValues)
    model = {"mu": mu, "sigma": sigma, "W": W}
    return model

def myPCA_svd(X):
    '''
    Performs PCA using Singular Value Decomposition(SVD) technique 
    Arguments:
        X       -- A numpy array of size mxn.
                   m is the number of observations.
                   n is the number of variables.
    Returns:
        model   -- A python dictionary containting your parameters:
                mu      -- A numpy array of size n x 1: The average of 
                           rows in matrix X.
                sigma   -- A numpy array of size n x 1: The standard 
                           deviation of the observed data along each 
                           principal axis.
                W       -- A numpy array of size n x min(m,n): The 
                           eigenvectors of the covariance matrix.
    '''

    import numpy as np
    
    # find the array size
    m, n = X.shape 
    
    # find the mean
    mu = np.mean(X, axis=0)
    
    # centre the data around the mean 
    X_centred = X - np.repeat(np.column_stack(mu), m, axis=0)
    
    # we could simply use (U, S, V) = np.linalg.svd(X) but let's do it
    # step-by-step aiming to understand the process underneath
    # find eigenvalues of X^T * X and sort them in a descending order
    [eigenValues, W] = np.linalg.eig(np.dot(X_centred.T, X_centred))
    eigenValues[::-1].sort()

    # find singular values and create diagonal matrix
    singularValues = np.diag(np.sqrt(eigenValues))
    
    # find transposed right singular vectors 
    V = W # V.T are right singular vectors
    
    # find left singular vectors    
    #U = np.dot(np.dot(X, V), np.linalg.inv(singularValues))
    
    # find standard deviation along each of the principal axes
    sigma = np.diag(np.sqrt(singularValues**2 / (m-1)))
    
    # test validity
    #Y = np.dot(np.dot(U, singularValues), V.T)

    model = {"mu": mu, "sigma": sigma, "W": V}
    return model

def is_diagonal(matrix):
    '''
    Finds if the input matrix is diagonal 
    Arguments:
        matrix      -- input matrix
        
    Returns:
        True    -- If the matrix is diagonal
        False   -- If the matrix is not diagonal
    '''
        
    import numpy as np
    
    dummy_matrix = np.ones(matrix.shape, dtype=np.uint8)
    
    # Fill the diagonal of dummy matrix with 0.
    np.fill_diagonal(dummy_matrix, 0)
    
    return np.count_nonzero(np.multiply(dummy_matrix, matrix)) == 0

def test_pca_implementation(fun, X, eps = 1e-8):
    '''
    Tests if the PCA function is correctly implemented. Checks: (1) Mean 
    calculation, (2) If covariance matrix is diagonal, (3) Standard deviation
    calculation. Notifies the user about the successfullness of the checks.
    
    Arguments:
        fun     -- The function handle to the pca implementation
        X       -- A numpy array of size m x n
        eps     -- A small threshold value used to check if a value is zero
                   or not
                       
    Returns:
        True    -- If the code is correct
        False   -- If the code is not correct
    '''
    
    import numpy as np
    
    model = fun(X)
    mu = model["mu"]
    W = model["W"]
    sigma = model["sigma"]
    sigma = np.reshape(sigma, (sigma.shape[0], 1))
    
    # check if the vector mu is equal to the average of rows in the matrix X
    mu_right = False
    if np.all(mu == np.mean(X, axis=0, keepdims=True)):
        mu_right = True
        print ("SUCCESS: Mean is calculated right")
    else:
        print ("ERROR: Mean calculation")
    
    # compute the covariance matrix of the transformed data
    Y = np.dot(X, W)
    Y = Y - np.mean(Y, axis=0, keepdims=True)
    S = np.dot(Y.T, Y)/(Y.shape[0]-1)   
    
    # Convert elements that are less than eps (zero threshold) to 0
    for row in range(0, S.shape[0]):
        for col in range(0, S.shape[1]):
            if (S[row, col] <= eps):
                S[row, col] = 0
    
    # check if the estimated covariance matrix is diagonal
    # if yes - new features are independent of each other   
    cov_mat_right = False
    if is_diagonal(S):
        cov_mat_right = True
        print ("SUCCESS: Covariance matrix is diagonal")
    else:
        print ("ERROR: Covariance matrix is not diagonal")
        
    # check if the sigma values are equal to the squared root of the diagonal
    # entries in the covariance matrix S
    sigma_right = False
    if np.all(np.isclose(np.absolute(sigma[:,0].T**2 - np.diag(S)), eps)):
        sigma_right = True
        print ("SUCCESS: Standard deviation is calculated right")
    else:
        print ("ERROR: Standard deviation calculation")

    
    if (mu_right == cov_mat_right == sigma_right == True):
        return True
    else:
        return False 
        
    
def read_single_vtk(inputPath, npts):
    '''
        Arguments:
            inputPath   -- Path to a single vtk file
            
        Returns:
            pts         -- A numpy array of size n x 3
                           n is the total number of points. In each row, 
                           'x', 'y', and 'z' coordinates are stored
    '''
        
    # open file
    vtkFile = open(inputPath, 'r')
  
    # pre-allocate an empty npts x 3 matrix
    pts = np.empty([npts, 3])
        
    # start working through the file
    with vtkFile as f:
        for i, l in enumerate(f):
            # data starts in line 6
            if (i >= 6) & (i < 27):
                #print i
                #print l
                # one line contains 9 values - must be split into 3x3
                a = map(float, l.split())
                pts[(i-6)*3, :] = a[0:3]
                pts[(i-6)*3+1, :] = a[3:6]
                pts[(i-6)*3+2, :] = a[6:9]
                
            # the last line with data - only contains 6 values
            if i == 27:
                a = map(float, l.split())
                pts[(i-6)*3, :] = a[0:3]
                pts[(i-6)*3+1, :] = a[3:6]
    
    vtkFile.close()
    return pts


def read_vtk_folder(rootFolder, npts):
    '''
        Arguments:
            rootPath    -- Path to a folder containing vtk files
            npts        -- Number of points
            
        Returns:
            data        -- A numpy array of size m x n
                           m is the total number of subjects. 
                           n is the total number of npts * 2
    '''

    data = np.array([]).reshape(0,npts*2)
    for (root, dirs, files) in os.walk(rootFolder):
        for file in files:
            if file.endswith(".vtk"):
                pts = read_single_vtk(os.path.join(rootFolder, file), npts)
                pts = np.delete(pts, (2), axis = 1)
                pts = np.reshape(pts, (1,npts*2), order = 1)
                data = np.append(data, pts, axis = 0)
                print (file + " read")
                
    return data
            


#==============================================================================
# # read points of all subjects
# npts = 65            
# data = read_vtk_folder("C:\Users\saras\Google Drive\EEE6230\data", npts)
# 
# # learn principal modes of variation
# model = myPCA_eig(data)
# 
# # visualise the first three modes of variation 
# import matplotlib.pyplot as plt
# mu = model["mu"]
# sigma = model["sigma"]
# W = model["W"]
# 
# mode = 0
# 
# pts1 = mu - 3*sigma[mode]*W[:,mode]
# pts2 = mu + 3*sigma[mode]*W[:,mode]
# 
# plt.figure()
# plt.scatter(pts1[range(0,npts)], pts1[range(npts,2*npts)])
# plt.gca().invert_yaxis()
# plt.axis("equal")
# 
# plt.figure()
# plt.scatter(mu[range(0,npts)], mu[range(npts, 2*npts)])
# plt.gca().invert_yaxis()
# plt.axis("equal")
# 
# plt.figure()
# plt.scatter(pts2[range(0,npts)], pts2[range(npts, 2*npts)])
# plt.gca().invert_yaxis()
# plt.axis("equal")
#==============================================================================





# create the Matrix A with the values given in the project guide
A = np.array([[-149, -50, -154], 
              [537, 180, 546], 
              [-27, -9, -25]])
    

#print (test_pca_implementation(myPCA_eig, A))


    

        
                                   

                           
b# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 22:13:44 2017
@author: Sarunas Nejus
"""

def is_diagonal(matrix):
    '''
    Finds if the input matrix is diagonal 
    Arguments:
        matrix  -- input matrix
        
    Returns:
        True    -- If the matrix is diagonal
        False   -- If the matrix is not diagonal
    '''
        
    import numpy as np
    dummy_matrix = np.ones(matrix.shape, dtype=np.uint8)
    
    # Fill the diagonal of dummy matrix with 0.
    np.fill_diagonal(dummy_matrix, 0)
    return np.count_nonzero(np.multiply(dummy_matrix, matrix)) == 0


def householder_transformation(b, k):
    '''
    Does one step of householder transformation. 
    Note: k < m otherwise it will not work (division by 0 occurs)
        Arguments:
            b   -- A numpy array of size m x 1
            k   -- An integer index between 1 and m-1. 
                   The algorithm will set the entries below k to zero. 
        
        Returns:
            H   -- A numpy array of size m x m: The Householder Matrix
    '''
    import numpy as np

    # input vector length
    m = b.shape[0]
    
    # check if a legit k is provided
    if k >= m:
        print ("ERROR: k =", k, " is equal or greater than the input vector length.")
        return
    
    # pre-define r (transformed) vector 
    r = np.zeros(m)

    # preserve input vector values up to index k
    r[0:k-1] = b[0:k-1]
    
    # get the norm of the remaining values (value 's')
    # k-1 since the vector index starts at 0
    r[k-1] = np.linalg.norm(b[k-1:m]) 
    
    # vector normal to the Householder mirror
    w = np.transpose(b-r[np.newaxis]) # newaxis allows transposing b-r

    # Householder matrix
    H = np.identity(m) - (2 / np.linalg.norm(w)**2) * w * w.T
    
    # only use the plot if m = 2
# =============================================================================
#     import matplotlib.pyplot as plt
#     plt.quiver([0, 0, 0], [0, 0, 0], [b[0], r[0], w[0]], [b[1], r[1], w[1]], color=['r','r','b'], 
#                angles='xy', scale_units='xy', scale=1)
#     plt.xlim(-10, 10)
#     plt.ylim(-10, 10)
#     plt.show()
#     plt.show()
# =============================================================================
    return H
    
    
def bidiag_reduction(A):
    '''
    Compute the bidiagonal reduction of matrix A = UBV^T
        Arguments:
            A   -- A numpy array of size m x n
        
        Returns:
            B   -- A numpy array of size m x n. Upper bidiagonal matrix.
            U   -- A numpy array of size m x n. Unitary matrix.
            V   -- A numpy array of size n x n. Unitary matrix.
    '''
    import numpy as np
    
    [m, n] = A.shape
    
    # initialise parameters 
    U = np.identity(m)
    V = np.identity(n)
    B = A

    # check the size of the input matrix and set the number of householder steps
    # accordingly
    if m > n:
        col = n+1
    elif m == n:
        col = n
    else:
        print('ERROR: The number of rows in the input matrix must be equal or higher to the number of columns')
        return        
    
    for k in range(1, col):
        # zeroe out bottom-left of the matrix working through the columns
        Q = householder_transformation(B[:, k-1], k)       
        B = np.dot(Q, B)    
        U = np.dot(U, Q)
        
        # zeroe out top-right of the matrix working through the rows
        if (k <= n-2):
            P = householder_transformation(B[k-1, :], k+1)
            B = np.dot(B, P)
            V = np.dot(V, P)
    return U, B, V


def test_bidiag_reduction(fun, A, eps = 1e-8):
    '''
    Test if the bidiagonal reduction algorithm performs as expected
            Arguments:
                fun     -- The function handle.
                A       -- A numpy array of size m x n. Data matrix.
                eps     -- A small value used as a threshold to check
                           if a value is zero or not.
            Returns:
                True    -- If the code is correct.
                False   -- If the code is not correct.
    '''
    import numpy as np
    
    # obtain the outputs of the function under test
    U, B, V = fun(A)
    
    # of rows and columns
    [m, n] = A.shape
    
    # initialise identity matrices for the checks
    I_U = np.identity(m)
    I_V = np.identity(n)
    
    # Step 1: check if A = U*B*V^T
    factorisationCorrect = False
    product = np.dot(np.dot(U, B), V.T)
    if np.allclose(A, product, rtol=0, atol=eps):
        print('SUCCESS: Step 1: A = U*B*V^T')  
        factorisationCorrect = True
    else:
        print('ERROR: Step 1: A != U*B*V^T')
    
    # Step 2: Check if U is a unitary matrix, U*U^T = I 
    U_isUnitary = False
    product = np.dot(U, U.T)
    if np.allclose(I_U, product, rtol=0, atol=eps):
        print('SUCCESS: Step 2: U is a unitary matrix')
        U_isUnitary = True
    else:
        print('ERROR: Step 2: U is not a unitary matrix')

    # Step 3: Check if V.T is a unitary matrix, V*V^T = I _isUnitary = False
    product =  np.dot(V, V.T)
    if np.allclose(I_V, product, rtol=0, atol=eps):
        print('SUCCESS: Step 3: V.T is a unitary matrix')
        V_isUnitary = True
    else:
        print('ERROR: Step 3: V.T is not a unitary matrix')
        
    # Step 4: check if B is an upper bidiagonal matrix
    # make sure values less than eps are zero
    B[abs(B) < eps] = 0
    B_isUpBidiag = False
    
    # check if the upper triangle (above superdiagonal) contains zeroes
    if np.allclose(B[0:m-1, 1:], np.tril(B[0:m-1, 1:])):
        # check if lower triangle under diagonal contains zeroes
        if np.allclose(B, np.triu(B)):
            B_isUpBidiag = True
            print('SUCCESS: Step 4: B is an upper bidigiagonal matrix')
        else:
            B_isUpBidiag = False
            print('ERROR: Step 4: B is not an upper bidigiagonal matrix')
    else:
        B_isUpBidiag = False
        print('ERROR: Step 4: B is not an upper bidigiagonal matrix')
    
    # check if all checks were successfull      
    if(factorisationCorrect == U_isUnitary == V_isUnitary == B_isUpBidiag == True ):
        return True
    else:
        return False
    
def givens_rot(alpha, beta, i, j, m):
    '''
    Find Givens Rotation matrix that transforms the input vector of [..., alpha, beta, ...]
    into [..., sqrt(alpha^2 + beta^2), 0, ...]
    Arguments:
        alpha   -- The i-th component of a vector b
        beta    -- The j-th component of a vector b
        i       -- The index i
        j       -- The index j
        m       -- The length of vector b
            
    Returns:
        R       -- A numpy array of size m x m. Givens Rotation Matrix
    '''
    # pre-define the matrix and find the norm    
    R = np.identity(m)    
    
    if alpha == 0:
        cos = 0
        sin = 1
    else:
        r = np.sqrt(alpha**2 + beta**2)
        # calculate cos and sin terms
        cos = alpha / r
        sin = -beta / r
    
    # place the values within the Givens Rotation matrix
    R[i, i] = cos
    R[j, j] = cos
    R[i, j] = -sin
    R[j, i] = sin

    return R
    
def golub_kahan_svd_step(B, U, V, iLower, iUpper):
    '''
    Does one iteration of Golub-Kahan chasing operation down the diagonal
    Arguments:
        B       -- A numpy array of size m x n. Upper diagonal matrix
        U       -- A numpy array of size m x m. Unitary matrix
        V       -- A numpy array of size n x n. Unitary matrix
        iLower, iUpper -- Identify the submatrix B22
        
    Returns:
        B       -- A numpy array of size m x n. Upper diagonal matrix with 
                   smaller values on the upper diagonal elements
        U       -- A numpy array of size m x m. Unitary matrix
        V       -- A numpy array of size n x n. Unitary matrix
    '''

    # extract the diagonal block from the input
    B22 = B[iLower:iUpper, iLower:iUpper]
    
    # save time and storage by multiplying only the bottom 3x3 block of B22
    if iUpper-iLower >=3:
        # tridiagonalise the bottom right, 3x3 block (no need to calculate the entire matrix)
        C = np.dot(B22[-3:, -3:].T, B22[-3:, -3:])
        # extract the right bottom 2x2 block
        C = C[-2:, -2:]
    else:
        # extract the right bottom 2x2 block
        C = np.dot(B22[-2:, -2:].T, B22[-2:, -2:])

    # find its eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(C)
    
    # find which eigenvalue is closer to the bottom right value and set mu
    mu = eigenvalues[np.abs(eigenvalues - C[1, 1]).argmin()]
    
    # set initial alpha and beta
    alpha = B22[iLower, iLower]**2 - mu
    beta = B22[iLower, iLower] * B22[iLower, iLower+1]

    m, n = B.shape
    
    for k in range(iLower, iUpper-1):
        # row-wise
        R = givens_rot(alpha, beta, k, k+1, n)
        B = np.dot(B, R.T)
        V = np.dot(V, R.T)
        
        # column-wise
        alpha = B[k, k]; beta = B[k+1, k]
        R = givens_rot(alpha, beta, k, k+1, m)
        B = np.dot(R, B)
        U = np.dot(U, R.T)
        
        # set alpha and beta for next iteration
        if (k < iUpper-2):
            alpha = B[k, k+1]; beta = B[k, k+2]

    return U, B, V
    
def golub_reinsch_svd(A, MAX_ITR = 2000, eps = 1e-8):
    '''
    Performs Golub-Reinsch iterative SVD algorithm for the input matrix
    Arguments:
        A       -- A numpy array of size m x n with m >= n
        MAX_ITR -- Number of maximum iterations
        eps     -- A small value used as a threshold to check if a value is
                   zero or not
                   
    Returns:
        U       -- Unitary matrix of size m x m.
        S       -- A numpy array of length n - singular values.
        V       -- Unitary matrix of size n x n.
        count -- Number of iterations before convergence
    '''
    U, B, V = bidiag_reduction(A) 
    m, n = B.shape    
    count = 0

    for q in range(0, n):
        # check if the matrix is diagonal
        while(is_diagonal(B[n-q-2:n-q,n-q-2:n-q]) == False):
            # if not, increase the count and run the svd step
            if count == MAX_ITR:
                print('WARNING: Reached maximum # of iterations without convergence')
                print('Returning the corresponding U, B, V ...')
                return U, B, V, count
            
            count = count + 1
            [U, B, V] = golub_kahan_svd_step(B,U,V,0,n-q)

            # check values in the superdiagonal - if less than the threshold,
            # set to zero
            for j in range(0, n-1):
                if(abs(B[j,j+1]) <= (abs(B[j,j])+abs(B[j+1,j+1])) * eps):
                    B[j,j+1] = 0
            B[abs(B) < eps] = 0
            
    # eliminate small values from the matrix
    B[abs(B) < eps] = 0
    B = B[:n, :n]
    
    # sort the singular values in a descending order if the diagonalisation
    # algorithm completed successfully
    if is_diagonal(B):
        U, B, V = sort_singular_values(U, B, V)
    S = np.diag(B)

    return U, S, V, count

def sort_singular_values(U, B, V):
    '''
    Sorts singular values in a descending order
    Arguments:
        U       -- A numpy array of size m x m. Unitary matrix
        B       -- A numpy array of size m x n. Unsorted singular values
        V       -- A numpy array of size n x n. Unitary matrix
                   
    Returns:
        U       -- A numpy array of size m x m. Corresponding unitary matrix
        B       -- A numpy array of size m x n. Sorted singular values
        V       -- A numpy array of size n x n. Corresponding unitary matrix
    '''
    
    # find the sorting order and enable writing 
    order = np.flip(np.argsort(np.diag(B)), 0)
    B.setflags(write=1)
    U.setflags(write=1)
    V.setflags(write=1)
    
    # sort the values in a descending order
    for i in range(0, order.shape[0]):
        if order[i] != i:
            # swap values/columns
            B[i, i], B[order[i], order[i]] = B[order[i], order[i]], B[i, i]
            U[:, [i, order[i]]] = U[:, [order[i], i]]
            V[:, [i, order[i]]] = V[:, [order[i], i]]
            
            # update the order indexes with the swap
            index = np.where(order==i)
            order[index[0]] = order[i]
            
    return U, B, V

def test_golub_reinsch_svd(fun, A, eps=1e-8):
    '''
    Tests Golub-Reinsch SVD algorithm implementation
    Arguments:
        fun     -- The function handle
        A       -- A numpy array of size m x n. Data matrix
        eps     -- A small value used as a threshold to check if a value is 
                   zero or not
                   
    Returns:
        True    -- If the code is correct
        False   -- If the code is not correct
    '''
        
    import numpy as np
   
    # obtain the outputs of the function under test
    U, S, V, counter = fun(A)
    
    # # of rows and columns
    m, n = A.shape
    
    # initialise identity matrices for the checks
    I_U = np.identity(m)
    I_V = np.identity(n)
    
    # Step 1: check if the SVD algorithm converged
    svdConverged = False
    if S.ndim == 1:
        svdConverged = True
        print('SUCCESS: Step 1: SVD converged successfully')
    else:
        print('ERROR: Step 1: SVD did not converge')
        
    # Step 2: check if singular values are all positive
    singularValPos = False
    if np.all(S >= 0):
        singularValPos = True
        print('SUCCESS: Step 2: Singular values are positive')
    else:
        print('ERROR: Step 2: Singular values contain negative elements')
        
    # Step 3: check if A = U*B*V^T
    B = np.zeros([m, n])
    if m >= n:
        B[:n, :n] = np.diag(S)
    else:
        B[:m, :m] = np.diag(S)
    factorisationCorrect = False
    product = np.dot(np.dot(U, B), V)
    if np.allclose(A, product, rtol=0, atol=np.amax(S)*eps):
        factorisationCorrect = True
        print('SUCCESS: Step 3: A = U*B*V^T')  
    else:
        print('ERROR: Step 3: A != U*B*V^T')
        
    # Step 4: Check if U is a unitary matrix, U*U^T = I 
    U_isUnitary = False
    product = np.dot(U, U.T)
    if np.allclose(I_U, product, rtol=0, atol=eps):
        U_isUnitary = True
        print('SUCCESS: Step 4: U is a unitary matrix')
    else:
        print('ERROR: Step 4: U is not a unitary matrix')

    # Step 5: Check if V.T is a unitary matrix
    V_isUnitary = False
    product =  np.dot(V, V.T)
    if np.allclose(I_V, product, rtol=0, atol=eps):
        V_isUnitary = True
        print('SUCCESS: Step 5: V.T is a unitary matrix')
    else:
        print('ERROR: Step 5: V.T is not a unitary matrix')
        
    # check if all checks were successfull      
    if(svdConverged == singularValPos == factorisationCorrect == U_isUnitary == V_isUnitary == True ):
        return True
    else:
        return False
    
def mysvd(A):
    '''
    Processes the input matrix and calls SVD function
    Arguments:
        A       -- A numpy array of size m x n, where m and n can be of any size. 
                   Data matrix
                   
    Returns:
        U       -- Unitary matrix of size m x m.
        S       -- A numpy array of length n - singular values.
        V       -- Unitary matrix of size n x n.
        count -- Number of iterations before convergence
    '''
    m, n = A.shape
    
    if (m < 2) or (n < 2):
        print('ERROR: Input is not a matrix')
        return
    
    if m >= n:
        U, S, V, count = golub_reinsch_svd(A)
        return U, S, V.T, count
    else:
        U, S, V, count = golub_reinsch_svd(A.T)
        return V, S, U.T, count
        
    
import numpy as np
import time

A = np.random.randn(80, 70)*10

# compare the performances
start = time.time()
U, S, V, count = mysvd(A)
end = time.time()
print("mysvd algorithm. Time elapsed:")
print(end - start)

start = time.time()
U_new, S_new, V_new = np.linalg.svd(A)
end = time.time()
print("Numpy svd algorithm. Time elapsed:")
print(end - start)

#U, S, V, count = mysvd(A)

#print(test_golub_reinsch_svd(mysvd, A))


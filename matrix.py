#!/usr/bin/python
# -*- coding: utf-8 -*-
########################################################################
# Name:
# 		matrix
# Description:
# 		Operations with Matrices from scratch
# Author:
# 		wesley wu
# Python:
#       3.5
# Version:
#		1.0
########################################################################
import copy as cp
import random as rd


# init the matrix with some value, value = 0 and value <> 0 (random int from 0 to 100)
def matrix_init(rows,columns,value):
    # init a null 2D array
    result = [[] for row in range(rows)]
    for i in range(rows):
        for j in range(columns):
            iv = 0 if value == 0 else rd.randrange(100)
            result[i].append(iv)
    return result


# matrix deep copy
def matrix_copy(x):
    return cp.deepcopy(x)


# two martix multply
def matrix_multiply(x,y):
    # make sure that x and y are conformable
    if (not x) or (not y) or (len(x[0]) != len(y)):
        print("Error: x and y are not conformable!")
        return None
    result = matrix_init(len(x), len(y[0]), 0)
    # iterate the rows of x
    for i in range(len(x)):
        # iterate the columns of y
        for j in range(len(y[0])):
            # iterate the rows of y
            for k in range(len(x[0])):
                result[i][j] += x[i][k] * y[k][j]
    return result


# vector init value = 0 and value <> 0 (random int from 0 to 100)
def vector_init(rows, value):
    result = [[] for i in range(rows)]
    for i in range(rows):
        iv = 0 if value == 0 else rd.randrange(100)
        result[i].append(iv)
    return result

# vector transpose, transform [[a],[b],[c]] to [a,b,c]
def vector_to_array(x):
    result = []
    for i in range(len(x)):
        result.append(x[i][0])
    return result


# array to vector, transform [a,b,c] to [[a],[b],[c]]
def array_to_vector(x):
    result = [[] for i in range(len(x))]
    for i in range(len(x)):
        result[i].append(x[i])
    return result


# vector production, x, and y should be vector [[a],[b],[c]]
def vectors_dot(x, y):
    # make sure that x and y are conformable
    if len(x) != len(y):
        print("Error: x and y are not conformable!")
        return None
    result = 0
    for i in range(len(x)):
        result += x[i][0] * y[i][0]
    return result


# vector round, keep decimal points
def vector_round(x, dpts):
    result = x
    for i in range(len(x)):
            result[i][0] = round(result[i][0], dpts)
    return result


# matrix transpose
def matrix_transpose(x):
    result = matrix_init(len(x[0]), len(x), 0)
    for i in range(len(x)):
        for j in range(len(x[0])):
            result[i][j] = x[j][i]
    return result


# matrix add
def matrix_add(x, y):
    # make sure that x and y are conformable
    if (len(x) != len(y)) or (len(x[0]) != len(y[0])):
        print("Error: x and y are not conformable!")
        return None
    result = matrix_init(len(x), len(x[0]), 0)
    for i in range(len(x)):
        for j in range(len(x[0])):
            result[i][j] =  x[i][j] + y[i][j]
    return result


# matrix subtract
def matrix_subtract(x, y):
    # make sure that x and y are conformable
    if (len(x) != len(y)) or (len(x[0]) != len(y[0])):
        print("Error: x and y are not conformable!")
        return None
    result = matrix_init(len(x), len(x[0]), 0)
    for i in range(len(x)):
        for j in range(len(x[0])):
            result[i][j] = x[i][j] - y[i][j]
    return result


# matrix round, keep decimal points
def matrix_round(x, dpts):
    result = x
    for i in range(len(x)):
        for j in range(len(x[0])):
            result[i][j] = round(result[i][j], dpts)
    return result


# make a square identity matrix
def matrix_identity(n):
    result = matrix_init(n,n,0)
    for i in range(n):
        result[i][i] = 1
    return result


# matrix multiply by scalar, x is matrix ,y is a scalar
def matrix_mulitply_with_scalar(x,y):
    result = matrix_init(len(x), len(x[0]), 0)
    for i in range(len(x)):
        for j in range(len(x[0])):
            result[i][j] = x[i][j] * y
    return result


# check if only zero exist at or below row i in cloumn j
def matrix_check_column_zero(x, i, j):
    non_zeros = []
    # the total number of non zero items
    zero_sum = 0
    # the index of first non zero item in the j column, start from 0
    first_non_zero = -1
    # start form i row and iterate all other rows
    for k in range(i, len(x)):
        non_zero = 1 if x[k][j] != 0 else 0
        non_zeros.append(non_zero)
        # update the init value of first_non_zero if find the non zero value
        if first_non_zero == -1 and non_zero:
            first_non_zero = k
        zero_sum = sum(non_zeros)

    return zero_sum, first_non_zero


# swap the two rows in a matrix, i and j start from zero
def matrix_swap_rows(x, i, j):
    '''result = matrix_copy(x)'''
    result = x
    result[i], result[j] = result[j], result[i]
    return result


# gauss jordan elimination
def matrix_rref_gasussjordan(x, cols):
    # clone the input
    tmp = matrix_copy(x)
    # begin the main loop
    # iterate through the rows
    i = 0
    # iterate through the columns, we only process the columns of original matrix
    for j in range(cols):
        '''print("process row {0} and column {1}".format(i, j))'''

        # verify the any non zero values below the current row in the current column
        zero_sum, first_non_zero = matrix_check_column_zero(tmp, i, j)

        if zero_sum == 0:
            # and if j is the end column, the job is done
            if j == len(tmp[0]):
                return tmp
            # this is a fat singular matrix
            print("This is a singular matrix!")
            return None

        # if the x[i][j] == 0 and there is non zero value below in the column,should swap it above
        if first_non_zero != i:
            tmp = matrix_swap_rows(tmp, i, first_non_zero)

        # divide x[i] by x[i][j] to make the x[i][j] equal one
        tmp[i] = [k / tmp[i][j] for k in tmp[i]]

        # other row should subtract the current x[i][j]*scalar to make zero below the i row in the column j
        for r in range(0, cols):
            if r != i:
                scaled_row = [tmp[r][j] * m for m in tmp[i]]
                tmp[r] = [tmp[r][m] - scaled_row[m] for m in range(0, len(scaled_row))]

        # if we iterate through the rows or columns, we done the job
        if i == cols or j == len(x[0]):
            break

        i += 1

    return tmp

# matrix inverse, use gauss-jordan elimination
# reference http://www.vikparuchuri.com/blog/inverting-your-very-own-matrix/
def matrix_invert_gaussjordan(x):
    '''
    we start by adding the identity matrix to the right of our matrix, and then apply the gauss_jordan elimination.
    finally we changed the original matrix into a row echelon form(identity matrix here), so get the inverted matrix
    at the right hand of the matrix
    '''
    # clone the input
    tmp = matrix_copy(x)

    # make a identity matrix and append it to the right of the tmp matrix
    idm = matrix_identity(len(x))
    for i in range(len(x)):
        tmp[i] += idm[i]
    # apply the gauss jordan elimination
    tmp = matrix_rref_gasussjordan(tmp, len(x))

    # now we should strip the right hand matrix, it is the inverted one
    if (tmp):
        for i in range(0, len(x)):
            ''' the original matrix : tmp[i] = tmp[i][0:len(x[0])] '''
            tmp[i] = tmp[i][len(x[0]):len(tmp[i])]

    return tmp


# solve the Ax=b , A is a square matrix and b is a non zero vector
def linalg_solve_gaussjordan(A, b):
    '''
        we start by adding the b to the right of our matrix, and then apply the gauss_jordan elimination.
        finally we changed the original matrix into a row echelon form(identity matrix here), so get the solved vector
        at the right hand of the matrix
        A, should be a non singular matrix
        b, should be vector: [[a],[b],[c]]
        return: vector [[a],[b],[c]]
        '''
    # make sure b is non zero
    if sum(b[:][0]) == 0:
        print("b should be a non zero vector!")
        return None
    # clone the input
    tmp = matrix_copy(A)

    # make a augmented matrix by append b to the right of the tmp matrix
    for i in range(len(A)):
        tmp[i].append(b[i][0])

    # apply the gauss jordan elimination
    tmp = matrix_rref_gasussjordan(tmp, len(A))

    # now we should strip the right hand matrix, it is the inverted one
    if (tmp != None):
        for i in range(0, len(A)):
            ''' the original matrix : tmp[i] = tmp[i][0:len(A[0])] '''
            tmp[i] = tmp[i][len(A[0]):len(tmp[i])]

    return tmp


# find the determinant of a matrix
def matrix_recursive_determinant(x, mul = 1):
    """
        Find the determinant in a recursive fashion. use the laplace expansion.
        x - Matrix object
    """
    cols = len(x)
    # return if the x is one dimension
    if cols == 1:
        return mul * x[0][0]
    else:
        sign = -1
        total = 0
        # i represent the ith row and ith column
        for i in range(cols):
            m = []
            # always start from the second row
            for j in range(1, cols):
                buff = []
                for k in range(cols):
                    if k != i:
                        buff.append(x[j][k])
                m.append(buff)
            sign *= -1
            total += mul * matrix_recursive_determinant(m, sign * x[0][i])
    return total

# create the pivoting matrix
def matrix_pivotize(m):
    """Creates the pivoting matrix for m."""
    n = len(m)
    ID = [[float(i == j) for i in range(n)] for j in range(n)]
    for j in range(n):
        row = max(range(j, n), key=lambda i: abs(m[i][j]))
        if j != row:
            ID[j], ID[row] = ID[row], ID[j]
    return ID


# LU decomposition, exists if and only if A is non singular matrix
def matrix_lu_factor(A):
    """Decomposes a nxn matrix A by PA=LU and returns L, U and P.
       A = P'LU
    """
    n = len(A)
    L = [[0.0] * n for i in range(n)]
    U = [[0.0] * n for i in range(n)]
    P = matrix_pivotize(A)
    A2 = matrix_multiply(P, A)
    for j in range(n):
        L[j][j] = 1.0
        for i in range(j+1):
            s1 = sum(U[k][j] * L[i][k] for k in range(i))
            U[i][j] = A2[i][j] - s1
        for i in range(j, n):
            s2 = sum(U[k][j] * L[i][k] for k in range(j))
            if U[j][j] != 0:
                L[i][j] = (A2[i][j] - s2) / U[j][j]
            else:
                # this is a fat singular matrix
                print("This is a singular matrix!")
                return (None, None, None)

    return (L, U, P)

# solves Ax = b given A's LU decomposition
def linalg_solve_lu(A, b):
    """Solves Ax = b given an LU factored matrix A and permuation vector p
    USAGE:
        x = solve( A, b )
        A, should be a non singular matrix
        b, should be a non-zero vector [[a],[b],[c]]
    """
    # make sure b is non zero
    if sum(b[:][0]) == 0:
        print("b should be a non zero vector!")
        return None

    n, m = (len(A), len(A[0]))
    if n != m:
        print("Error: input matrix is not square")
        return None
    # decomposition of A
    (L, U, P) = matrix_lu_factor(A)

    if (L == None) or (U == None) or (P == None):
        print("Error: This is a singular matrix!")
        return None
    # PAx = Pb   ==> PA = LU　and  LUx = Pb
    # Solution of n by n lower triangular system Ly = Pb
    p = []
    for k in range(n):
        p.append(P[k].index(1))
    y = [0 for k in range(n)]
    y[0] = b[p[0]][0]
    for i in range(1,n):
         y[i] = b[p[i]][0] - vectors_dot(array_to_vector(L[i][0:i]), array_to_vector(y[0:i]))

    # Solution of upper triangular system Ux = y
    x = [0 for k in range(n)]
    x[n-1] = y[n-1] / U[n-1][n-1]
    for i in range(n - 2, -1, -1):
        x[i] = (y[i] - vectors_dot(array_to_vector(U[i][i + 1:n]), array_to_vector(x[i + 1:n]))) / U[i][i]

    # transform x from array to vector
    x = array_to_vector(x)
    # All done, return solution vector
    return x


# ======================== verify the function ================================================#
# define matrices
A = [[4, 7, 17 ],
     [5, 7, 11],
     [2, 9, 13]
    ]
B = [[2, 3, 9, 4],
     [1, 2, 1, 5],
     [5, 4, 2, 6]
    ]

C = [[3, 6],
     [6, 12]
    ]

# define the vectors
b = [[41], [37], [89]]
c = [[41], [37], [89]]

# print(vectors_dot(b, c))

# A = matrix_init(100,100,1)
# b = vector_init(100,1)

# verify the gauss-jordan algorithm
x = linalg_solve_gaussjordan(A, b)
print("Gauss-jordan Solution:{0}".format(x))
print("Original Vector  b:{0}".format(b))
print("Verified Ax=b => b:{0}".format(matrix_multiply(A, x)))

# verify the LU decomposition algorithm
y = linalg_solve_lu(A, b)
print("LU decomposition Solution:{0}".format(y))
print("Original Vector  b:{0}".format(b))
print("Verified Ax=b => b:{0}".format(matrix_multiply(A, y)))



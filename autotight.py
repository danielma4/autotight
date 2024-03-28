import numpy as np
import cvxpy as cp
import scipy as sp
import math


#Pedagogical problem before Autotight
n = 3
a_array = np.array([4, 15, 2])
np.random.seed(5)

#Q matrix (cost)
Q_mat = np.zeros([n+2, n+2])
for i in range(2, n+2):
  Q_mat[i, i] = 1

#A_i matrices (constraint)
#n matrices, each (n+2)x(n+2)
A = np.zeros([n, n+2, n+2])

#for each matrix
for i in range(n):
  A[i, 0, 2+i] = A[i, 2+i, 0] = -a_array[i]
  A[i, 1, 2+i] = A[i, 2+i, 1] = 1

#defining decision variable, objective, and constraints
X = cp.Variable((n+2, n+2), symmetric=True)
#X is PSD
constraints = [X >> 0]
constraints += [
        cp.trace(A[i] @ X) == 0 for i in range(n)
        ]

prob = cp.Problem(cp.Minimize(cp.trace(Q_mat @ X)),
                  constraints)
prob.solve(solver=cp.MOSEK, verbose = False)

eigenvalues = sorted(np.linalg.eigvals(X.value), reverse = True)
print("The optimal value is", prob.value)
print("A solution X is\n", X.value)
print("using X, the sum is", np.sum([(1 / math.sqrt((X.value[1,1])) - a_array[i]) ** 2 for i in range(n)]))
print("Ratio between two largest eigenvalues of X*: ", eigenvalues[0] / eigenvalues[1])




#Autotight Algorithm
#half vectorization function
def vech(matrix):
     if len(matrix) != len(matrix[0]):
         raise ValueError("not a square matrix")
     rows = int((len(matrix)*(len(matrix)+1))/2)
     newVec = np.zeros([rows, 1])
     index = 0
     for col in range(len(matrix[0])):
         for row in range(col + 1):
             if row == col:
                 newVec[index][0] = matrix[row][col] / math.sqrt(2)
                 index += 1
             else:
                 newVec[index][0] = matrix[row][col]
                 index += 1
     return newVec

#inverse half vectorization, which creates a symmetric matrix
def inv_vech(vech):
    vech_flat = vech.flatten()
    n = len(vech_flat)
    mat_size = int(math.sqrt(2 * n + (1/4)) - (1/2))

    A = populateUpperTri(mat_size, vech_flat)
    A = A + A.T
    np.fill_diagonal(A, np.diagonal(A) / 2)
    return A

def populateUpperTri(mat_size, vec):
    A = np.zeros([mat_size, mat_size])
    index = 0
    for col in range(mat_size):
        for row in range(col + 1):
            if row == col:
                A[row][col] = vec[index] * math.sqrt(2)
                index += 1
            else:
                A[row][col] = vec[index]
                index += 1
    return A

#formulates the Y data matrix
def formulateY(N):
    print("N = ", N)
    vech_size = int(N)
    total_pts = int(1.2 * N)
    Y = np.empty([vech_size, 0])
    for i in range(total_pts):
        theta = np.random.rand(1)
        z_vals = 1 / (theta - a_array)
        x_feas = np.hstack(([1], theta, z_vals)) #creates the lifted vector
        Y = np.hstack((Y, vech(x_feas[None, :].T @ x_feas[None, :])))
    return Y

big_N = (n+2)*(n+3)/2
Y = formulateY(big_N)
print("shape of Y: ", Y.shape)

#QR factorization
Q, R = sp.linalg.qr(Y)
learned_constraints = Q[:, np.linalg.matrix_rank(Y) + 1:]
print("learned constraints shape: ", learned_constraints.shape)


#these are the inverse vech of the learned constraint column vectors
constraints.clear()
constraints += [X >> 0]
constraints += [
        cp.trace(inv_vech(learned_constraints[:, i]) @ X) == 0 for i in range(len(learned_constraints[0]))
        ]
constraints += [
        X[0][0] == 1
        ]
prob = cp.Problem(cp.Minimize(cp.trace(Q_mat @ X)),
                  constraints)
prob.solve(solver=cp.MOSEK, verbose = False)


# Print result.
print("The optimal value is", prob.value)
print("A solution X is\n", X.value)
print("using X, the sum is", np.sum([(1 / (X.value[0,1] - a_array[i])) ** 2 for i in range(n)]))

#checking for rank tightness by comparing greatest two eigenvalues
eigenvalues = sorted(np.linalg.eigvals(X.value), reverse = True)
print("Ratio between two largest eigenvalues of X*: ", eigenvalues[0] / eigenvalues[1])

import numpy as np

# generate positive semidefinite matrix with rank 1
def generate_rank_one_psd_matrix(n):
    v = np.random.rand(n)
    A = np.outer(v, v)
    return A

n=4
# Generate a list of 30 positive semidefinite matrices with rank 1
psd_matrices = [generate_rank_one_psd_matrix(n)for _ in range(30)]


# Print the matrices
for i, matrix in enumerate(psd_matrices):
    print(f"Matrix {i+1}:")
    print(matrix)
    print()

 #requirements: positive semidefinite / trace 1

#positive semidefinite: all eiganvalues >=0
#trace 1: value after calculation has diagonal add up to 1




#Todo：Calculate its best direction of measurement

#Todo: Calculate its best CHSH value

#Todo:Generate 20 parameters

#Todo: Extract one of the 20 parameters, add all the sine and cosine angles to the test

#Todo: Add all the parameters to the dataset

#Todo: Create CHSH values of these testing angles 
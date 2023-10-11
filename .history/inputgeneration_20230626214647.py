import numpy as np

# generate positive semidefinite matrix with rank 1
def generate_rank_one_psd_matrix(n):
    v = np.random.rand(n)
    A = np.outer(v, v)
    normalized_A = A / np.trace(A)  # Normalize matrix A by dividing by its 2-norm
    norm_2 = np.linalg.norm(normalized_A, ord=2)
    eigenvalues = np.linalg.eigvals(normalized_A)  # Calculate eigenvalues of matrix A
    return normalized_A, norm_2, eigenvalues

def generate_positive_semidefinite_matrices(n, num_matrices):
    matrices = []

    while len(matrices) < num_matrices:
        re_matrix = np.random.rand(n, n)
        im_matrix = np.random.rand(n, n)

        matrix = re_matrix + 1j * im_matrix
        matrix = np.dot(matrix, matrix.conj().T)  # Make the matrix positive semidefinite

        if np.trace(matrix).real == 1 and np.linalg.matrix_rank(matrix) > 1:
            eigvals = np.lina.eigvalsh(matrix)
            print(f{"Eigenvalues for Matrixrices) + 1}:\n{eigvals}\n")
            matrices.append(matrix)

    return matrices

matrices_list = generate_positive_semidefinite_matrices(4, 30)

for i, matrix in enumerate(mat(f"Matrix {i+1}:\n{matrix}\n")
#print the eigan value of matricies



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
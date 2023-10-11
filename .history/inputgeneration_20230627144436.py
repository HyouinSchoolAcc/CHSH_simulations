import numpy as np

# generate positive semidefinite matrix with rank 1
def generate_psd_matrix_with_eigenvalues():
    random_numbers = np.random.rand(4)
    #eigenvalues1=random_numbers/sum(random_numbers)
    eigenvalues1 = (0.3,0.3,0.2,0.2)
    n = 4
    # Generate random complex eigenvectors
    eigenvectors1 = np.random.rand(n, n) + 1j * np.random.rand(n, n)

    # Construct the PSD matrix using eigendecomposition
    hermitian_matrix1 = np.dot(eigenvectors1, np.diag(eigenvalues1))
    matrix1 = np.dot(hermitian_matrix1, np.transpose(np.conjugate(eigenvectors1)))
    pre_processed_matrix = matrix1
    matrix = pre_processed_matrix/sum(np.linalg.eigvals(pre_processed_matrix))
    return (matrix)


# Generate a list of 30 positive semidefinite matrices with rank 1
psd_matrices = [generate_psd_matrix_with_eigenvalues() for a in range(30)]
print (np.round(np.linalg.eigvals(psd_matrices[0]),decimals=2))
print (np.round(psd_matrices[0], decimals=2))

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
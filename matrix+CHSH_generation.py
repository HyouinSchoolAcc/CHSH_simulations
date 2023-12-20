import numpy as np
import math
#generate 1 psd matrix 
def generate_psd_matrix_with_eigenvalues():
    random_numbers = np.random.rand(4)
    eigenvalues1=random_numbers/sum(random_numbers)
    #eigenvalues1 = (0.3,0.3,0.2,0.2)
    n = 4
    # Generate random complex eigenvectors
    eigenvectors1 = np.random.rand(n, n) + 1j * np.random.rand(n, n)

    # Construct the PSD matrix using eigendecomposition
    hermitian_matrix1 = np.dot(eigenvectors1, np.diag(eigenvalues1))
    pre_processed_matrix  = np.dot(hermitian_matrix1, np.transpose(np.conjugate(eigenvectors1)))
    
    matrix = pre_processed_matrix/sum(np.linalg.eigvals(pre_processed_matrix))
    return (matrix)


def CHSH_val_calc(matrix, T_11, T_12, T_13, T_21, T_22, T_23, T_31, T_32, T_33):
    T = np.zeros((3, 3), dtype=np.complex128)
    T[0,0]=np.trace(np.dot(np.array(T_11),np.array(matrix)))
    T[0,1]=np.trace(np.dot(np.array(T_12),np.array(matrix)))
    T[0,2]=np.trace(np.dot(np.array(T_13),np.array(matrix)))
    T[1,0]=np.trace(np.dot(np.array(T_21),np.array(matrix)))
    T[1,1]=np.trace(np.dot(np.array(T_22),np.array(matrix)))
    T[1,2]=np.trace(np.dot(np.array(T_23),np.array(matrix)))
    T[2,0]=np.trace(np.dot(np.array(T_31),np.array(matrix)))
    T[2,1]=np.trace(np.dot(np.array(T_32),np.array(matrix)))
    T[2,2]=np.trace(np.dot(np.array(T_33),np.array(matrix)))
    U, s, Vt = np.linalg.svd(T)

    # Get the two largest values of s and square them
    largest_two_s_values = sorted(s, reverse=True)[:2]
    squares_of_largest_two_s_values = [x**2 for x in largest_two_s_values]

    # Add these squared values together
    sum_of_squares = sum(squares_of_largest_two_s_values)

    # Get the square root of the sum
    result = 2*math.sqrt(sum_of_squares)

    return result


matrix = generate_psd_matrix_with_eigenvalues()
print (np.round(np.linalg.eigvals(matrix),decimals=2))
print (np.round(matrix, decimals=2))

#write the kroniker matrix 
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
identity = np.eye(2)
T_11=np.kron(sigma_x, sigma_x)
T_12=np.kron(sigma_x, sigma_y)
T_13=np.kron(sigma_x, sigma_z)
T_21=np.kron(sigma_y, sigma_x)
T_22=np.kron(sigma_y, sigma_y)
T_23=np.kron(sigma_y, sigma_z)
T_31=np.kron(sigma_z, sigma_x)
T_32=np.kron(sigma_z, sigma_y)
T_33=np.kron(sigma_z, sigma_z)
T = np.zeros((3, 3), dtype=np.complex128)

for a in range (10000):
    matrix = generate_psd_matrix_with_eigenvalues()
    result = CHSH_val_calc(matrix, T_11, T_12, T_13, T_21, T_22, T_23, T_31, T_32, T_33)
    print(result)


    real_part = np.real(matrix)
    imaginary_part = np.imag(matrix)

    # Create a 2-channel 4x4 input image
    input_image = np.stack([real_part, imaginary_part], axis=2)

    file_path = r"data\data.txt"
    #file_path = r"c:\Users\17536\Desktop\CHSH\test.txt"
    with open(file_path, 'a') as f:
        # Write the matrix
        for i in range(input_image.shape[0]):
            for j in range(input_image.shape[1]):
                row_data = " ".join(str(x) for x in input_image[i,j ,:])
                f.write(row_data + "\n")
        
        # Write the result
        f.write(str(result)+"\n")

    print("Data written to file:", file_path)




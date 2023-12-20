import numpy as np
import math

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

# This function computes 4 outcomes values using the input matrix and the input angles
def function_2(A_1,B_1,matrix):
    A = np.deg2rad(A_1)
    B = np.deg2rad(B_1)

    # Define ket_0 and ket_1 if they are not already defined
    ket_0 = np.array([1, 0])
    ket_1 = np.array([0, 1])

    # Calculate ket_phi_A, ket_phi_B, ket_phi_A_, and ket_phi_B_
    ket_phi_A = np.cos(A) * ket_0 + np.sin(A) * ket_1
    ket_phi_B = np.cos(B) * ket_0 + np.sin(B) * ket_1
    ket_phi_A_ = -np.sin(A) * ket_0 + np.cos(A) * ket_1
    ket_phi_B_ = -np.sin(B) * ket_0 + np.cos(B) * ket_1

    # Calculate the array for the input function's measurement variables
    side_array1 = np.array([i * j for i in ket_phi_A for j in ket_phi_B])
    row_vector1 = side_array1.reshape(1, -1)
    column_vector1 = side_array1.reshape(-1, 1)
    side_array2 = np.array([i * j for i in ket_phi_A_ for j in ket_phi_B_])
    row_vector2 = side_array2.reshape(1, -1)
    column_vector2 = side_array2.reshape(-1, 1)
    side_array3 = np.array([i * j for i in ket_phi_A_ for j in ket_phi_B])
    row_vector3 = side_array3.reshape(1, -1)
    column_vector3 = side_array3.reshape(-1, 1)
    side_array4 = np.array([i * j for i in ket_phi_A for j in ket_phi_B_])
    row_vector4 = side_array4.reshape(1, -1)
    column_vector4 = side_array4.reshape(-1, 1)
    
    #Do the matrix multiplication
    result_vector1 = np.abs(row_vector1 @ matrix @ column_vector1)
    result_vector2 = np.abs(row_vector2 @ matrix @ column_vector2)
    result_vector3 = np.abs(row_vector3 @ matrix @ column_vector3)
    result_vector4 = np.abs(row_vector4 @ matrix @ column_vector4)
    
    return (result_vector1,result_vector2,result_vector3,result_vector4)
    

#initialize some variables for CHSH
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

#results for measurements
results = []
result = []
for a in range (10000):
    matrix = generate_psd_matrix_with_eigenvalues()
    CHSH_result = CHSH_val_calc(matrix, T_11, T_12, T_13, T_21, T_22, T_23, T_31, T_32, T_33)
    result.append(function_2(0, 22.5, matrix))
    result.append(function_2(0, 67.5, matrix))
    result.append(function_2(45, 22.5, matrix))
    result.append(function_2(0, 67.5, matrix))
    result.append(function_2(302.6487, 80.4775, matrix))
    result.append(function_2(199.3459, 307.7366, matrix))
    result.append(function_2(221.461, 99.953, matrix))
    result.append(function_2(13.686,140.040, matrix))
    results.append(np.array(result).flatten()) #results should now have a 20-element array, this should be written to the file and the output be the CHSH value.
    result.clear()
    #write the results to a file
    #now we will add in the results value to the file, and then the CHSH value, alternating
    file_path = r"data\measure_data.txt"
    with open(file_path, "a") as file:
        for res in results:
            # Writing the 20-element array
            file.write(", ".join(map(str, res)))
            # Adding a semicolon and then the CHSH value
    
            file.write("; " + str(CHSH_result) + "\n")
    #         print(res)
    # print("Data written to file:", file_path)
    # print(CHSH_result)

    results.clear()
for a in range (500):
    matrix = generate_psd_matrix_with_eigenvalues()
    CHSH_result = CHSH_val_calc(matrix, T_11, T_12, T_13, T_21, T_22, T_23, T_31, T_32, T_33)
    result.append(function_2(0, 22.5, matrix))
    result.append(function_2(0, 67.5, matrix))
    result.append(function_2(45, 22.5, matrix))
    result.append(function_2(0, 67.5, matrix))
    result.append(function_2(302.6487, 80.4775, matrix))
    result.append(function_2(199.3459, 307.7366, matrix))
    result.append(function_2(221.461, 99.953, matrix))
    result.append(function_2(13.686,140.040, matrix))
    results.append(np.array(result).flatten()) #results should now have a 20-element array, this should be written to the file and the output be the CHSH value.
    result.clear()
    #write the results to a file
    #now we will add in the results value to the file, and then the CHSH value, alternating
    file_path = r"data\measure_test.txt"
    with open(file_path, "a") as file:
        for res in results:
            # Writing the 20-element array
            file.write(", ".join(map(str, res)))
            # Adding a semicolon and then the CHSH value
    
            file.write("; " + str(CHSH_result) + "\n")
    #         print(res)
    # print("Data written to file:", file_path)
    # print(CHSH_result)

    results.clear()
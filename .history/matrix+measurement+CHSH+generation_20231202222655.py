import numpy as np

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
    
# Define the ket_psi_plus array (or a list of them)
ket_psis = [np.array([0.70710678+0j, 0+0j, 0+0j, 0.70710678+0j]), np.array([0.70322+3j, 0-0.5j, 0+0j, 0.120678+0j])]

# Create a list of outer products for each element in ket_psis
ket_psi_pluses = [np.outer(element, element) for element in ket_psis]
# Example of how to call the function with different ket_psi_pluses
results = []
result = []
for ket_psi_plus in ket_psi_pluses:
    result.append(function_2(0, 22.5, ket_psi_plus))
    result.append(function_2(0, 67.5, ket_psi_plus))
    result.append(function_2(45, 22.5, ket_psi_plus))
    result.append(function_2(0, 67.5, ket_psi_plus))
    result.append(function_2(302.6487, 80.4775, ket_psi_plus))
    results.append(np.array(result).flatten())
    result.clear()
results=np.array(results)

# Flatten the results array
print(results)

# Todo: Add in the code to generate the matricies, and have the list of 20 matricies be written to the file instead of the matrix being written.


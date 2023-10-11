import numpy as np

#Todo: Create an arbitrary real density matrix
def random_matrix (a):
    matrix=np.zeros(4,4,a)
    for b in range(0,a):
        matrix[b] = np.random(4,4)
        matrix[b].normalize(1)
        filter(<0,matrix[b]) 
        if matrix[0] is not empty
            continue

 #requirements: positive semidefinite / trace 1

#positive semidefinite: all eiganvalues >=0
#trace 1: value after calculation has diagonal add up to 1




#Todo：Calculate its best direction of measurement

#Todo: Calculate its best CHSH value

#Todo:Generate 20 parameters

#Todo: Extract one of the 20 parameters, add all the sine and cosine angles to the test

#Todo: Add all the parameters to the dataset

#Todo: Create CHSH values of these testing angles 
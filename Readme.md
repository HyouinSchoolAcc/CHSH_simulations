## Objective
The objective of this project is to create a tool that can identify based on theoretical prepared state and some measurement angles to determine the optimal CHSH inequality and the corresponding noise associated.

Think of it like if I have an apple, I need to take a photo to give it to someone else. That someone else now use the photo to tell that I took a picture of an apple.
In this case, the apple is the density_matrix, the photo are the "measured values" I generated. And the metric of how good my state is is the CHSH_value.

## Use of each file:
matrix+CHSH_generation.py: This file generates a standard density matrix and computes its CHSH mathmatically
#the following files are used for predicing CHSH value from input density matrix
{
    model_simple_neural.py: This file uses a simple neural network to predict the CHSH value of a density matrix
        results:at max 66% accuracy on test set, bad
    convol.py: This file uses a convolutional neural network to predict the CHSH value of a density matrix
        results:bad
    optuna_simple_model.py: This file uses optuna to optimize the simple neural network
        results:has been demonstrated to achieve 1.00 accuracy on test set,though result not saved
    optuna_multi_layer_model.py: This file uses optuna to optimize the convolutional neural network
        results:has consistantly produced many models that can achieve >95% accuracy on test set
    test_base_CHSH.py: This file uses the optuna_multi_layer_model.py's model.
        #note: the multiple layer models have different parameters, so using different models will require the use to adjust the layer sizes manually in the variable Layer_Sizes.
}

matrix+measurement+CHSH+generation.py: this generates 32 inputs and correlates with one CHSH output
        note: the angles are 4 sets from naive CHSH angles and 4 sets from randomly generated angles
#the following files are used for predicing CHSH values from the measured outputs alone
{

    measure_simple/measure_multi_layer.py: this is the training from 32 inputs and one CHSH output using deep nets, accuracy:0.8331
    measure_random_forest.py: this is training from the same dataset using random forest, accuracy:0.76
}
#note: The current best model only achieves 83% during training, which is terrible. I'll look for ways to increase
the chances more.

matrix+measurement+CHSH+noise.py
#the following files are taking in (matrix, measured values) to predict the noise source, of which are:
    angle uncertainty(measurement angle+=random(0,0.1))
    shotnoise uncertainty(measured value+=random(0,0.1))
    phase change uncertainty(densty matrix elements go through a phase change = e**(random(0,0.1)))
{
    //Tobedone

}

## How to verify:
1. install python packages pytorch, optuna, numpy, scipy, and other packages that may be missing
2. Run matrix+CHSH_generation.py, changing the filepath to the data's destination once, and to test once.
(run matrix+CHSH_generation.py twice, once to generate data, once to test the model)
3. Run optuna_multi_layer_model.py, changing the file path of both test and source model accordingly.

##How to train:
1. install  python packages pytorch, optuna, numpy, scipy, and other packages that may be missing
2. pick your model of choice, and run the corresponding file
3. change the file path of both test and source model accordingly.
4. If you wish to save the model and not running multi_layer, copy multi_layer's save model code into your file.
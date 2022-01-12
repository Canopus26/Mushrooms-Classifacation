# Mushrooms-Classifacation

Implemented neural networks from scratch to classify if Mushrooms are edible or poisonous.

Description:

data_preprocessing.py: In this file I converted all categorical values in numeric using hot encoder and I dropped row with null value and than I splitted data in 60% train,20% validate and 20% test and store them in different files as writtend in code.
  
formulas.py: In this file I calculated logistic activation function,derivative of output node with respect to input node,square error function and differnece between actual output and target value. 

models.py: In this file I implmented eval() for forward propogation in NN and implemented backprop() for backpropogation calculation in NN.I changed cfile() and added close(self) in cfile() to remove previous error.

proj_test.py:In this file I ran model on preprocessed data using eval(),backprop() and formulas for activation function and calculating accuracy.For layer 1 I used 10 nodes and for layer 2 I used 5 nodes.

My_Output.docx : I added screenshots of error for training,validation and testing data set.

After training error is calculated we have to press enter button to calculate validation error and similarly after calculating validation error.

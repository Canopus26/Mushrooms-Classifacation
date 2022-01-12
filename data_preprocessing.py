
#import the required libraries
import pandas as pd
from sklearn.model_selection import train_test_split

#giving names to columns
column_name = ['classes', 'cap-shape', 'cap-surface', 'cap-color', 'bruises?', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']

#getting dataset from agaricus-lepiota.data
X = pd.read_csv('agaricus-lepiota.data', names =column_name , na_values = "?")

#removing all null valued row
X.dropna(axis = 0, how = 'any', inplace = True)

#converting edible and poisonous to 0 and 1 value
y = X.classes
X.drop('classes', axis = 1, inplace = True)
y = y.map({'e': 0, 'p': 1})

#adding name of features in one array
features = ['cap-shape', 'cap-surface', 'cap-color', 'bruises?', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']

#hot encoder to convert all catogrical attributes in
X = pd.get_dummies(X, columns = features)

# split data set in train,test and validation
# 60% train,20% validation and 20% test data

train,test = train_test_split(X,test_size=0.2, random_state=1)

train,validate  = train_test_split(train, test_size=0.25, random_state=1)


					   
#saving dataset into text files
train.to_csv('training_data.txt', header = False, index=False)
validate.to_csv('validation_data.txt', header = False, index=False)
test.to_csv('testing_data.txt', header = False, index=False)


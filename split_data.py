import csv
import os
import numpy as np

'''Divide the data in creditcard.csv into two csv files train.csv and test.csv Among them, there are 227846 data in train.csv and 56961 data in test.csv'''
labels = []
data = []
a_train_file = 'data/train.csv'
a_test_file = 'data/test.csv'
a_file = 'creditcard.csv'

seed = 3
np.random.seed(seed)
train_indices = np.random.choice(284807, 227846, replace=False) # Set random number generation to randomly select 80% random numbers
residue = np.array(list(set(range(284807)) - set(train_indices)))
# If the combined data of the training set and the test set add up to a whole data set, this operation is not required
test_indices = np.random.choice(len(residue), 56961, replace=False) 
with open(a_file)as afile:
    a_reader = csv.reader(afile)  # Read all data from the original data set and save it to a_reader
    labels = next(a_reader)  # Extract the first row and set it as labels
    for row in a_reader:  # Extract the data of each row in a_reader and save it in the data list
        data.append(row)


# generate train.csv
if not os.path.exists(a_train_file):
    with open(a_train_file, "w", newline='') as a_trian:
        writer = csv.writer(a_trian)
        writer.writerows([labels])  
        writer.writerows(np.array(data)[train_indices])
        a_trian.close()

# generate test.csv
if not os.path.exists(a_test_file):
    with open(a_test_file, "w", newline='')as a_test:
        writer = csv.writer(a_test)
        writer.writerows([labels])  
        writer.writerows(np.array(data)[test_indices])
        a_test.close()
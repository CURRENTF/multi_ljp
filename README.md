# HRN

## Dataset resources 

Due to the sensitivity of the dataset, we will publish our dataset after the paper is in manuscript.

## Training and testing

**1.Download each training dataset**

Due to the sensitivity of the dataset, we will publish our dataset after the paper is in manuscript.

**2.Download mt5-base**

Download the mt5-base pre-training model, save the model parameters as best_.pkl, and place it in the checkpoint folder

**3.Train the HRN model on the training set and test it on the validation set** 

python train.py

**4.Test the effect of HRN on the test set**

python test.py

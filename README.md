# ACL409

## Dataset resources 

The data set is stored in the data folder, data.zip contains three files data_test.json, data_valid.json, data_train.json are test set, validation set and training set respectively

## Training and testing

**1.Prepare dataset**

First unzip data.zip, then use the following command to generate data suitable for the model.
```commandline
python generate_fid_data.py
```

**2.Download mt5-base pretrained model**

Download the mt5-base pre-training model, save the model parameters as mt5_.pkl, and place it in the checkpoint folder

**3.Train the HRN model on the training set and test it on the validation set** 
```commandline
python train.py
```

**4.Test the effect of HRN on the test set**

```commandline
python test.py
```
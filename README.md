# Revolt BI Hack

This repository contains a source to recreate our deep learning model
that detects N18 diagnosis from laboratory data.


# What we have achieved:

We created a model that automatically predicts CKD. Based on the data provided, we are able to predict the disease more than 800 days before diagnosis with an accuracy of more than 84%. 
We were also able to extract features from the model that seem to be important for CKD detection. 
The solution is general and can also be used for the detection of other diseases.
We can scan national laboratory databases to detect hidden affected individuals; the whole process is automatic. 

Early treatment is much cheaper and more effective for many diseases. In the case of CKD, early detection of the onset of the disease can significantly improve health outcomes for patients and save money for healthcare payers. Considering that one patient on haemodialysis (which may be necessary to treat late-stage CKD) costs approximately CZK 1 million (EUR 41 000) per year in the Czech Republic, the cost savings alone can be quite significant.


# Structure of the repository

* XXX TODO: Initial scripts
* preprocessing_pivoting.ipynb - Jupyter Notebook for preprocessing in final form that is used for training
* targets.ipynb - Jupyter notebook for creating a file that finds patients with diagnose that we are searching for
* train.py - The main script that creates model and perform the training and validation
* inspect.py - The script for computation of gradients of model to get some insight into the model
* inspect_analysis.ipynb - Visualization of gradients of input features


# Dependencies

* Tensorflow, Numpy, Pandas


# Dataprocessing pipeline

![Pipeline](imgs/arch.jpg)

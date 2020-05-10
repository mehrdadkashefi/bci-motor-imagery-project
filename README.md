BCI motor Imagery Classification with a Deep Convolutional Neural Network

Programmer: Mehrdad Kashefi

This program is a replication of the following paper:
A novel deep learning approach for classification of EEG motor imagery signals

by Yousef Rezaei Tabar and Ugur Halici

Open motor imagery dataset from BCI competition IV was used.Subject 1 is uploaded for testing the code. To access the full data use the following link:
http://www.bbci.de/competition/iv/

What each program does:

preprocessing.m:

This is a Matlab code that reads the sample data and creates images from EEG data based on the method described in the paper. The output of this code will be used in "convnet_auto.py", "convnet.py" and  "auto.py"

auto.py:

A python code for classifying the EEG images merely using the auto-encoder layer

convnet.py:

A python code for classifying the EEG images only using the convolutional layer.

convnet_auto.py:

The python code for classifying EEG images using both convnet and then auto-encoder, this is the main method presented in paper.

csp.py:

A python code for classifying EEG using classical method of using frequency band features of EEG and then feature selection using Common Spatial Pattern (CSP) algorythm.

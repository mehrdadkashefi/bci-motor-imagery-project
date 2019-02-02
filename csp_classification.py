################################################################
# Programmer : Mehrdad Kashefi i8
# Date: Feb 1st 2019
# Version : Initial Version
################################################################
# Objective
# This program classifies BCI competition IV with CSP
################################################################
import numpy as np
import scipy.io
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.utils import shuffle
from csp import csp
import scipy.signal as sig


num_subject = 9
num_filter = 4
num_fold = 10

subject_accuracy = np.zeros((num_subject, 1))
for subj in range(num_subject):

    mat = scipy.io.loadmat('/home/mehrdad/Datasets/BCI_IV_2b/Subject_'+str(subj+1)+'.mat')
    data = mat['total_data']
    label = mat['total_label']

    # Filering in 8 to 30 Hz
    [b, a] = sig.butter(3, [6, 30], btype='bandpass', analog=False, output='ba', fs=250)

    for channel in range(data.shape[2]):
        for tria in range(data.shape[0]):
            data[tria, :, channel] = sig.filtfilt(b, a, data[tria, :, channel])



    data, label = shuffle(data, label)

    kf = KFold(n_splits=num_fold, shuffle=False)
    kf.get_n_splits(data)

    fold_accuracy = np.zeros((num_fold, 1))
    fold_count = 0
    for train_index, test_index in kf.split(data):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]

        class_1 = X_train[np.squeeze(y_train == 0), :, :]
        class_2 = X_train[np.squeeze(y_train == 1), :, :]

        w = csp(class_1, class_2, num_filter)

        X_train_csp = np.dot(X_train, w)
        X_train_csp_norm = np.log(np.var(X_train_csp, 1))

        X_test_csp = np.dot(X_test, w)
        X_test_csp_norm = np.log(np.var(X_test_csp, 1))

        clf = svm.SVC()
        clf.fit(X_train_csp_norm, np.ravel(y_train))
        prediction = clf.predict(X_test_csp_norm)
        prediction = np.reshape(prediction, (len(prediction), 1))

        fold_accuracy[fold_count, 0] = sum(prediction == y_test)/len(prediction)
        fold_count += 1

    subject_accuracy[subj, 0] = np.mean(fold_accuracy)
    print('The mean accuracy for subject '+str(subj+1)+' is ', np.mean(fold_accuracy))
print("###########################")
print("The mean accuracy for all subject is ", np.mean(subject_accuracy))

# Programmer: Mehrdad Kashefi
# Classification using CNN

# Importing libraries
import scipy.io
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

num_subject = 9
subject_acc_train = np.zeros((num_subject, 1))
subject_acc_test = np.zeros((num_subject, 1))
for subj in range(num_subject):
    # Loading Data and Label
    mat = scipy.io.loadmat('data_subjet_'+str(subj+1))
    Data = mat['data_processed']
    print("The Shape of input is ", Data.shape)

    Label = mat['label_processed']
    # Shuffling Data
    [Data, Label] = shuffle(Data, Label)
    # K-Fold
    NumFold = 10
    kf = KFold(n_splits=NumFold, random_state=None, shuffle=False)
    kf.get_n_splits(Data)
    print(kf)
    Results = np.zeros((NumFold, 2))
    Results_train = np.zeros((NumFold, 2))
    Accuracy_test = np.zeros((NumFold, 1))
    Accuracy_train = np.zeros((NumFold, 1))
    FoldCount = 0
    RegVal = 0.01
    for train_index, test_index in kf.split(Data):

        # print("TRAIN:", train_index, "TEST:", test_index)  Print Folds
        x_train, x_test = Data[train_index], Data[test_index]
        y_train, y_test = Label[train_index], Label[test_index]

        TrueLabel = y_test
        PredictLabel = np.zeros((len(y_test), 1))
        prediction_label_train = np.zeros((len(y_train), 1))

        # y_train = to_categorical(y_train, num_classes=2)
        # y_test = to_categorical(y_test, num_classes=2)

        #  kernel_regularizer=regularizers.l2(RegVal)

        model = Sequential()
        model.add(Permute((2, 1), input_shape=(93, 32)))
        model.add(Flatten())
        model.add(Dense(900, activation='relu', use_bias=True))
        model.add(Dense(500, activation='relu', use_bias=True))
        model.add(Dense(200, activation='relu', use_bias=True))
        model.add(Dense(100, activation='relu', use_bias=True))
        model.add(Dense(40, activation='relu', use_bias=True))
        model.add(Dense(15, activation='relu', use_bias=True))
        model.add(Dense(5, activation='relu', use_bias=True))
        model.add(Dense(1, activation='sigmoid', use_bias=True))


        # Plotting layers
        # Uncomment this line if you have graphvis and pydot installed
        # plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

        # Opt = SGD(lr=0.001, decay=1e-6, momentum=0.01, nesterov=False)
        Opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        # Opt = Adagrad(lr=0.01, epsilon=None, decay=0.0)
        model.compile(loss='binary_crossentropy', optimizer=Opt, metrics=['accuracy'])

        callback = [EarlyStopping(monitor='val_accuracy', min_delta=0, patience=50, verbose=0, mode='auto', baseline=None,
                                  restore_best_weights=True)]
        model.fit(x_train, y_train, validation_data=(x_test, y_test), verbose=1, shuffle=True, batch_size=None, epochs=200, callbacks=callback)
        # Validating the model
        Results[FoldCount, :] = model.evaluate(x_test, y_test)
        Results_train[FoldCount, :] = model.evaluate(x_train, y_train)
        Prediction = model.predict(x_test)
        prediction_training = model.predict(x_train)

        # Hard Thresholding Test Prediction
        for i in range(len(Prediction)):
            if Prediction[i, 0] >= 0.5:
                PredictLabel[i, 0] = 1
            else:
                PredictLabel[i, 0] = 0

        # Hard Thresholding Train Prediction

        for i in range(len(prediction_training)):
            if prediction_training[i, 0] >= 0.5:
                prediction_label_train[i, 0] = 1
            else:
                prediction_label_train[i, 0] = 0

        cnf_matrix = confusion_matrix(y_train, prediction_label_train)
        print("The", str(FoldCount), "th Fold Confusion Matrix for Train data is ")
        print(cnf_matrix)
        Acc = sum(y_train == prediction_label_train) / len(prediction_label_train)
        print("The", str(FoldCount), "th  Train accuracy for this fold is ", Acc)
        Accuracy_train[FoldCount, 0] = Acc

        cnf_matrix = confusion_matrix(TrueLabel, PredictLabel)
        print("The", str(FoldCount), "th Fold Confusion Matrix for test data is ")
        print(cnf_matrix)
        Acc = sum(TrueLabel == PredictLabel)/len(PredictLabel)
        print("The", str(FoldCount), "th  Test accuracy for this fold is ", Acc)
        Accuracy_test[FoldCount, 0] = Acc

        FoldCount = FoldCount + 1

    # printing Final Values
    print("===============+++++++++++++++++============")
    print("The results for Test data averagely is ", np.mean(Results, 0))
    print("The results for Training data averagely is", np.mean(Results_train, 0))
    print("===============+++++++++++++++++============")
    print("Train Accuracy for each fold is ")
    print(np.transpose(Accuracy_train))
    print("==============")
    print("Test Accuracy for each fold is ")
    print(np.transpose(Accuracy_test))
    print("===============+++++++++++++++++============")
    print("The Final Train Accuracy is ", np.mean(Accuracy_train))
    print("The Final Test Accuracy is ", np.mean(Accuracy_test))

    subject_acc_test[subj, 0]= np.mean(Accuracy_test)
    subject_acc_train[subj, 0] = np.mean(Accuracy_train)

print("===============+++++++++++++++++============")
print("Train Accuracy for each Subject is ")
print(np.transpose(subject_acc_train))
print("==============")
print("Test Accuracy for each fold is ")
print(np.transpose(subject_acc_test))
print("===============+++++++++++++++++============")
print("The Final Train Accuracy for each subject is ", np.mean(subject_acc_train))
print("The Final Test Accuracy for each subject is ", np.mean(subject_acc_test))

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, Embedding, AlphaDropout
from keras.callbacks import Callback, ModelCheckpoint
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score


pd.options.mode.chained_assignment = None

class roc_callback(Callback):

    def __init__(self, training_data, validation_data):
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc_val: %s \n' % (str(round(roc_val, 5))), end=100 * ' ' + '\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


def FitDuLieuVaoMoHinhCNN(x_train, y_train, x_valid, y_valid, params):

    conv_dropout_1 = params['conv_dropout_1']
    conv_dropout_2 = params['conv_dropout_2']
    conv_dropout_3 = params['conv_dropout_3']
    conv_dropout_4 = params['conv_dropout_4']
    conv_dropout_5 = params['conv_dropout_5']
    dense_dropout = params['dense_dropout']
    dense_dim = params['dense_dim']
    optimizer = params['optimizer']
    batch_size = params['batch_size']
    epochs = params['epochs']
    callbacks = params['callbacks']

    input_length = 4096

    model = Sequential()
    model.add(Embedding(256, 16, input_length=input_length))
    model.add(Dropout(conv_dropout_1))
    model.add(Conv1D(48, 32, strides=4, padding='same', dilation_rate=1, activation='relu', use_bias=True,
                     kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(Dropout(conv_dropout_2))
    model.add(Conv1D(96, 32, strides=4, padding='same', dilation_rate=1, activation='relu', use_bias=True,
                     kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(Dropout(conv_dropout_3))
    model.add(MaxPooling1D(pool_size=4, strides=None, padding='valid'))
    model.add(Conv1D(128, 16, strides=8, padding='same', dilation_rate=1, activation='relu', use_bias=True,
                     kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(Dropout(conv_dropout_4))
    model.add(Conv1D(192, 16, strides=8, padding='same', dilation_rate=1, activation='relu', use_bias=True,
                     kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(Dropout(conv_dropout_5))

    model.add(Flatten())

    model.add(Dense(dense_dim, activation='selu'))
    model.add(Dropout(dense_dropout))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, validation_split=0.1, batch_size=batch_size, epochs=epochs, validation_data=(x_valid, y_valid),
              callbacks=callbacks)

    return model

def TienXuLyFileTrain(File_train):
    File_train = "data.csv"

    DeepLearningFiletrain = pd.read_csv(File_train, sep = "|")
    DeepLearningFiletrain = DeepLearningFiletrain.drop(['FileName','MD5'],axis=1)

    DeepLearningFiletrain.to_csv("DeepLearningFiletrain.csv", index = False)


    DeepLearningFiletrain = pd.read_csv("DeepLearningFiletrain.csv", skiprows=1, header=None)

    row_count, column_count = DeepLearningFiletrain.shape

    for My_row in range(row_count):
        for My_column in range(column_count):
            if((DeepLearningFiletrain[My_column][My_row]) >= 256):
                DeepLearningFiletrain[My_column][My_row] = 255

    DeepLearningFiletrain.to_csv("DeepLearningFiletrain.csv", index = False, header = False)

    f_append_data = open("DeepLearningFiletrain.csv", "r")

    lines = f_append_data.readlines()

    line = lines[0]

    print(line)

    i = line.count(",")

    need = 4096 - (i+1)

    line = line.replace("\n", "")

    for i in range(need):
        line += ",0"

    line += "\n"
    lines[0] = line

    f_append_data.close()

    f_append_data = open("DeepLearningFiletrain.csv", "w")
    f_append_data.writelines(lines)
        


def TienXuLyDuLieu(File_train, File_Label):

    TienXuLyFileTrain(File_train)
    train = pd.read_csv("DeepLearningFiletrain.csv", header=None, dtype=np.float16 )

    X_raw = train.values
    dimension_X = (~np.isnan(X_raw)).sum(
    	1)

    train.fillna(0, inplace=True)

    train.replace(np.inf, 0, inplace=True)
    train = train.astype(np.int16)

    train.to_csv("test.csv")


    label = pd.read_csv(File_Label, sep = '|', dtype={'ID': np.int32, 'label': np.int8})

    train_labels = pd.concat([train, label], axis = 1)	
    train_labels = train_labels.assign(num_dim=pd.Series(dimension_X))

    train_labels.sort_values(by='num_dim', ascending=False, inplace=True)


    train = train_labels.drop(['ID', 'label', 'num_dim'], axis=1)

    new_X = train.values

    new_Y = train_labels['label'].values

    return new_X, new_Y



def TrainDuLieu(X, y, model_number=0, random_state=0):


    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=random_state)
    for train_index, valid_index in sss.split(X, y):
        x_train, x_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]

    checkpoint_filepath = "model_{}".format(model_number) + "-{epoch:02d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_acc', verbose=1, save_weights_only=True, mode='max')

    callbacks_list = [checkpoint, roc_callback(training_data=(x_train, y_train), validation_data=(x_valid, y_valid))]

    params_for_validating_model = {
        'batch_size': 128,
        'conv_dropout_1': 0.2,
        'conv_dropout_2': 0.2,
        'conv_dropout_3': 0.2,
        'conv_dropout_4': 0.2,
        'conv_dropout_5': 0.2,
        'dense_dim': 64,
        'dense_dropout': 0.5,
        'epochs': 50,
        'optimizer': 'adam',
        'callbacks': callbacks_list
    }

    validated_model = FitDuLieuVaoMoHinhCNN(x_train, y_train, x_valid, y_valid, params_for_validating_model)


def main():
	print("MO HINH CNN")

if __name__ == '__main__':
	main()

X, y = TienXuLyDuLieu("data.csv", "label.csv")
#print(X, y)
TrainDuLieu(X, y, model_number=0, random_state=0)

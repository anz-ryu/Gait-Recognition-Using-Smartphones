"""Human activity recognition using smartphones dataset and an LSTM RNN."""

# https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition

# The MIT License (MIT)
#
# Copyright (c) 2016 Guillaume Chevalier
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Also thanks to Zhao Yu for converting the ".ipynb" notebook to this ".py"
# file which I continued to maintain.

# Note that the dataset must be already downloaded for this script to work.
# To download the dataset, do:
#     $ cd data/
#     $ python download_dataset.py


import tensorflow as tf
from tensorflow.keras import Model, layers
import os
import numpy as np


# Load "X" (the neural network's training and testing inputs)

def changex(xchanged):
    X_signals = []
    for l in xchanged:
        arr=l.flatten()
        X_signals.append(arr)
    return X_signals
def load_X(path):
    X_signals = []
    files = os.listdir(path)
    for my_file in files:
        fileName = os.path.join(path,my_file)
        file = open(fileName, 'r')
        X_signals.append(
            [np.array(cell, dtype=np.float32) for cell in [
                row.strip().split(' ') for row in file
            ]]
        )
        file.close()
        #X_signals = 6*totalStepNum*128
    X_signals = np.transpose(np.array(X_signals), (1, 0, 2))#(totalStepNum*6*128)
    return X_signals.reshape(-1,6,256,1)#(totalStepNum*6*128*1)

def load_y(y_path):
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()
    # Substract 1 to each output class for friendly 0-based indexing
    y_ = y_ - 1
    #one_hot
    y_ = y_.reshape(len(y_))
    n_values = 2
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing

    Note: it would be more interesting to use a HyperOpt search space:
    https://github.com/hyperopt/hyperopt
    """

    def __init__(self, X_train, X_test):
        # Input data
        self.n_layers = 1   # nb of layers
        self.train_count = len(X_train)  # 7352 training series
        self.test_data_count = len(X_test)  # 2947 testing series
        self.n_steps = 32 # 128 time_steps per series

        # Training
        self.learning_rate = 0.001
        self.lambda_loss_amount = 0.0015
        self.training_epochs = 10
        self.batch_size = 512

        # LSTM structure
        self.n_inputs = 128  # Features count is of 9: 3 * 3D sensors features over time
        self.n_hidden = 64  # nb of neurons inside the neural network
        self.n_classes = 2  # Final output classes
        self.W = {
            'hidden': tf.Variable(tf.random.normal([self.n_inputs, self.n_hidden])),
            'output': tf.Variable(tf.random.normal([self.n_hidden, self.n_classes]))
        }
        self.biases = {
            'hidden': tf.Variable(tf.random.normal([self.n_hidden], mean=1.0)),
            'output': tf.Variable(tf.random.normal([self.n_classes]))
        }


class HAR_Model(Model):
    def __init__(self, config):
        super(HAR_Model, self).__init__()
        
        # CNN layers
        self.conv1 = layers.Conv2D(32, (1, 9), strides=(1, 2), padding='same',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.pool1 = layers.MaxPool2D((1, 2), strides=(1, 2), padding='valid')
        self.conv2 = layers.Conv2D(64, (1, 3), strides=(1, 1), padding='same',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.pool2 = layers.MaxPool2D((1, 2), strides=(1, 2), padding='valid')
        self.conv3 = layers.Conv2D(128, (6, 1), strides=(1, 1), padding='valid', activation='relu')
        
        # LSTM layers
        self.lstm1 = layers.LSTM(config.n_hidden, return_sequences=True, 
                                dropout=0.25, recurrent_dropout=0.25)
        self.lstm2 = layers.LSTM(config.n_hidden)
        
        # Output layer
        self.dense = layers.Dense(config.n_classes)

    def call(self, inputs):
        x1, x2 = inputs
        
        # Process first input
        h1 = self.conv1(x1)
        h1 = self.pool1(h1)
        h1 = self.conv2(h1)
        h1 = self.pool2(h1)
        h1 = self.conv3(h1)
        
        # Process second input
        h2 = self.conv1(x2)
        h2 = self.pool1(h2)
        h2 = self.conv2(h2)
        h2 = self.pool2(h2)
        h2 = self.conv3(h2)
        
        # Reshape and concatenate
        t1 = tf.reshape(h1, [-1, 16, 128])
        t2 = tf.reshape(h2, [-1, 16, 128])
        ct = tf.concat([t1, t2], axis=1)
        
        # LSTM layers
        x = self.lstm1(ct)
        x = self.lstm2(x)
        
        return self.dense(x)


def one_hot(y_):
    """
    Function to encode output labels from number indexes.

    E.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    """
    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


if __name__ == "__main__":

    # -----------------------------
    # Step 1: load and prepare data
    # -----------------------------

    # Those are separate normalised input features for the neural network
    INPUT_SIGNAL_TYPES = [
        "_acc_x",
        "_acc_y",
        "_acc_z",
        "_gyr_x",
        "_gyr_y",
        "_gyr_z",
    ]

    DATA_PATH = "data/"
    DATASET_PATH = "data/"
    print("\n" + "Dataset is now located at: " + DATASET_PATH)
    TRAIN = "train/"
    TEST = "test/"

    X_train_signals_paths = [
        DATASET_PATH + TRAIN + "data/"  + "train" + signal + '.txt' for signal in INPUT_SIGNAL_TYPES
    ]
    X_test_signals_paths = [
        DATASET_PATH + TEST + "data/" + "test" + signal + '.txt' for signal in INPUT_SIGNAL_TYPES
    ]


    y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
    y_test_path = DATASET_PATH + TEST + "y_test.txt"
    X_train = load_X('./data/train/data/')
    X_test = load_X('./data/test/data/')

    train_label = load_y(y_train_path)
    test_label = load_y(y_test_path)

    # -----------------------------------
    # Step 2: define parameters for model
    # -----------------------------------

    config = Config(X_train, X_test)
    print("Some useful info to get an insight on dataset's shape and normalisation:")
    print("features shape, labels shape, each features mean, each features standard deviation")
    print(X_test.shape, test_label.shape,
          np.mean(X_test), np.std(X_test))
    print("the dataset is therefore properly normalised, as expected.")

    # ------------------------------------------------------
    # Step 3: Let's get serious and build the neural network
    # ------------------------------------------------------

    # モデルの構築
    model = HAR_Model(config)
    
    # オプティマイザーと損失関数の定義
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=config.learning_rate)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    
    # モデルのコンパイル
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy']
    )
    
    # データの準備
    train_x1 = X_train[:,:,:128]
    train_x2 = X_train[:,:,128:]
    test_x1 = X_test[:,:,:128]
    test_x2 = X_test[:,:,128:]
    
    # 学習率スケジューラーの追加
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # モデルの訓練
    history = model.fit(
        [train_x1, train_x2], 
        train_label,
        batch_size=config.batch_size,
        epochs=config.training_epochs,
        validation_data=([test_x1, test_x2], test_label),
        callbacks=[lr_scheduler, early_stopping]
    )
    
    # モデルの保存
    model.save('lstm_model')

    # ------------------------------------------------------------------
    # Step 5: Training is good, but having visual insight is even better
    # ------------------------------------------------------------------

    # Note: the code is in the .ipynb and in the README file
    # Try running the "ipython notebook" command to open the .ipynb notebook

    # ------------------------------------------------------------------
    # Step 6: And finally, the multi-class confusion matrix and metrics!
    # ------------------------------------------------------------------

    # Note: the code is in the .ipynb and in the README file
    # Try running the "ipython notebook" command to open the .ipynb notebook

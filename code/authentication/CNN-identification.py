import tensorflow as tf
import numpy as np
import os
import random

# GPUメモリ設定
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def load_X(path):
    X_signals = []
    files = os.listdir(path)
    files.sort(key=str.lower)
    #['train_acc_x.txt', 'train_acc_y.txt', 'train_acc_z.txt', 'train_gyr_x.txt', 'train_gyr_y.txt', 'train_gyr_z.txt']
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
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

# モデルの定義
def create_model():
    inputs = tf.keras.Input(shape=(6, 256, 1))
    
    # 第1畳み込みブロック
    x = tf.keras.layers.Conv2D(32, (1, 9), strides=(1, 2), padding='same')(inputs)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(x)
    
    # 第2畳み込みブロック
    x = tf.keras.layers.Conv2D(64, (1, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.ReLU()(x)
    
    # 第3畳み込みブロック
    x = tf.keras.layers.Conv2D(128, (1, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(x)
    
    # 第4畳み込みブロック
    x = tf.keras.layers.Conv2D(128, (6, 1), strides=(1, 1), padding='valid')(x)
    x = tf.keras.layers.ReLU()(x)
    
    # 全結合層
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# メイン処理
batch_size = 512
epochs = 10

# データの読み込み
X_train = load_X('./data/train/data')
print(f"X_train shape: {X_train.shape}")
train_label = load_y('./data/train/y_train.txt')
print(f"train_label shape: {train_label.shape}")
X_test = load_X('./data/test/data')
test_label = load_y('./data/test/y_test.txt')

# モデルの構築
model = create_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# チェックポイントの設定
checkpoint_path = "./cnn_ckpt/model"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

# 学習履歴を保存するためのカスタムコールバック
class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, filename):
        super(CustomCallback, self).__init__()
        self.file = open(filename, 'w')
    
    def on_epoch_end(self, epoch, logs=None):
        self.file.write(f'Epoch {epoch} - loss: {logs["loss"]:.4f} - accuracy: {logs["accuracy"]:.4f} '
                       f'- val_loss: {logs["val_loss"]:.4f} - val_accuracy: {logs["val_accuracy"]:.4f}\n')
    
    def on_train_end(self, logs=None):
        self.file.close()

# 既存のチェックポイントがあれば読み込む
if os.path.exists('./cnn_ckpt'):
    model.load_weights(tf.train.latest_checkpoint('./cnn_ckpt/'))

# モデルの学習
history = model.fit(
    X_train, train_label,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_test, test_label),
    callbacks=[
        checkpoint_callback,
        CustomCallback('./result_cnn.txt')
    ],
    shuffle=True
)

# 最終的な評価
best_accuracy = max(history.history['val_accuracy'])
print(f"Best accuracy: {best_accuracy}")

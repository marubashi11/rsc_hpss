from program.CNN.AI_structure import CNN_structure as cnn
from program.CNN.dataset import CNN_dataset_for_image as md
from program.CNN.etc import learning_rate as scheduler
from keras.callbacks import EarlyStopping, LearningRateScheduler, Callback, ModelCheckpoint
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import os

# def custom_recall(y_true, y_pred):
#     # 正解と予測のtensorからrecallを計算.
#     true_pos = tf.keras.backend.sum(y_true * y_pred)
#     total_pos = tf.keras.backend.sum(y_true)
#     return true_pos / (total_pos + tf.keras.backend.epsilon())

''''''
set_num = '1'
''''''

session_config = tf.compat.v1.ConfigProto()
session_config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=session_config)

H_train_dir = "/home/marubashi/ドキュメント/HPSS_test_Python/dataset_cycle/dataset" + set_num + "/train/image/H~/"
P_train_dir = "/home/marubashi/ドキュメント/HPSS_test_Python/dataset_cycle/dataset" + set_num + "/train/image/P/"
O_train_dir = "/home/marubashi/ドキュメント/HPSS_test_Python/dataset_cycle/dataset" + set_num + "/train/image/O~/"
H_test_dir = "/home/marubashi/ドキュメント/HPSS_test_Python/dataset_cycle/dataset" + set_num + "/test/image/H~/"
P_test_dir = "/home/marubashi/ドキュメント/HPSS_test_Python/dataset_cycle/dataset" + set_num + "/test/image/P/"
O_test_dir = "/home/marubashi/ドキュメント/HPSS_test_Python/dataset_cycle/dataset" + set_num + "/test/image/O~/"
model_dir = "/home/marubashi/ドキュメント/HPSS_test_Python/dataset_cycle/dataset" + set_num + "/model_OHP/"

# 画像の設定.
height = 224
width = 224
channels = 3

loss_func = "categorical_crossentropy"
n_categories = 4

lr = 0.001
epochs = 20  # 最大エポック数.

# データセットの作成
H_train_x = []
H_train_y = []
P_train_x = []
P_train_y = []
O_train_x = []
O_train_y = []
H_test_x = []
H_test_y = []
P_test_x = []
P_test_y = []
O_test_x = []
O_test_y = []
# 訓練データ作成.
print('loading "H_train" ...')
H_train_x, H_train_y = md.make_hpss_train_dataset(H_train_dir,
                                                  H_train_x,
                                                  H_train_y,
                                                  classification_type='a',
                                                  channels=channels)
print('loading "P_train" ...')
P_train_x, _ = md.make_hpss_train_dataset(P_train_dir,
                                          P_train_x,
                                          P_train_y,
                                          classification_type='a',
                                          channels=channels)
print('loading "O_train" ...')
O_train_x, _ = md.make_hpss_train_dataset(O_train_dir,
                                          O_train_x,
                                          O_train_y,
                                          classification_type='a',
                                          channels=channels)
# 評価データ作成.
print('loading "H_test" ...')
H_test_x, H_test_y = md.make_hpss_train_dataset(H_test_dir,
                                                H_test_x,
                                                H_test_y,
                                                classification_type='a',
                                                channels=channels)
print('loading "P_test" ...')
P_test_x, _ = md.make_hpss_train_dataset(P_test_dir,
                                         P_test_x,
                                         P_test_y,
                                         classification_type='a',
                                         channels=channels)
print('loading "O_test" ...')
O_test_x, _ = md.make_hpss_train_dataset(O_test_dir,
                                         O_test_x,
                                         O_test_y,
                                         classification_type='a',
                                         channels=channels)

# model
model = cnn.VGG16_multi(height, width, channels, n_categories)
model.summary()

model.compile(optimizer=Adam(lr=lr),
              loss=loss_func,
              metrics=['accuracy'])
# metrics=['accuracy', custom_recall])

step_decay = scheduler.StepDecay(initAlpha=lr, factor=0.5, dropEvery=25)
cp_cb = ModelCheckpoint(filepath=os.path.join(model_dir, "model_OHP-epoch{epoch:02d}-acc{val_acc:.2f}.hdf5"),
                        monitor='val_loss',
                        save_best_only=True,
                        save_weights_only=True,
                        verbose=1)
es_cb = EarlyStopping(monitor='loss', patience=3, verbose=1)
lr_cb = LearningRateScheduler(step_decay)

model.fit([O_train_x, H_train_x, P_train_x], H_train_y,
          epochs=epochs,
          batch_size=32,
          validation_data=([O_test_x, H_test_x, P_test_x], H_test_y),
          callbacks=[cp_cb, es_cb, lr_cb])

json_string = model.to_json()
open(os.path.join(model_dir, "OHP_VGG" + ".json"), "w").write(json_string)

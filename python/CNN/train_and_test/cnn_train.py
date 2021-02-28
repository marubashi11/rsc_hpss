from program.CNN.AI_structure import CNN_structure as cnn
from program.CNN.dataset import CNN_dataset_for_image as md
from program.CNN.etc import learning_rate as scheduler
import keras.backend as K
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, Callback
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf
import os
from sklearn.metrics import f1_score

session_config = tf.compat.v1.ConfigProto()
session_config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=session_config)

''''''
# 使用するデータセットの種類("cycle", "shift").
split_type = 'cycle'
# 使用するデータセットの番号(1 ~ 5).
set_num = '1'
# 使用するスペクトログラム('O', 'H', 'P', 'O~', 'H~', 'P~').
image_type = 'P'
# 分類する呼吸音('w': wheezeの有無, 'c': crackleの有無, 'a': 4種類の呼吸音).
c_type = 'a'
''''''
# 画像の設定.
height = 224
width = 224
channels = 3
# モデルと重みを保存するディレクトリ.
model_dir = "/home/marubashi/ドキュメント/HPSS_test_Python/dataset_" + split_type + "/dataset" + set_num + "/model/"


# 加重クロスエントロピー
def loss(y_true, y_pred):
    def weighted_categorical_cross_entropy(y_true, y_pred):
        w = tf.reduce_sum(y_true) / tf.cast(tf.size(y_true), tf.float32)
        loss = w * tf.nn.softmax_cross_entropy_with_logits(onehot_labels, logits)
        return loss
    return weighted_categorical_cross_entropy


def custom_recall(y_true, y_pred):
    # 正解と予測のtensorからrecallを計算.
    true_pos = K.sum(y_true * y_pred)
    total_pos = K.sum(y_true)
    return true_pos / (total_pos + K.epsilon())


def normalize_y_pred(y_pred):
    return K.one_hot(K.argmax(y_pred), y_pred.shape[-1])


def crackle_recall(y_true, y_pred):
    class_label = 1
    y_pred = normalize_y_pred(y_pred)
    tp = K.cast(K.equal(y_true[:, class_label] + y_pred[:, class_label], 2), K.floatx())
    return K.sum(tp) / (K.sum(y_true[:, class_label]) + K.epsilon())


def crackle_precision(y_true, y_pred):
    class_label = 1
    y_pred = normalize_y_pred(y_pred)
    tp = K.cast(K.equal(y_true[:, class_label] + y_pred[:, class_label], 2), K.floatx())
    return K.sum(tp) / (K.sum(y_pred[:, class_label]) + K.epsilon())


def crackle_f1(y_true, y_pred):
    class_label = 1
    y_pred = normalize_y_pred(y_pred)
    tp = K.cast(K.equal(y_true[:, class_label] + y_pred[:, class_label], 2), K.floatx())
    precision = K.sum(tp) / (K.sum(y_pred[:, class_label]) + K.epsilon())
    recall = K.sum(tp) / (K.sum(y_true[:, class_label]) + K.epsilon())
    print("pre: ", precision[0])
    print("rec: ", recall[0])
    return (2 * precision * recall) / (precision + recall + K.epsilon())


def wheeze_f1(y_true, y_pred):
    class_label = 2
    y_pred = normalize_y_pred(y_pred)
    tp = K.cast(K.equal(y_true[:, class_label] + y_pred[:, class_label], 2), K.floatx())
    precision = K.sum(tp) / (K.sum(y_pred[:, class_label]) + K.epsilon())
    recall = K.sum(tp) / (K.sum(y_true[:, class_label]) + K.epsilon())
    return (2 * precision * recall) / (precision + recall + K.epsilon())


# F1スコアを評価関数に使用する場合, コールバックで計算する方が正確.
class F1CallBack(Callback):
    def __init__(self, model, x_val, y_val):
        super().__init__()
        self.model = model
        self.x_val = x_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict(self.x_val)
        # f1_val = f1_score(self.y_val, np.round(pred), average='weighted')
        f1_val = crackle_f1(self.y_val, np.round(pred))
        # # F1スコアがしきい値以上の場合に重みを保存.
        # if f1_val >= 0.5:
        #     check_point = os.path.join(model_dir, "model-" + image_type + c_type
        #                                + "-epoch{epoch:02d}-cf1" + str(f1_val) + "-.hdf5")
        #     self.model.save_weights(check_point)
        #     print("saved.")


train_dir = "/home/marubashi/ドキュメント/HPSS_test_Python/dataset_" + split_type + "/dataset" + set_num \
            + "/train/image/" + image_type + "/"
test_dir = "/home/marubashi/ドキュメント/HPSS_test_Python/dataset_" + split_type + "/dataset" + set_num \
           + "/test/image/" + image_type + "/"

if c_type == 'c' or c_type == 'w':
    loss_func = "binary_crossentropy"
    n_categories = 2
else:
    loss_func = "categorical_crossentropy"
    # loss_func = "cosine_proximity"
    n_categories = 4

lr = 0.001      # 初期学習率. やや高めに設定してみる(lr=0.0001 -> 0.001).
epochs = 20     # 最大エポック数.

# データセットの作成
x_train = []
y_train = []
x_test = []
y_test = []
print("Loading training data...")
x_train, y_train = md.make_hpss_train_dataset(train_dir, x_train, y_train,
                                              classification_type=c_type,
                                              channels=channels)
print("Loading test data...")
x_test, y_test = md.make_hpss_train_dataset(test_dir, x_test, y_test,
                                            classification_type=c_type,
                                            channels=channels)

# model
model = cnn.VGG16_custom(height, width, channels, n_categories)
model.summary()

model.compile(optimizer=Adam(lr=lr),
              loss=loss_func,
              metrics=['accuracy', crackle_recall])
# metrics=['accuracy', custom_recall])

step_decay = scheduler.StepDecay(initAlpha=lr, factor=0.5, dropEvery=25)
cp_cb = ModelCheckpoint(filepath=os.path.join(model_dir, "model-" + image_type + c_type
                                              + "-epoch{epoch:02d}-acc{val_acc:.2f}.hdf5"),
                        monitor='val_loss',
                        mode='min',
                        save_best_only=True,
                        save_weights_only=True,
                        verbose=1)
es_cb = EarlyStopping(monitor='loss', patience=3, verbose=1)
lr_cb = LearningRateScheduler(step_decay)
f1_cb = F1CallBack(model, x_test, y_test)

model.fit(x_train, y_train,
          epochs=epochs,
          batch_size=32,
          validation_data=(x_test, y_test),
          callbacks=[cp_cb, es_cb, lr_cb])

json_string = model.to_json()
open(os.path.join(model_dir, image_type + "_VGG" + ".json"), "w").write(json_string)

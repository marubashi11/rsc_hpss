# coding:UTF-8
# 訓練したCNNのテストを行う. softmaxを用いた場合のConfusion Matrixの確認用.
from program.CNN.dataset import CNN_dataset_for_image as md
from keras.models import model_from_json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import tensorflow as tf
import os

session_config = tf.compat.v1.ConfigProto()
session_config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=session_config)

''''''
# 使用するデータセットの種類("cycle", "shift").
split_type = 'cycle'
# 使用するデータセットの番号(1 ~ 5).
set_num = '1'
# 使用するスペクトログラム.
image_type = 'P'
# 分類する呼吸音('w': wheezeの有無, 'c': crackleの有無, 'a': 4種類の呼吸音).
c_type = 'a'
# 使用する重みのepochとacc.
epc = "11"
acc = "0.60"
# 画像の設定.
height = 224
width = 224
channels = 3
''''''
# モデルと重みの指定.
model_dir = "/home/marubashi/ドキュメント/HPSS_test_Python/dataset_" + split_type + "/dataset" + set_num + "/model/"
model_filename = image_type + "_VGG.json"
weights_filename = "model-" + image_type + c_type + "-epoch" + epc + "-acc" + acc + ".hdf5"
# テストデータ用ディレクトリ.
test_dir = "/home/marubashi/ドキュメント/HPSS_test_Python/dataset_" + split_type + "/dataset" + set_num\
           + "/test/image/" + image_type + "/"

# データセットの作成
x_test = []
y_test = []
test_name = []

x_test, y_test, test_name = \
    md.make_hpss_test_dataset(test_dir, x_test, y_test, test_name, classification_type=c_type, channels=channels)

json_string = open(os.path.join(model_dir, model_filename)).read()
model = model_from_json(json_string)
model.load_weights(os.path.join(model_dir, weights_filename))

y_preds = model.predict(x_test, verbose=1)
# scores = []
# for x in y_preds:
#     scores.append(x[0])

y_pred_ = np.argmax(y_preds, axis=1)
y_test_ = np.argmax(y_test, axis=1)

print(accuracy_score(y_test_, y_pred_))
print(classification_report(y_test_, y_pred_))
print(confusion_matrix(y_test_, y_pred_))

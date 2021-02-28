# coding:UTF-8

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
set_num = '1'
# 使用する重みのepochとacc.
epc = "09"
acc = "0.60"
''''''

H_test_dir = "/home/marubashi/ドキュメント/HPSS_test_Python/dataset_cycle/dataset" + set_num + "/test/image/H~/"
P_test_dir = "/home/marubashi/ドキュメント/HPSS_test_Python/dataset_cycle/dataset" + set_num + "/test/image/P~/"
O_test_dir = "/home/marubashi/ドキュメント/HPSS_test_Python/dataset_cycle/dataset" + set_num + "/test/image/O~/"
model_dir = "/home/marubashi/ドキュメント/HPSS_test_Python/dataset_cycle/dataset" + set_num + "/model_OHP/"

model_filename = "OHP_VGG.json"
weights_filename = "model_OHP-epoch" + epc + "-acc" + acc + ".hdf5"

height = 224
width = 224
channels = 3

# データセットの作成
H_x = []
P_x = []
O_x = []
P_y = []
H_y = []
O_y = []

H_x, H_y = \
    md.make_hpss_train_dataset(H_test_dir, H_x, H_y, classification_type='a', channels=channels)
P_x, _ = \
    md.make_hpss_train_dataset(P_test_dir, P_x, P_y, classification_type='a', channels=channels)
O_x, _ = \
    md.make_hpss_train_dataset(O_test_dir, O_x, O_y, classification_type='a', channels=channels)

json_string = open(os.path.join(model_dir, model_filename)).read()
model = model_from_json(json_string)
model.load_weights(os.path.join(model_dir, weights_filename))

y_preds = model.predict([O_x, H_x, P_x], verbose=1)

scores = []

for x in y_preds:
    scores.append(x[0])

y_pred_ = np.argmax(y_preds, axis=1)
y_test_ = np.argmax(H_y, axis=1)

print(accuracy_score(y_test_, y_pred_))
print(classification_report(y_test_, y_pred_))
print(confusion_matrix(y_test_, y_pred_))

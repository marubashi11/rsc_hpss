# coding:UTF-8
# SVM入力用のCNN特徴量を作成し保存する.
from program.CNN.dataset import CNN_dataset_for_image as md
from keras import Model
from keras.models import model_from_json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import pickle
import umap

session_config = tf.compat.v1.ConfigProto()
session_config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=session_config)

''''''
# 使用するデータセットの種類("cycle", "shift").
split_type = "cycle"
# 使用するデータセットの番号(1 ~ 5).
set_num = '1'
# 使用するデータ("train", "test").
data_type = "train"
# 分類する呼吸音('w': wheezeの有無, 'c': crackleの有無, 'a': 4種類の呼吸音).
c_type = 'a'
# 使用する重みのepochとacc.
epc = "09"
acc = "0.60"
''''''
src_dir = "/home/marubashi/ドキュメント/HPSS_test_Python/dataset_" + split_type + "/dataset" + set_num + "/"\
          + data_type + "/image/"
O_dir = src_dir + "O~/"
H_dir = src_dir + "H~/"
P_dir = src_dir + "P/"
model_dir = "/home/marubashi/ドキュメント/HPSS_test_Python/dataset_" + split_type + "/dataset" + set_num + "/model_OHP/"
model_filename = "OHP_VGG.json"
weights_filename = "model_OHP" + "-epoch" + epc + "-acc" + acc + ".hdf5"

height = 224
width = 224
channels = 3

''' データセットの作成 '''
O_x = []
O_y = []
H_x = []
H_y = []
P_x = []
P_y = []
print("loading 'O' data...")
O_x, _ = md.make_hpss_train_dataset(O_dir, O_x, O_y, classification_type=c_type, channels=channels)
print("loading 'H' data...")
H_x, H_y = md.make_hpss_train_dataset(H_dir, H_x, H_y, classification_type=c_type, channels=channels)
print("loading 'P' data...")
P_x, _ = md.make_hpss_train_dataset(P_dir, P_x, P_y, classification_type=c_type, channels=channels)

''' モデルと重みの読み込み '''
json_string = open(os.path.join(model_dir, model_filename)).read()
model = model_from_json(json_string)
model.load_weights(os.path.join(model_dir, weights_filename))

''' 中間層出力(特徴量)を得るためのモデルを作成 '''
layer_name = 'GAP'
hidden_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

''' 訓練/テストデータに対するCNN特徴量を獲得 '''
hidden_output = hidden_model.predict([O_x, H_x, P_x], verbose=1)
# # 可視化.
# hid_co = umap.UMAP().fit(hidden_output)
# plt.scatter(hid_co.embedding_[:, 0], hid_co.embedding_[:, 1], c=np.argmax(y, axis=1), cmap='plasma')
# plt.colorbar()
# plt.savefig(model_dir + 'train_umap.png', bbox_inches='tight')

''' 正解ラベルを一次元化 '''
y_label_ = np.argmax(H_y, axis=1)

# ''' 精度スコアの表示 '''
# print(accuracy_score(y_label_, y_pred_))
# print(classification_report(y_label_, y_pred_))
# print(confusion_matrix(y_label_, y_pred_))

''' 特徴量と正解ラベルの保存 '''
pickle.dump(hidden_output, open(os.path.join(model_dir, "OHP-" + data_type + "_feature.txt"), "wb"))
# 正解ラベルのリストとファイル名のリストはHのみ保存.
pickle.dump(y_label_, open(os.path.join(model_dir, "OHP-" + data_type + "_label.txt"), "wb"))

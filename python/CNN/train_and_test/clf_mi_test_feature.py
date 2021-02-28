# coding:UTF-8
# OHP3種類のCNN特徴量を入力する分類器のテスト.
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import tensorflow as tf
import os
import pickle

session_config = tf.compat.v1.ConfigProto()
session_config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=session_config)

''''''
# 使用するデータセットの種類("cycle", "shift").
split_type = "cycle"
# 使用するデータセットの番号(1 ~ 5).
set_num = '1'
# 使用する分類器.("SVM", "LinReg", "LogReg", "RF", "LGBM")
classifier = "SVM"
''''''
# 分類器を読み込むディレクトリ.
model_dir = "/home/marubashi/ドキュメント/HPSS_test_Python/dataset_" + split_type + "/dataset" + set_num + "/model_OHP/"
# # 分類失敗したファイルのリストを保存するディレクトリ.
# f_dir = "/home/marubashi/ドキュメント/HPSS_test_Python/dataset_" + split_type + "/dataset" + set_num + "/"

''' OHPそれぞれのテストデータに対する特徴量を読み込み, それらをスタックしたリストを作成 '''
OHP_feature = pickle.load(open(os.path.join(model_dir, "OHP-test_feature.txt"), "rb"))

# ''' ファイル名リストの読み込み '''
# name_list = pickle.load(open(os.path.join(model_dir, "H~-test_name.txt"), "rb"))

''' 正解ラベルの読み込み '''
y_test_ = pickle.load(open(os.path.join(model_dir, "OHP-test_label.txt"), "rb"))

''' 分類器の読み込み '''
clf = pickle.load(open(os.path.join(model_dir, "trained_" + classifier + ".sav"), "rb"))

''' テストデータに対する予測ラベルリストの作成 '''
y_pred_ = clf.predict(OHP_feature)
# 一部の分類器は予測ラベルの小数点以下を四捨五入する必要がある.
if classifier == "LinReg" or classifier == "RF" or classifier == "LGBM":
    y_pred_ = np.round(y_pred_)

''' 結果表示 '''
print(accuracy_score(y_test_, y_pred_))
print(classification_report(y_test_, y_pred_))
print(confusion_matrix(y_test_, y_pred_))

# ''' 分類失敗したもののリストを作成し保存 '''
# f_list = []
#
# for i in range(len(y_test_)):
#     if y_test_[i] != y_pred_[i]:
#         name = name_list[i] + "_ans" + str(y_test_[i]) + "_pre" + str(y_pred_[i])
#         f_list.append(name)
#
# pickle.dump(f_list, open(os.path.join(f_dir, "f_list_" + set_num + ".txt"), "wb"))

# coding:UTF-8
# OHP3種類のCNN特徴量を入力する分類器のテスト.
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import numpy as np
import tensorflow as tf
import os
import pickle
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

session_config = tf.compat.v1.ConfigProto()
session_config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=session_config)

''''''
# 使用するデータセットの種類("cycle", "shift").
split_type = "cycle"
# 使用するデータセットの番号(1 ~ 5).
set_num = '1'
# 使用するスペクトログラム('O', 'H', 'P', 'O~', 'H~', 'P~').
image_type = 'P'
# 使用する分類器.("SVM", "LinReg", "LogReg", "RF", "LGBM")
classifier = "SVM"
''''''
# 分類器を読み込むディレクトリ.
model_dir = "/home/marubashi/ドキュメント/HPSS_test_Python/dataset_" + split_type + "/dataset" + set_num + "/model/"

''' CNN特徴量の読み込み '''
train_feature_list = pickle.load(open(os.path.join(model_dir, image_type + "-train_feature.txt"), "rb"))
test_feature_list = pickle.load(open(os.path.join(model_dir, image_type + "-test_feature.txt"), "rb"))

''' 正解ラベルの読み込み '''
y_train_ = pickle.load(open(os.path.join(model_dir, "H~-train_label.txt"), "rb"))
y_test_ = pickle.load(open(os.path.join(model_dir, "H~-test_label.txt"), "rb"))

''' 分類器の選択 '''
if classifier == "SVM":
    # SVM.
    clf = SVC(kernel="linear")
    # clf = SVC(kernel='rbf', C=2 ** 10, gamma=1.0)     # 線形カーネル以外を使用.
    # clf = OneVsRestClassifier(clf)                    # one-vs-rest戦略を使用.
elif classifier == "LinReg":
    # 線形回帰.
    clf = LinearRegression()
elif classifier == "LogReg":
    # ロジスティック回帰.
    clf = LogisticRegression()
elif classifier == "RF":
    # ランダムフォレスト.
    clf = RandomForestRegressor()
elif classifier == "LGBM":
    # LightGBM.
    clf = LGBMRegressor()
else:
    # それ以外の指定のときは強制的にone-vs-oneのSVM(線形カーネル)を使用.
    classifier = "SVM"
    clf = SVC(kernel="linear")

''' 分類器の訓練と保存 '''
clf.fit(train_feature_list, y_train_)
pickle.dump(clf, open(os.path.join(model_dir, image_type + "_trained_" + classifier + ".sav"), "wb"))

# ''' 分類器の読み込み '''
# clf = pickle.load(open(os.path.join(model_dir, image_type + "_trained_" + classifier + ".sav"), "rb"))

''' テストデータに対する予測ラベルリストの作成 '''
y_pred_ = clf.predict(test_feature_list)
# 一部の分類器は予測ラベルの小数点以下を四捨五入する必要がある.
if classifier == "LinReg" or classifier == "RF" or classifier == "LGBM":
    y_pred_ = np.round(y_pred_)

''' 結果表示 '''
print(accuracy_score(y_test_, y_pred_))
print(classification_report(y_test_, y_pred_))
print(confusion_matrix(y_test_, y_pred_))

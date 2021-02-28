# coding:UTF-8
# 保存された各モデルの中間層出力と正解ラベルを使って分類器の訓練を行う.
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
# 使用する分類器.("SVM", "LinReg", "LogReg", "RF", "LGBM")
classifier = "SVM"
''''''
# 分類器を保存するディレクトリ.
model_dir = "/home/marubashi/ドキュメント/HPSS_test_Python/dataset_" + split_type + "/dataset" + set_num + "/model_OHP/"

''' 訓練予測の読み込み '''
OHP_feature = pickle.load(open(os.path.join(model_dir, "OHP-train_feature.txt"), "rb"))
''' 正解ラベルの読み込み '''
y_train_ = pickle.load(open(os.path.join(model_dir, "OHP-train_label.txt"), "rb"))

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
''' 分類器の訓練 '''
clf.fit(OHP_feature, y_train_)
''' 分類器の保存 '''
pickle.dump(clf, open(os.path.join(model_dir, "trained_" + classifier + ".sav"), "wb"))

# ''' 精度スコアの表示 '''
# print(accuracy_score(y_label_, y_pred_))
# print(classification_report(y_label_, y_pred_))
# print(confusion_matrix(y_label_, y_pred_))

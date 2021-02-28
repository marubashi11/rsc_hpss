# CNNのデータセット作成用関数を書き込んでいるプログラム

from keras.utils import np_utils
from PIL import Image
import numpy as np
import glob
import os


def make_dataset(path, x, y, folder, height, width, n_categories, channels):
    for index, name in enumerate(folder):
        dir = path + name
        files = glob.glob(dir + "/*.png")
        files.sort()
        for i, file in enumerate(files):
            image = Image.open(file)
            if channels == 1:
                image = image.convert("L").convert("RGB")
            else:
                # エラーが出たため追加. RGBA(4ch)からRGB(3ch)への変換.
                image = image.convert("RGB")
            image = image.resize((height, width))
            data = np.asarray(image)
            x.append(data)
            y.append(index)
            # z.append(os.path.basename(file).rstrip(".png"))

    x = np.array(x)
    y = np.array(y)
    # z = np.array(z)
    if channels == 1:
        x = np.delete(x, [1, 2], axis=3)

    x = x.astype("float32")
    x = x / 255.0

    y = np_utils.to_categorical(y, n_categories)

    return x, y


def make_dataset_for_test(path, x, y, z, folder, height, width, n_categories, channels):
    for index, name in enumerate(folder):
        dir = path + name
        files = glob.glob(dir + "/*.png")
        files.sort()
        for i, file in enumerate(files):
            image = Image.open(file)
            if channels == 1:
                image = image.convert("L").convert("RGB")
            else:
                # エラーが出たため追加. RGBA(4ch)からRGB(3ch)への変換.
                image = image.convert("RGB")
            image = image.resize((height, width))
            data = np.asarray(image)
            x.append(data)
            y.append(index)
            z.append(os.path.basename(file).rstrip(".png"))

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    if channels == 1:
        x = np.delete(x, [1, 2], axis=3)

    x = x.astype("float32")
    x = x / 255.0

    y = np_utils.to_categorical(y, n_categories)

    return x, y, z


# HPSS用. 正解ラベルはclassification_typeによって場合分けする.
def make_hpss_train_dataset(path, x, y, classification_type='a', height=224, width=224, channels=3):
    folder = ["normal", "crackle", "wheeze", "crackle_wheeze"]
    n_categories = 2

    for index, name in enumerate(folder):
        dir = path + name
        files = glob.glob(dir + "/*.png")
        files.sort()
        for i, file in enumerate(files):
            image = Image.open(file)
            if channels == 1:
                image = image.convert("LA").convert("RGB")
            else:
                image = image.convert("RGB")
            image = image.resize((height, width))
            data = np.asarray(image)
            x.append(data)

            # classification_typeによって場合分けし, 正解ラベルを設定.
            if classification_type == 'w':
                if index == 2 or index == 3:
                    y.append(0)
                else:
                    y.append(1)
            elif classification_type == 'c':
                if index == 1 or index == 3:
                    y.append(0)
                else:
                    y.append(1)
            else:
                n_categories = len(folder)
                y.append(index)

    x = np.array(x)
    y = np.array(y)
    if channels == 1:
        x = np.delete(x, [1, 2], axis=3)
    x = x.astype("float32")
    x = x / 255.0

    y = np_utils.to_categorical(y, n_categories)

    return x, y


# 分類結果の分析のため名前も保存する.
def make_hpss_test_dataset(path, x, y, name_list, classification_type='a', height=224, width=224, channels=3):
    folder = ["normal", "crackle", "wheeze", "crackle_wheeze"]
    n_categories = 2

    for index, name in enumerate(folder):
        dir = path + name
        files = glob.glob(dir + "/*.png")
        files.sort()
        for i, file in enumerate(files):
            image = Image.open(file)
            if channels == 1:
                image = image.convert("LA").convert("RGB")
            else:
                image = image.convert("RGB")
            image = image.resize((height, width))
            data = np.asarray(image)
            x.append(data)
            name_list.append(os.path.basename(file).rstrip(".png"))

            # classification_typeによって場合分けし, 正解ラベルを設定.
            if classification_type == 'w':
                if index == 2 or index == 3:
                    y.append(0)
                else:
                    y.append(1)
            elif classification_type == 'c':
                if index == 1 or index == 3:
                    y.append(0)
                else:
                    y.append(1)
            else:
                n_categories = len(folder)
                y.append(index)

    x = np.array(x)
    y = np.array(y)
    name_list = np.array(name_list)
    if channels == 1:
        x = np.delete(x, [1, 2], axis=3)
    x = x.astype("float32")
    x = x / 255.0

    y = np_utils.to_categorical(y, n_categories)

    return x, y, name_list

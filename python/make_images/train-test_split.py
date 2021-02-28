# データセットをN分割し, N個の訓練-テストデータセットを作成する.
import copy
import glob
import itertools
import os
import shutil

# 実行する呼吸音クラス.
class_list = ['normal', 'crackle', 'wheeze', 'crackle_wheeze']
# 分割後のデータを保存するフォルダ.
dst_folder = "/home/marubashi/ドキュメント/HPSS_test_Python/dataset_cycle/"


def train_test_split(label):
    # 分割元のデータセット.
    src_path = "/home/marubashi/ドキュメント/HPSS_test_Python/audio_data/split_cycle/" + label + "/*"

    ''' srcのすべてのファイルをリスト化 '''
    files = glob.glob(src_path)

    ''' ファイルのN分割 '''
    N = 5       # 分割数.
    div_files = [files[i::N] for i in range(N)]

    ''' N分割した内の1つをテスト用, 残りを訓練用とするN個のデータセットを作成 '''
    for i in range(N):
        # N分割したデータのi番目をテスト用とする.
        test_files = copy.copy(div_files[i])
        # 残りは訓練用.
        train_files = copy.copy(div_files)
        del train_files[i]
        train_files = list(itertools.chain.from_iterable(train_files))

        # ファイルの保存(コピー).
        for ts in test_files:
            name = os.path.basename(ts)
            test_path = dst_folder + "dataset" + str(i + 1) + "/test/audio/" + label + "/" + name
            shutil.copyfile(ts, test_path)
        for tr in train_files:
            name = os.path.basename(tr)
            train_path = dst_folder + "dataset" + str(i + 1) + "/train/audio/" + label + "/" + name
            shutil.copyfile(tr, train_path)


# train_test_splitの一括実行.
for sound_class in class_list:
    print(sound_class)      # 進捗確認用.
    train_test_split(sound_class)

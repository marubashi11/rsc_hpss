# メディアンフィルタベースのHPSS. 一度に計4000枚程度の画像を開くとフリーズするので細かく分割して実行すべき.
import program.make_images.filter_function as filter
import glob
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal
import soundfile as sf

''''''
# 使用するデータ("train", "test").
data_type = 'train'
# HPSSを実行するクラス(フォルダ分けに使用).
class_list = ['normal']
# 保存するスペクトログラム(0: 保存しない, 1: 保存する).
save_fig = {'O': 0, 'H': 0, 'P': 1}
# 使用するデータセットの種類("cycle", "shift").
split_type = 'cycle'
# 使用するデータセットの番号(1 ~ 5).
set_num = '5'
''''''
# フォルダの指定.
src_folder = "/home/marubashi/ドキュメント/HPSS_test_Python/dataset_" + split_type + "/dataset" + set_num\
             + "/" + data_type + "/audio/"
spectrogram_dst_folder = "/home/marubashi/ドキュメント/HPSS_test_Python/dataset_" + split_type + "/dataset" + set_num\
                         + "/" + data_type + "/image/"
audio_dst_folder = "/home/marubashi/ドキュメント/HPSS_test_Python/audio_data/H_and_P/"


def hpss(file_name, label):
    # スペクトログラムの色設定.
    spec_color = "magma"

    ''' 音声ファイルの読み込み(+モノラルに変換) '''
    input_path = src_folder + label + "/" + file_name + ".wav"
    data, sr = librosa.load(input_path, sr=4000, mono=True)  # スペクトログラムの周波数軸は 0 ~ (sr / 2) [Hz]

    ''' 音声信号の表示 '''
    # plt.figure(figsize=(15, 5))
    # librosa.display.waveplot(data, sr)
    # aggを使用しているため以下の関数で表示できないらしい.
    # plt.plot(data)
    # plt.show()
    # # 図を保存.
    # sig_filename = filename + "_sig.png"
    # sig_filepath = "./diagrams/" + sig_filename
    # plt.savefig(sig_filepath)

    ''' 帯域制限 '''
    fp = np.array([50, 1800])  # 通過域端周波数[Hz]
    fs = np.array([10, 2000])  # 阻止域端周波数[Hz]
    gpass = 3  # 通過域端最大損失[dB]
    gstop = 40  # 阻止域端最小損失[dB]
    filt_data = filter.bandpass(data, sr, fp, fs, gpass, gstop)

    ''' STFT '''
    window_size = int(0.064 * sr)
    overlap_size = int(window_size / 8)
    amp_stft = np.abs(librosa.stft(filt_data,
                                   n_fft=window_size,
                                   hop_length=overlap_size,
                                   win_length=window_size,
                                   window=signal.get_window("hamming", window_size)))

    ''' スペクトログラムの作成 '''
    # 図の設定.
    fig_width = 224 / 100
    fig_height = 224 / 100
    # スペクトログラムの表示帯域[Hz]を制限.
    min_freq = 0
    max_freq = 1800
    if save_fig['O']:
        fig = plt.figure(figsize=(fig_width, fig_height))
        fig.add_axes([0, 0, 1, 1])  # [x0, y0, width(比率), height(比率)]
        librosa.display.specshow(librosa.amplitude_to_db(amp_stft, ref=np.max),
                                 y_axis='linear',
                                 x_axis='time',
                                 sr=sr,
                                 cmap=spec_color)
        plt.ylim(min_freq, max_freq)
        # スペクトログラムの保存.
        o_filepath = spectrogram_dst_folder + "O~/" + label + "/" + file_name + "_O.png"
        plt.savefig(o_filepath)

    ''' HPSS '''
    # h_data, p_data = librosa.effects.hpss(data)
    hp_margin = (5.0, 5.0)  # HPSSのマージン(>=1.0). 大きいほど分離が顕著になる.
    h_mask, p_mask = librosa.decompose.hpss(amp_stft,
                                            margin=hp_margin,
                                            kernel_size=15,
                                            power=2.0,
                                            mask=True)
    h_stft = amp_stft * h_mask
    p_stft = amp_stft * p_mask

    ''' Hスペクトログラムの作成 '''
    if save_fig['H']:
        fig = plt.figure(figsize=(fig_width, fig_height))
        fig.add_axes([0, 0, 1, 1])
        librosa.display.specshow(librosa.amplitude_to_db(h_stft, ref=np.max),
                                 y_axis='linear',
                                 x_axis='time',
                                 sr=sr,
                                 cmap=spec_color)
        plt.ylim(min_freq, max_freq)
        # Hスペクトログラムの保存.
        h_filepath = spectrogram_dst_folder + "H~/" + label + "/" + file_name + "_H.png"
        plt.savefig(h_filepath)

    ''' Pスペクトログラムの作成 '''
    if save_fig['P']:
        fig = plt.figure(figsize=(fig_width, fig_height))
        fig.add_axes([0, 0, 1, 1])
        librosa.display.specshow(librosa.amplitude_to_db(p_stft, ref=np.max),
                                 y_axis='linear',
                                 x_axis='time',
                                 sr=sr,
                                 cmap=spec_color)
        plt.ylim(min_freq, max_freq)
        # Pスペクトログラムの保存.
        p_filepath = spectrogram_dst_folder + "P~/" + label + "/" + file_name + "_P.png"
        plt.savefig(p_filepath)

    ''' H, Pそれぞれを音声ファイルとして保存 '''
    # hwav_path = audio_dst_folder + label + "/" + file_name + "_H.wav"
    # pwav_path = audio_dst_folder + label + "/" + file_name + "_P.wav"
    # sf.write(hwav_path, h_data, sr)
    # sf.write(pwav_path, p_data, sr)


# print("filename = ")
# name = input()
# hpss(name)

''' HPSSの一括実行 '''
for sound_class in class_list:
    print(sound_class)                # 進捗確認用.
    # 使用するwavファイルをすべて選択し格納.
    src_path = src_folder + sound_class + "/*"
    files = glob.glob(src_path)
    file_total = len(files)     # 進捗確認用.
    i = 1                       # 進捗確認用.
    # files内の各ファイルにHPSSを実行.
    for n in files:
        print("\r%d / %d" % (i, file_total), end='')     # 進捗確認用
        name = os.path.basename(n).rstrip(".wav")
        hpss(name, sound_class)
        i += 1                  # 進捗確認用.
    print()                     # 進捗確認用.

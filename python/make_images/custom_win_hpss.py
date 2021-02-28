# 窓幅大でH, 窓幅小でPを抽出するHPSSを行う.
import program.make_images.filter_function as filter
import glob
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal

''''''
# HPSSを実行するクラス. 一度に3000枚以上実行するとフリーズすることがある.
sound_classes = ['normal']
# 保存するスペクトログラムの選択(0: 保存しない, 1: 保存する).
save_fig = {'O': 0, 'H': 1, 'P': 0}
# フォルダの指定.
src_folder = "/home/marubashi/ドキュメント/HPSS_test_Python/dataset_cycle/dataset1/train/audio/"
spectrogram_dst_folder = "/home/marubashi/ドキュメント/HPSS_test_Python/dataset_cycle/dataset1/train/image/"
# スペクトログラムの表示帯域[Hz].
min_freq = 0
max_freq = 2000
# 図の大きさ設定.
fig_width = 224 / 100
fig_height = 224 / 100
# HPSSのマージン(>=1.0). 大きいほど分離が顕著になる.
hp_margin = (4.0, 5.0)
''''''


def hpss(file_name, sound_label):
    # スペクトログラムの色設定.
    spec_color = "magma"

    ''' 音声ファイルの読み込み(+モノラル変換) '''
    input_path = src_folder + sound_label + "/" + file_name + ".wav"
    data, sr = librosa.load(input_path, sr=4000, mono=True)  # スペクトログラムの周波数軸は 0 ~ (sr / 2) [Hz]

    ''' 帯域制限 '''
    fp = np.array([50, 1800])      # 通過域端周波数[Hz]
    fs = np.array([10, 2000])      # 阻止域端周波数[Hz]
    gpass = 3   # 通過域端最大損失[dB]
    gstop = 40  # 阻止域端最小損失[dB]
    filt_data = filter.bandpass(data, sr, fp, fs, gpass, gstop)

    ''' STFT '''
    window_long = int(0.048 * sr)
    window_short = int(0.016 * sr)
    overlap_long = int(window_long / 2)
    overlap_short = int(window_short / 2)
    stft_long = np.abs(librosa.stft(filt_data,
                                    n_fft=window_long,
                                    hop_length=overlap_long,
                                    win_length=window_long,
                                    window=signal.get_window("hamming", window_long)))
    stft_short = np.abs(librosa.stft(filt_data,
                                     n_fft=window_short,
                                     hop_length=overlap_short,
                                     win_length=window_short,
                                     window=signal.get_window("hamming", window_short)))

    ''' 元スペクトログラムの作成 '''
    if save_fig['O']:
        # 元スペクトログラム作成.
        fig = plt.figure(figsize=(fig_width, fig_height))
        fig.add_axes([0, 0, 1, 1])  # [x0, y0, width(比率), height(比率)]
        librosa.display.specshow(librosa.amplitude_to_db(stft_short, ref=np.max),
                                 y_axis='linear',
                                 x_axis='time',
                                 sr=sr,
                                 cmap=spec_color)
        # 元スペクトログラム保存.
        plt.ylim(min_freq, max_freq)
        o_filepath = spectrogram_dst_folder + "O-/" + sound_label + "/" + file_name + "_O.png"
        plt.savefig(o_filepath)

    ''' HPSS '''
    if save_fig['H']:
        # 大きめの窓幅でのHPSS.
        h_mask, _ = librosa.decompose.hpss(stft_long,
                                           margin=hp_margin,
                                           kernel_size=12,
                                           power=2.0,
                                           mask=True)
        h_stft = stft_long * h_mask
        # Hスペクトログラムの作成.
        # h_mel = librosa.feature.melspectrogram(S=h_stft, sr=sr, n_mels=128)
        fig = plt.figure(figsize=(fig_width, fig_height))
        fig.add_axes([0, 0, 1, 1])
        librosa.display.specshow(librosa.amplitude_to_db(h_stft, ref=np.max),
                                 y_axis='linear',
                                 x_axis='time',
                                 sr=sr,
                                 cmap=spec_color)
        plt.ylim(min_freq, max_freq)
        # Hスペクトログラムの保存.
        h_filepath = spectrogram_dst_folder + "H-/" + sound_label + "/" + file_name + "_H.png"
        plt.savefig(h_filepath)

    if save_fig['P']:
        # 小さめの窓幅でのHPSS.
        _, p_mask = librosa.decompose.hpss(stft_short,
                                           margin=hp_margin,
                                           kernel_size=10,
                                           power=2.0,
                                           mask=True)
        p_stft = stft_short * p_mask
        # Pスペクトログラムの作成.
        # p_mel = librosa.feature.melspectrogram(S=p_stft, sr=sr, n_mels=32)
        fig = plt.figure(figsize=(fig_width, fig_height))
        fig.add_axes([0, 0, 1, 1])
        librosa.display.specshow(librosa.amplitude_to_db(p_stft, ref=np.max),
                                 y_axis='linear',
                                 x_axis='time',
                                 sr=sr,
                                 cmap=spec_color)
        plt.ylim(min_freq, max_freq)
        # Pスペクトログラムの保存.
        p_filepath = spectrogram_dst_folder + "P-/" + sound_label + "/" + file_name + "_P.png"
        plt.savefig(p_filepath)


''' HPSSの実行 '''
for label in sound_classes:
    print(label)
    src_path = src_folder + label + "/*"
    files = glob.glob(src_path)
    file_total = len(files)     # 進捗確認用.
    i = 1                       # 進捗確認用.
    # files内の各ファイルにHPSSを実行.
    for n in files:
        print("\r%d / %d" % (i, file_total), end='')     # 進捗確認用.
        name = os.path.basename(n).rstrip(".wav")
        hpss(name, label)
        i += 1                  # 進捗確認用.
    print()                     # 進捗確認用.

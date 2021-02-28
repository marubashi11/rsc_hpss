# coding:UTF-8
# 呼吸音データの周期ごとに区切って保存するプログラム

import glob
import numpy as np
import os
import program.make_images.wav_info as wi

# 音声データファイルパスの指定
input_path = "/home/marubashi/ドキュメント/HPSS_test_Python/audio_data/equalized/*.wav"
txt_path = "/home/marubashi/ドキュメント/ICBHI_Challenge_2017/audio_and_txt_files/"
output_path = "/home/marubashi/ドキュメント/HPSS_test_Python/audio_data/split_cycle/"

# 元データのフォルダからwavファイルをすべて格納.
files = glob.glob(input_path)

for n in files:

    # WAVEファイル読み出し.
    ch, width, fr, fn, time, data = wi.get_wav_info(n)
    X = np.frombuffer(data, dtype="int16")
    # 選択したwavファイルと同名のtxtファイルのパスを格納.
    txt = txt_path + os.path.basename(n).rstrip(".wav") + ".txt"
    # txtファイルの読み出し.
    txt_data = open(txt, "r")

    # 出力wavファイル末尾の番号をセット.
    count = 0

    for line in txt_data:

        count += 1
        # 各txt_data内のテキストを区切りごとに分割してdata配列に格納
        data = line.split()

        # 0番目の数値が呼吸周期の開始フレーム, 1番目の数値が終了フレームを表すのでそれぞれ格納.
        start_frame = float(data[0])
        end_frame = float(data[1])
        start_frame_sample = int(fr * start_frame * ch)
        end_frame_sample = int(fr * end_frame * ch)

        # フレームを指定して音声データを抜き出す.
        Y = X[start_frame_sample:end_frame_sample]

        if (int(data[2]) == 0) & (int(data[3]) == 0):
            output = output_path + "normal/" + os.path.basename(n).rstrip(".wav") + "_" + str(count) + ".wav"
            wi.set_wav_info(output, ch, width, fr, Y)
        elif (int(data[2]) == 1) & (int(data[3]) == 0):
            output = output_path + "crackle/" + os.path.basename(n).rstrip(".wav") + "_" + str(count) + ".wav"
            wi.set_wav_info(output, ch, width, fr, Y)
        elif (int(data[2]) == 0) & (int(data[3]) == 1):
            output = output_path + "wheeze/" + os.path.basename(n).rstrip(".wav") + "_" + str(count) + ".wav"
            wi.set_wav_info(output, ch, width, fr, Y)
        else:
            output = output_path + "crackle_wheeze/" + os.path.basename(n).rstrip(".wav") + "_" + str(count) + ".wav"
            wi.set_wav_info(output, ch, width, fr, Y)

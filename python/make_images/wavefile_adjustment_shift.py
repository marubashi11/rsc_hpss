# coding:UTF-8
# 呼吸音データの長さを調整するプログラム
import os
import glob
import numpy as np
from program.make_images.wav_info import get_wav_info
from program.make_images.wav_info import set_wav_info

# 呼吸音データの切り取り設定 [s]
frame_size = 3
frame_shift = 2

# 音声データファイルパスの指定
input_path = "/home/marubashi/ドキュメント/HPSS_test_Python/audio_data/equalized/*.wav"
txt_path = "/home/marubashi/ドキュメント/ICBHI_Challenge_2017/events/"
output_path = "/home/marubashi/ドキュメント/HPSS_test_Python/audio_data/split_shift/"

files = glob.glob(input_path)


# def save(files, count):
#     file_path = output_path + os.path.basename(files).rstrip(".wav") + "_" + str(count + 1) + ".wav"
#     set_wav_info(file_path, ch, width, fr, Y)


# def devide(c, w, file, count):
#     if (c == 0) and (w == 0):
#         file_path = output_path + "normal/" + os.path.basename(file).rstrip(".wav") + "_" + str(count + 1) + ".wav"
#         set_wav_info(file_path, ch, width, fr, Y)
#     elif (c >= 1) and (w == 0):
#         file_path = output_path + "crackle/" + os.path.basename(file).rstrip(".wav") + "_" + str(count + 1) + ".wav"
#         set_wav_info(file_path, ch, width, fr, Y)
#     elif (c == 0) and (w >= 1):
#         file_path = output_path + "wheeze/" + os.path.basename(file).rstrip(".wav") + "_" + str(count + 1) + ".wav"
#         set_wav_info(file_path, ch, width, fr, Y)
#     else:
#         file_path = output_path + "crackle_wheeze/" + os.path.basename(file).rstrip(".wav") + "_" + str(count + 1) + ".wav"
#         set_wav_info(file_path, ch, width, fr, Y)


for n in files:

    # WAVEファイル読み出し.
    ch, width, fr, fn, time, data = get_wav_info(n)

    # dataを-1~1のfloat型ndarrayに変換.
    # X = np.frombuffer(data, dtype="int16") / 32768.0
    X = np.frombuffer(data, dtype="int16")

    # 時間をサンプル数に変換.
    frame_size_sample = int(fr * frame_size * ch)
    frame_shift_sample = int(fr * frame_shift * ch)

    # フレームの総数(シフト可能な最大回数)を求める.
    max_frame = int((len(X) - (frame_size_sample - frame_shift_sample)) / frame_shift_sample)

    # フレームの個数だけ処理を繰り返す.
    start_frame = 0
    end_frame = start_frame + frame_size_sample

    for frame_count in range(max_frame):

        crackle = 0
        wheeze = 0

        # 出力データの生成.
        Y = X[start_frame:end_frame]

        # txtファイル読み出し.
        txt = txt_path + os.path.basename(n).rstrip(".wav") + "_events.txt"
        txt_data = open(txt, "r")

        for line in txt_data:

            check = 0
            data = line.split()

            # 切り出し区間に何らかのイベントが発生していればチェック.
            if (float(data[0]) >= float(start_frame / fr)) and (float(data[0]) <= float(end_frame / fr)):
                check = 1
            if (float(data[1]) >= float(start_frame / fr)) and (float(data[1]) <= float(end_frame / fr)):
                check = 1
            if (float(data[0]) < float(start_frame / fr)) and (float(data[1]) > float(end_frame / fr)):
                check = 1

            if check == 1:
                if data[2] == "crackle":
                    crackle += 1
                if data[2] == "wheeze":
                    wheeze += 1

        # save.
        # devide(crackle, wheeze, n, frame_count)
        if (crackle == 0) and (wheeze == 0):
            print(0)
            file_path = output_path + "normal/" + os.path.basename(n).rstrip(".wav") + "_"\
                        + str(frame_count + 1) + ".wav"
            set_wav_info(file_path, ch, width, fr, Y)
        elif (crackle >= 1) and (wheeze == 0):
            print(1)
            file_path = output_path + "crackle/" + os.path.basename(n).rstrip(".wav") + "_"\
                        + str(frame_count + 1) + ".wav"
            set_wav_info(file_path, ch, width, fr, Y)
        elif (crackle == 0) and (wheeze >= 1):
            file_path = output_path + "wheeze/" + os.path.basename(n).rstrip(".wav") + "_"\
                        + str(frame_count + 1) + ".wav"
            set_wav_info(file_path, ch, width, fr, Y)
        else:
            file_path = output_path + "crackle_wheeze/" + os.path.basename(n).rstrip(".wav") + "_"\
                        + str(frame_count + 1) + ".wav"
            set_wav_info(file_path, ch, width, fr, Y)

        # startとendをshift分だけずらす.
        start_frame = start_frame + frame_shift_sample
        end_frame = start_frame + frame_size_sample

# coding:UTF-8

import wave


def get_wav_info(file):

    wr = wave.open(file, "rb")

    ch = wr.getnchannels()  # チャンネル数
    width = wr.getsampwidth() # 量子化バイト数
    fr = wr.getframerate()  # サンプリングレート
    fn = wr.getnframes()  # フレーム数
    time = 1.0 * fn / fr
    data = wr.readframes(wr.getnframes())

    wr.close()

    return ch, width, fr, fn, time, data


def set_wav_info(file, ch, width, fr, X):

    ww = wave.open(file, "w")
    ww.setnchannels(ch)
    ww.setsampwidth(width)
    ww.setframerate(fr)
    ww.writeframes(X)
    ww.close()

from matplotlib import pyplot as plt
import numpy as np
from scipy import signal


def bandpass(x, sr, fp, fs, gpass, gstop):
    fn = sr / 2                                     # ナイキスト周波数
    wp = fp / fn                                    # ナイキスト周波数で通過域端周波数を正規化
    ws = fs / fn                                    # ナイキスト周波数で阻止域端周波数を正規化
    N, Wn = signal.buttord(wp, ws, gpass, gstop)    # オーダーとバターワースの正規化周波数を計算
    b, a = signal.butter(N, Wn, "band")             # フィルタ伝達関数の分子と分母を計算
    y = signal.filtfilt(b, a, x)                    # 信号に対してフィルタをかける
    return y


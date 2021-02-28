# 分類失敗データを記録したtxtファイルを読み出すだけ.
import pickle

split_type = 'cycle'
set_num = '1'
file_path = "/home/marubashi/ドキュメント/HPSS_test_Python/dataset_" + split_type + "/dataset" + set_num \
            + "/f_list_" + set_num + ".txt"
new_file_path = "/home/marubashi/ドキュメント/HPSS_test_Python/dataset_" + split_type + "/dataset" + set_num \
                + "/F_list_" + set_num + ".txt"

f_list = pickle.load(open(file_path, "rb"))

with open(file_path, 'w') as f:
    f.writelines(f_list)

with open(new_file_path, 'w') as f:
    for name in f_list:
        f.write("%s\n" % name)

# print(f_list)

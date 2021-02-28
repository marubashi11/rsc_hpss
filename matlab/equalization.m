% 音声データを正規化して保存.
input_path = ...
    '/home/marubashi/ドキュメント/ICBHI_Challenge_2017/audio_and_txt_files/';
output_path = ...
    '/home/marubashi/ドキュメント/HPSS_test_Python/audio_data/equalized/';

WAV_files = dir([input_path, '*.wav']);
num_files = length(WAV_files);

max_peak = 1;
new_sampling_rate = 4000;

for n = 1:num_files
    file_path = [input_path WAV_files(n).name];
    [path, name, ext] = fileparts(file_path); 
    [data, Fs] = audioread(file_path);
    info = audioinfo(file_path);
    
    if info.NumChannels == 2
        data = mean(data, 2);
    end
    
    % 音量正規化.
    peak = max(abs(data));
    magnification = max_peak/peak;
    data = data*magnification;
    
    if Fs ~= new_sampling_rate
        [P, Q] = rat(new_sampling_rate/Fs, 0.0001);
        data = resample(data, P, Q);
    end
    
    audiowrite([output_path name ext], data, new_sampling_rate);
    
end
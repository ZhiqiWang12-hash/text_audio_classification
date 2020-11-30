import librosa
import numpy as np


#将采样的音频样本长度统一
def normalizeVoiceLen(y, normalizedLen):#normalizedLen是统一的长度
    nframes = len(y)
    y = np.reshape(y, [nframes, 1]).T #[1,16000]
    # 归一化音频长度为2s,32000数据点
    if (nframes < normalizedLen):
        res = normalizedLen - nframes
        res_data = np.zeros([1, res], dtype=np.float32)
        y = np.reshape(y, [nframes, 1]).T
        y = np.c_[y, res_data]#拼接[1,32000]
    else:
        y = y[:, 0:normalizedLen]
    #(32000,)
    return y[0]

#据声音的采样率确定一个合适的语音帧长用于傅立叶变换
def getNearestLen(framelength, sr):
    framesize = framelength * sr#16000*0.25
    # print('framesize:', framesize)
    # 找到与当前framesize最接近的2的正整数次方
    nfftdict = {}
    lists = [32, 64, 128, 256, 512, 1024]
    for i in lists:
        nfftdict[i] = abs(framesize - i)
    sortlist = sorted(nfftdict.items(), key=lambda x: x[1])  # 按与当前framesize差值升序排列
# sortlist: [(1024, 2976.0), (512, 3488.0), (256, 3744.0), (128, 3872.0), (64, 3936.0), (32, 3968.0)]
    framesize = int(sortlist[0][0])  # 取最接近当前framesize的那个2的正整数次方值为新的framesize,1024
    return framesize


def get_mfcc(wav,sr=16000,framelength=0.25):
    N_FFT=getNearestLen(framelength,sr)
    mfcc_data_0 = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=13, n_fft=N_FFT, hop_length=int(N_FFT / 4))
    feature_0 = np.mean(mfcc_data_0, axis=0)
    return feature_0

def data_augmentation(X,label):
    wav_aug=list()
    label_aug=list()
    for i in range(len(X)):
        y=X[i]
        y_ps = librosa.effects.pitch_shift(y, sr, n_steps=3) #频率变换
        y_ts = librosa.effects.time_stretch(y, rate=1.2)     #时域变换
        wn = np.random.randn(len(y))
        data_wn = y + 0.005*wn #加入高斯白噪声
        wav_aug.append(y)
        wav_aug.append(y_ps)
        wav_aug.append(y_ts)
        wav_aug.append(data_wn)

        label_aug.append(label[i])
        label_aug.append(label[i])
        label_aug.append(label[i])
        label_aug.append(label[i])
    return wav_aug,label_aug

def get_mfcc_for_train(X,label,sr=16000,framelength=0.25,isAug=True):
    if isAug:
        wav_aug,label_aug=data_augmentation(X,label)
    else:
        wav_aug=X
        label_aug=label
    max_seq_len=0
    mfcc_features=[]
    for wav in wav_aug:
        feature_0=get_mfcc(wav,sr=sr,framelength=framelength)
        mfcc_features.append(feature_0)
        length=len(feature_0)
        if max_seq_len<length:
            max_seq_len=length
    mfcc_features_array=np.zeros((len(mfcc_features),max_seq_len))
    for i in range(mfcc_features_array.shape[0]):
        for j in range(len(mfcc_features[i])):
            mfcc_features_array[i,j]=mfcc_features[i][j]
    return mfcc_features_array,label_aug

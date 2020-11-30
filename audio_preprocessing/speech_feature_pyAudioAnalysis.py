from pyAudioAnalysis.pyAudioAnalysis.ShortTermFeatures import feature_extraction

import numpy as np

def get_speech_features(wav, sr=16000):
  feature,feature_name=feature_extraction(wav,sr,0.05*sr,0.025*sr)
  feature_0=np.zeros((feature.shape[1],3))

  for frame in range(feature.shape[1]):
    feature_0[frame,0]=feature[2,frame]
    feature_0[frame,1]=feature[3,frame]
    feature_0[frame,2]=np.mean(feature[8:21,frame])
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
      sf_features=[]
      for wav in wav_aug:
          feature_0=get_speech_features(wav, sr=16000)
          sf_features.append(feature_0)
          length=feature_0.shape[0]
          if max_seq_len<length:
              max_seq_len=length
      sf_features_array=np.zeros((len(sf_features),max_seq_len,3))
      for i in range(sf_features_array.shape[0]):
          for j in range(len(sf_features[i])):
              sf_features_array[i,j,:]=sf_features[i][j,:]
      return sf_features_array,label_aug

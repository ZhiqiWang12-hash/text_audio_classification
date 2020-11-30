# text_audio_classification
Chinese BERT classification with tf2.0 and audio classification with mfcc

## Text classification
In order to do the emotion recognition classification task on Chinese text, I developed this three different preprocessing mothods according to different model architecture.
These methods refers to bag-of-word classification and sequence classification.

The BERT encoder needs to import tf-models-official 
```
pip install -q tf-models-official==2.3.0
```

## Audio classification
For the preprocessing method, I've tried three different lib: librosa, [aubio](https://github.com/aubio/aubio/tree/6b84d815b7333e98b3b23ac0b80b9bd40648b93b) and 
[pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis)

## Multi-modal classification
This model combine the text with the audio features(mfcc). The architecture is referred to [Learning Alignment for Multimodal Emotion Recognition from Speech](https://arxiv.org/abs/1909.05645)


>@misc{xu2020learning,
      title={Learning Alignment for Multimodal Emotion Recognition from Speech}, 
      author={Haiyang Xu and Hui Zhang and Kun Han and Yun Wang and Yiping Peng and Xiangang Li},
      year={2020},
      eprint={1909.05645},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

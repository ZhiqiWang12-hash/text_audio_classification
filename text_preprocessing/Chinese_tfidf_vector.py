import jieba
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import preprocessing

from sklearn.decomposition import PCA



def stop_word_loading(file_stopping):
    stopping_word = []
    with open(file_stopping,encoding='UTF-8') as f:
        for lines in f.readlines():
            word = lines.strip(' ').strip('\n')#去除空格和换行符
            stopping_word.append(word)
    return stopping_word

def remove_words(sentence,stop_words):
    #tokenization
    #s = SnowNLP(sentence)
    #sentence_list=s.words
    sentence=' '.join(jieba.cut(sentence,cut_all=False))
    sentence_list=sentence.split()
    for word in sentence_list:
        if word in stop_words:
            sentence_list.remove(word)
    return sentence_list


def remove_words_bunch(X,stop_words):
    X_n = []
    MAX_SEQUENCE_LENGTH=0
    for x in X:
        sentence_list=remove_words(x,stop_words)
        length=len(sentence_list)
        if length> MAX_SEQUENCE_LENGTH:
            MAX_SEQUENCE_LENGTH=length
        X_n.append(sentence_list)
    print(MAX_SEQUENCE_LENGTH)
    return X_n,MAX_SEQUENCE_LENGTH

def vectorise(X,stop_word_path,token_pattern=r"(?u)\b\w+\b", max_df=0.6,max_features=300,isPCA=True,n_components=200,isSparse=False,):
    stop_words=stop_word_loading(stop_word_path)
    X_n,MAX_SEQUENCE_LENGTH=remove_words_bunch(X,stop_words)
    X_s=[]
    for x in X_n:
        string=''
        for item in x:
            string+=item+' '
        X_s.append(string)
    tfidf_model = TfidfVectorizer(token_pattern=token_pattern, max_df=max_df,stop_words=stop_words,max_features=max_features).fit(X_s)
    #print(tfidf_model.vocabulary_)
    X_sparse= tfidf_model.transform(X_s)

    if isPCA==True and isSparse==False:
        pca = PCA(n_components=200)
        pca.fit(X_sparse.A)
        X_p = pca.transform(X_sparse.A)
        scaler = preprocessing.StandardScaler().fit(X_p)
        X_pre=scaler.transform(X_p)

    return X_pre

        

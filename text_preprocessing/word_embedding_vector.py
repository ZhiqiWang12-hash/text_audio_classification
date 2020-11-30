import jieba
import numpy as np


def read_content_labels(pckl_content_path,pckl_label_path,content_file_path,label_file_path):
    X=[]
    Y=[]
    pckl_f_c=open(pckl_content_path,'rb')
    pckl_f_l=open(pckl_label_path,'rb')
    if pckl_f_c:
        X=pickle.load(pckl_f_c)
        Y=pickle.load(pckl_f_l)
    else:
        with open(content_file_path,'r') as f:
            line=f.readline()
            while line:
                X.append(line)
                line=f.readline()
        with open(label_file_path,'r') as f:
            line=f.readline()
            while line:
                Y.append(line)
                line=f.readline()
    return X,Y

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

def embedding_index_load(word2vec_path):
    embeddings_index = {}
    EMBEDDING_DIM=300
    with open(word2vec_path,'r') as f:
        line = f.readline()
        while line:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
            line=f.readline()
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index

def word_index_form(X,MAX_NB_WORDS=140):
    word_index={}
    count=0
    for x in X:
        for word in x:
            if word not in word_index:
                count+=1
                word_index[word]=count
    return word_index

def embedding_matrix_form(word_index,embedding_index,EMBEDDING_DIM=300):
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def word2sequence(word_index,X,MAX_SEQUENCE_LENGTH):
    X_s = np.zeros(len(X)*MAX_SEQUENCE_LENGTH).reshape(-1,MAX_SEQUENCE_LENGTH)
    for i in range(len(X)):
        for j in range(len(X[i])):
            if X[i][j] in word_index:
                X_s[i,j]=word_index[X[i][j]]
    return X_s

def vectorise(X,stop_word_path,embedding_path,embedding_dim=300):
    stop_words=stop_word_loading(stop_word_path)
    embedding_index = embedding_index_load(embedding_path)
    X_n,MAX_SEQUENCE_LENGTH=remove_words_bunch(X,stop_words)
    word_index = word_index_form(X_train_n)
    embedding_matrix=embedding_matrix_form(word_index,embedding_index,EMBEDDING_DIM=embedding_dim)
    X_t=word2sequence(word_index,X_n,MAX_SEQUENCE_LENGTH)
    len_word_index=len(word_index)
    return X_t,embedding_matrix,len_word_index,MAX_SEQUENCE_LENGTH

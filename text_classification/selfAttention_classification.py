from keras.preprocessing import sequence
#from tensorflow.keras.datasets import imdb
from matplotlib import pyplot as plt
import pandas as pd

from keras import backend as K
from keras.engine.topology import Layer

class Self_Attention(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Self_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        #inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3,input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)

        super(Self_Attention, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])

        print("WQ.shape",WQ.shape)

        print("K.permute_dimensions(WK, [0, 2, 1]).shape",K.permute_dimensions(WK, [0, 2, 1]).shape)


        QK = K.batch_dot(WQ,K.permute_dimensions(WK, [0, 2, 1]))

        QK = QK / (64**0.5)

        QK = K.softmax(QK)

        print("QK.shape",QK.shape)

        V = K.batch_dot(QK,WV)

        return V

    def compute_output_shape(self, input_shape):

        return (input_shape[0],input_shape[1],self.output_dim)


class selfAttention_classification():
    def __init__(self,embedding_matrix,MAX_SEQUENCE_LENGTH=144,len_word_index=18346,EMBEDDING_DIM=300):
        self.len_word_index= len_word_index
        self.embedding_matrix=embedding_matrix
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
        self.EMBEDDING_DIM=EMBEDDING_DIM

        self.model=self.build_model()
        optimizer = optimizers.Adam(lr=0.0001, beta_1=0.5)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])

    def build_model(self,num_label):
        embedding_layer = Embedding(self.len_word_index + 1,
                            self.EMBEDDING_DIM,
                            weights=[self.embedding_matrix],
                            input_length=self.MAX_SEQUENCE_LENGTH,
                            trainable=False)
        sentence_input=Input(shape=(self.MAX_SEQUENCE_LENGTH,))
        print(type(sentence_input))
        embedding_sentence=embedding_layer(sentence_input)
        print(type(embedding_sentence))

        c1 = Self_Attention(128)(embedding_sentence)


        c2 = GlobalAveragePooling1D()(c1)
        #c3 = Flatten()(c2)


        #c8=LSTM(50)(c7)
        #c9=Dense(25,kernel_initializer='uniform',activation='relu')(c2)
        #c10=Dropout(0.5)(c9)
        #c11=Dense(25,kernel_initializer='uniform',activation='relu')(c10)
        #c12=Dropout(0.5)(c11)
        output=Dense(num_label,kernel_initializer='uniform',activation='softmax')(c2)

        sentence_model=Model(sentence_input,output)
        sentence_model.summary()
        return sentence_model
    def train(self,X_train_t,Y_train_t,X_val_t,Y_val_t,batch_size=8,epochs=20,class_weight=None,shuffle=False):
        history=self.model.fit(X_train_t, Y_train_t,batch_size=batch_size,
                  epochs=epochs,
                  class_weight=class_weight,
                  shuffle=shuffle,
                  validation_data=(X_val_t,Y_val_t))
        return history
    def save_model(self):
        self.model.save('multi_model.h5')
    def predict(self,X):
        pred=self.model.predict(X)
        return pred
    def get_model(self):
        return self.model
    def set_compile(self,lr=0.001,loss='categorical_crossentropy',metrics=['accuracy']):
        optimizer = optimizers.Adam(lr=lr, beta_1=0.5)
        self.model.compile(optimizer=optimizer, loss=loss,metrics=metrics)

import numpy as np



from tensorflow.keras import regularizers
from tensorflow.keras.layers import Embedding,LSTM,Dense,Input,Dropout,MaxPooling1D,Conv1D,Flatten,concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


class cnn_classification():
    def __init__(self,embedding_matrix,MAX_SEQUENCE_LENGTH=144,len_word_index=18346,EMBEDDING_DIM=300):
        self.len_word_index= len_word_index
        self.embedding_matrix=embedding_matrix
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
        self.EMBEDDING_DIM=EMBEDDING_DIM

        self.model=self.build_model()
        optimizer = optimizers.Adam(lr=0.001, beta_1=0.5)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])

    def build_model(self,num_label):
        embedding_layer = Embedding(self.len_word_index + 1,
                            self.EMBEDDING_DIM,
                            weights=[self.embedding_matrix],
                            input_length=self.MAX_SEQUENCE_LENGTH,
                            trainable=False)
        sentence_input=Input(shape=(self.MAX_SEQUENCE_LENGTH,))
        embedding_sentence=embedding_layer(sentence_input)
        c1 = Conv1D(128, 5, activation='relu')(embedding_sentence)
        c2 = MaxPooling1D(3)(c1)
        c3= Conv1D(128, 5, activation='relu')(c1)
        c4= MaxPooling1D(5)(c3)
        c5 = Conv1D(128, 5, activation='relu')(c3)
        c6= MaxPooling1D(35)(c5)  # global max pooling
        c7= Flatten()(c6)

        #c8=LSTM(50)(c7)
        c9=Dense(50,kernel_initializer='uniform',activation='relu')(c7)
        c10=Dropout(0.5)(c9)
        output=Dense(num_label,kernel_initializer='uniform',activation='softmax')(c10)

        sentence_model=Model(sentence_input,output)
        sentence_model.summary()
        return sentence_model
    def train(self,X_train_t,Y_train_t,batch_size=8,epochs=20,validation_split=0.25,class_weight=None):
        self.model.fit(X_train_t, Y_train_t,batch_size=batch_size,
                  epochs=epochs,
                  class_weight=class_weight,
                  validation_split=validation_split)
    def save_model(self):
        self.model.save('model.h5')
    def predict(self,X):
        pred=self.model.predict(X)
        return pred
    def get_model(self):
        return self.model
    def set_compile(self,lr=0.001,loss='categorical_crossentropy',metrics=['accuracy']):
        optimizer = optimizers.Adam(lr=lr, beta_1=0.5)
        self.model.compile(optimizer=optimizer, loss=loss,metrics=metrics)

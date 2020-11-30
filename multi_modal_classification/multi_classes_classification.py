import keras
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Embedding, LSTM, Input, Dense, BatchNormalization,Attention,Concatenate,MaxPooling1D,Bidirectional,Flatten
from tensorflow.keras import optimizers


class audio_cnn_classification():
    def __init__(self,num_label,num_channel,max_seq_len):
        self.model=build_model(num_label,num_channel,max_seq_len)
        opt = optimizers.Adam(lr=0.001, decay=1e-6)
        #metrics = ['accuracy',  fmeasure, recall, precision]
        self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    def build_model(self,num_label,num_channel,mfcc_max_seq_len,text_max_seq_len,EMBEDDING_DIM=100,len_word_index,embedding_matrix):
        input_mfcc=Input(shape=(mfcc_max_seq_len,num_channel))
        input_text=Input(shape=(text_max_seq_len,))
        embedding_layer = Embedding(len_word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=text_max_seq_len,
                            trainable=False)

        text_1=embedding_layer(input_text)

        text_2=Bidirectional(LSTM(3, return_sequences=True))(text_1)
        mfcc_1=Bidirectional(LSTM(3, return_sequences=True))(input_mfcc)


        # Query-value attention of shape [batch_size, Tq, filters].
        query_value_attention_seq = Attention()([text_2, mfcc_1])

        input_LSTM_layer = Concatenate()([text_2, query_value_attention_seq])

        c0=Bidirectional(LSTM(3, return_sequences=True))(input_LSTM_layer)
        c1=MaxPooling1D((3,))(c0)
        c2=Flatten()(c1)
        output=Dense(num_label,activation='sigmoid')(c2)


        model=Model(inputs=[input_mfcc,input_text],outputs=[output])
        model.summary()
        return model
    def train(X_train,Y_train,X_val,Y_val,batch_size=8,class_weight=None,epochs=100):
        history = model.fit(X_train_, Y_train,
                  batch_size=batch_size,
                  class_weight=class_weight,
                  epochs=epochs,validation_data=(X_val,Y_val))

    def predict(self,X):
        pred = self.model.predict(X)
        return pred
    def get_model(self):
        return self.model
    def set_compile(self,lr=0.001, decay=1e-6,loss='binary_crossentropy', metrics=['accuracy']):
        opt = optimizers.Adam(lr=lr, decay=decay)
        #metrics = ['accuracy',  fmeasure, recall, precision]
        self.model.compile(loss=loss, optimizer=opt, metrics=metrics)

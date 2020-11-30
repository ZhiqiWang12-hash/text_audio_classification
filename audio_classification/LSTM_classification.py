import keras
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Embedding, LSTM, Input, Dense, BatchNormalization,Attention,Concatenate,MaxPooling1D,Bidirectional,Flatten
from tensorflow.keras import optimizers


class audio_cnn_classification():
    def __init__(self,num_label,num_channel,max_seq_len):
        self.model=build_model(num_label,num_channel,max_seq_len)
        opt = optimizers.Adam(lr=0.001, decay=1e-6)
        #metrics = ['accuracy',  fmeasure, recall, precision]
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    def build_model(self,num_label,num_channel,max_seq_len):
        input_mfcc=Input(shape=(max_seq_len,num_channel))
        mfcc_1=Bidirectional(LSTM(10, return_sequences=True))(input_mfcc)
        # Query-value attention of shape [batch_size, Tq, filters].
        query_value_attention_seq = Attention()([mfcc_1,mfcc_1] )

        input_LSTM_layer = Concatenate()([mfcc_1, query_value_attention_seq])

        c0=Bidirectional(LSTM(10, return_sequences=True))(input_LSTM_layer)
        c1=MaxPooling1D((3,))(c0)
        c2=Flatten()(c1)
        c3=Dense(100,activation='tanh')(c2)
        output=Dense(num_label,activation='softmax')(c3)


        model=Model(inputs=[input_mfcc],outputs=[output])

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
    def set_compile(self,lr=0.001, decay=1e-6,loss='categorical_crossentropy', metrics=['accuracy']):
        opt = optimizers.Adam(lr=lr, decay=decay)
        #metrics = ['accuracy',  fmeasure, recall, precision]
        self.model.compile(loss=loss, optimizer=opt, metrics=metrics)

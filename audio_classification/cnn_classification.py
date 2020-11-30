import keras
# from keras.utils import plot_model
from keras import regularizers
from keras import models
from keras import layers
from keras import optimizers
import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model


class audio_cnn_classification():
    def __init__(self,num_label,num_channel,max_seq_len):
        self.model=build_model(num_label,num_channel,max_seq_len)
        opt = optimizers.Adam(lr=0.000001, decay=1e-6)
        #metrics = ['accuracy',  fmeasure, recall, precision]
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    def build_model(self,num_label,num_channel,max_seq_len):
        model = models.Sequential()
        #输入为[batch,steps,input_dim],输出位[batch,new_step,filter],filter=256
        model.add(layers.Conv1D(256, 5, activation='tanh', input_shape=(max_seq_len, num_channel)))#[122, 256]
        #model.add(layers.BatchNormalization())
        model.add(layers.Conv1D(128, 5, padding='same', activation='tanh'))#,  kernel_regularizer=regularizers.l2(0.001)))#[122,128]
        #model.add(layers.BatchNormalization())
        #model.add(layers.Dropout(0.2))
        model.add(layers.MaxPooling1D(pool_size=(8)))#122/8 = 15
        model.add(layers.Conv1D(128, 5, activation='tanh', padding='same'))#, kernel_regularizer=regularizers.l2(0.001)))#[15, 128]
        #model.add(layers.Dropout(0.2))
        model.add(layers.Conv1D(128, 5, activation='tanh', padding='same'))#, kernel_regularizer=regularizers.l2(0.001)))#[15, 128]
        #model.add(layers.Dropout(0.2))
        model.add(layers.Conv1D(128, 5, padding='same', activation='tanh'))#, kernel_regularizer=regularizers.l2(0.001)))#[15,128]
        model.add(layers.BatchNormalization())
        ##model.add(layers.Dropout(0.2))
        model.add(layers.MaxPooling1D(pool_size=(3)))#[5,128]
        model.add(layers.Conv1D(256, 5, padding='same', activation='tanh'))#, kernel_regularizer=regularizers.l2(0.001)))#[5,256]
        model.add(layers.BatchNormalization())
        #model.add(layers.Dropout(0.2))
        model.add(layers.Flatten())#(,1280)
        #model.add(layers.Dense(100, activation='tanh',kernel_initializer='uniform'))
        #model.add(layers.BatchNormalization())
        #model.add(layers.Dropout(0.2))
        model.add(layers.Dense(500, activation='tanh',kernel_initializer='uniform'))
        #model.add(layers.BatchNormalization())
        #model.add(layers.Dropout(0.2))
        model.add(layers.Dense(100, activation='tanh',kernel_initializer='uniform'))
        #model.add(layers.BatchNormalization())
        #model.add(layers.Dropout(0.2))
        model.add(layers.Dense(50, activation='tanh',kernel_initializer='uniform'))
        #model.add(layers.Dense(50, activation='relu',kernel_initializer='uniform'))
        #model.add(layers.Dense(50, activation='relu',kernel_initializer='uniform'))
        #model.add(layers.Dense(50, activation='relu',kernel_initializer='uniform'))
        #model.add(layers.Dense(50, activation='relu',kernel_initializer='uniform'))
        #model.add(layers.Dense(50, activation='relu',kernel_initializer='uniform'))
        model.add(layers.Dense(25, activation='tanh',kernel_initializer='uniform'))
        model.add(layers.Dense(num_label, activation='softmax',kernel_initializer='uniform'))#(,6)
        # plot_model(model, to_file='mfcc_model.png', show_shapes=True)
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
    def set_compile(self,lr=0.000001, decay=1e-6,loss='categorical_crossentropy', metrics=['accuracy']):
        opt = optimizers.Adam(lr=lr, decay=decay)
        #metrics = ['accuracy',  fmeasure, recall, precision]
        self.model.compile(loss=loss, optimizer=opt, metrics=metrics)

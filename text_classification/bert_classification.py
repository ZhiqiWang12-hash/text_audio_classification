import os

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

from official.modeling import tf_utils
from official import nlp
from official.nlp import bert

# Load the required submodules
import official.nlp.optimization
import official.nlp.bert.bert_models
import official.nlp.bert.configs
import official.nlp.bert.run_classifier
import official.nlp.bert.tokenization
import official.nlp.data.classifier_data_lib
import official.nlp.modeling.losses
import official.nlp.modeling.models
import official.nlp.modeling.networks


import json

class bert_text_classification():
    def __init__(self,bert_config_file,num_label):
        self.hub_url_bert='https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/2'
        #tf.io.gfile.listdir(gs_folder_bert)
        hub_encoder = hub.KerasLayer(hub_url_bert,trainable=True)
        print(f"The Hub encoder has {len(hub_encoder.trainable_variables)} trainable variables")
        config_dict = json.loads(tf.io.gfile.GFile(bert_config_file).read())
        bert_config = bert.configs.BertConfig.from_dict(config_dict)
        self.hub_classifier, self.hub_encoder = bert.bert_models.classifier_model(
        # Caution: Most of `bert_config` is ignored if you pass a hub url.
        bert_config=bert_config,hub_module_url=hub_url_bert, num_labels=num_label)
        tf.keras.utils.plot_model(hub_classifier, show_shapes=True, dpi=64)

    def load_encoder_weights(self):
        checkpoint = tf.train.Checkpoint(model=self.hub_encoder)
        checkpoint.restore(
                    os.path.join(self.hub_url_bert, 'bert_model.ckpt')).assert_consumed()
    def set_compile(self,epochs = 3,batch_size = 12,eval_batch_size = 12):
        self.epochs=epochs
        self.batch_size=batch_size
        self.eval_batch_size=eval_batch_size
        train_data_size = len(X_train)
        steps_per_epoch = int(train_data_size / batch_size)
        num_train_steps = steps_per_epoch * epochs
        warmup_steps = int(epochs * train_data_size * 0.1 / batch_size)

        # creates an optimizer with learning rate schedule
        optimizer = nlp.optimization.create_optimizer(
            2e-5, num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)

        metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.hub_classifier.compile(
                    optimizer=optimizer,
                    loss=loss,
                    metrics=metrics)

    def train(self,X_train,Y_train,X_val,Y_val):
        hub_classifier.fit(
                    X_train, Y_train,
                    validation_data=(X_val, Y_val),
                    batch_size=12,
                    class_weight=None,
                    epochs=epochs)
    def predict(self,X_pre,num_label):
        result=self.hub_classifier(X_pre, training=False)
        result_array=np.zeros((num_pre,num_label))
        for i in range(result_array.shape[0]):
            label=tf.argmax(result[i]).numpy()
            result_array[i,label]=1
        return result_array

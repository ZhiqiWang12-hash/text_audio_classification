import os

import numpy as np

from official import nlp
from official.nlp import bert


def encode_sentence(s, tokenizer):
   tokens = list(tokenizer.tokenize(s))
   tokens.append('[SEP]')
   return tokenizer.convert_tokens_to_ids(tokens)

def bert_encode(X, tokenizer):
  num_examples = len(X)

  sentence1 = tf.ragged.constant([
      encode_sentence(s, tokenizer)
      for s in np.array(X)])
  #sentence2 = tf.ragged.constant([
  #    encode_sentence(s, tokenizer)
  #     for s in np.array(glue_dict["sentence2"])])

  cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])]*sentence1.shape[0]
  input_word_ids = tf.concat([cls, sentence1], axis=-1)

  input_mask = tf.ones_like(input_word_ids).to_tensor()

  type_cls = tf.zeros_like(cls)
  type_s1 = tf.zeros_like(sentence1)
  #type_s2 = tf.ones_like(sentence2)
  input_type_ids = tf.concat(
      [type_cls, type_s1], axis=-1).to_tensor()

  inputs = {
      'input_word_ids': input_word_ids.to_tensor(),
      'input_mask': input_mask,
      'input_type_ids': input_type_ids}

  return inputs

def bert_tokenisation(vocab_file):
    tokenizer = bert.tokenization.FullTokenizer(
    vocab_file=vocab_file,
     do_lower_case=True)
     print("Vocab size:", len(tokenizer.vocab))
     return tokenizer

def encoding(X,vocab_file):
    tokenizer=bert_tokenisation(vocab_file)
    bert_X = bert_encode(X, tokenizer)
    return bert_X

    

from __future__ import print_function
from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
import regex
import sys

def load_de_en_vocab():
    vocab = [line.split()[0] for line in codecs.open('preprocessed/de_en.vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def load_tw_vocab():
    vocab = [line.split()[0] for line in codecs.open('preprocessed/tw.vocab.tsv', 'r', 'utf-8').read().splitlines()]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def create_data(source_sents, target_sents, topic_wds):
    token2idx, idx2token = load_de_en_vocab()
    tw2idx, idx2tw = load_tw_vocab()
    turn_over_count = 0

    # Index
    x_list, y_list, tw_list, ytwd_list, y_decoder_input_list, Sources, Targets = [], [], [], [], [], [], []
    for source_sent, target_sent, topic_wd in zip(source_sents, target_sents, topic_wds):
        source_sent_split = source_sent.split(u"</d>")
        x=[]
        for sss in source_sent_split:
            if len(sss.split())==0:
                continue
            x.append( [token2idx.get(word, 1) for word in (sss + u" </S>").split()]) # 1: OOV, </S>: End of Text

        target_sent_split = target_sent.split(u"</d>")
        y = [token2idx.get(word, 1) for word in (target_sent_split[0] + u" </S>").split()]
        ytwd = [tw2idx.get(word, 1) for word in (target_sent_split[0] + u" </S>").split()]
        y_decoder_input = [token2idx.get(word, 1) for word in (target_sent_split[0]).split()]

        topic_sent_split = topic_wd.split(u"</d>")
        y_topic_wd = [token2idx.get(word, 1) for word in topic_sent_split[0].split()]  #topic word wo embedding no ta me

        if (len(y) <= hp.maxlen) and (len(x) <= hp.max_turn):
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            ytwd_list.append(np.array(ytwd))
            y_decoder_input_list.append(np.array(y_decoder_input))
            tw_list.append(np.array(y_topic_wd))
            Sources.append(source_sent)
            Targets.append(target_sent)
        else:
            turn_over_count += 1
            print("maxlen!%d",turn_over_count)
            #sys.exit(1)
    
    # Pad      
    X = np.zeros([len(x_list),hp.max_turn, hp.maxlen], np.int32)
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    YTWD = np.zeros([len(ytwd_list), hp.maxlen], np.int32)
    Y_DI = np.zeros([len(y_decoder_input_list), hp.maxlen], np.int32)
    TW = np.zeros([len(tw_list), hp.tw_maxlen], np.int32)
    X_length=np.zeros([len(x_list),hp.max_turn],np.int32)
    for i, (x, y, ytwd, y_decoder_input, tw) in enumerate(zip(x_list, y_list, ytwd_list, y_decoder_input_list, tw_list)):
        for j in range(len(x)):
            if j >= hp.max_turn :
                break
            if len(x[j])<hp.maxlen:
                X[i][j] = np.lib.pad(x[j], [0, hp.maxlen-len(x[j])], 'constant', constant_values=(0, 0))
            else:
                X[i][j]=x[j][:hp.maxlen]
            X_length[i][j] = len(x[j])
        #X[i] = X[i][:len(x)]
        #X_length[i] = X_length[i][:len(x)]
        Y[i] = np.lib.pad(y, [0, hp.maxlen-len(y)], 'constant', constant_values=(0, 0))
        YTWD[i] = np.lib.pad(ytwd, [0, hp.maxlen-len(ytwd)], 'constant', constant_values=(0, 0))
        Y_DI[i] = np.lib.pad(y_decoder_input, [0, hp.maxlen-len(y_decoder_input)], 'constant', constant_values=(0, 0))
        TW[i] = np.lib.pad(tw, [0, hp.tw_maxlen - len(tw)], 'constant', constant_values=(0, 0))

    return X, X_length, Y, YTWD, Y_DI, TW, Sources, Targets

def load_train_data():
    de_sents = [line for line in codecs.open(hp.source_train, 'r', 'utf-8').read().split("\n") if line]
    en_sents = [line for line in codecs.open(hp.target_train, 'r', 'utf-8').read().split("\n") if line]
    topic_wds = [line for line in codecs.open(hp.topic_train, 'r', 'utf-8').read().split("\n") if line]
    
    X, X_length, Y, YTWD, Y_DI, TW, Sources, Targets = create_data(de_sents, en_sents, topic_wds)
    return X, X_length, Y, YTWD, Y_DI, TW, Sources, Targets
    
def load_test_data():
    de_sents = [line for line in codecs.open(hp.source_test, 'r', 'utf-8').read().split("\n") if line]
    en_sents = [line for line in codecs.open(hp.target_test, 'r', 'utf-8').read().split("\n") if line]
    topic_wds = [line for line in codecs.open(hp.topic_test, 'r', 'utf-8').read().split("\n") if line]

    X, X_length, Y, YTWD, Y_DI, TW, Sources, Targets = create_data(de_sents, en_sents, topic_wds)
    return X, X_length, Y, TW, Sources, Targets

def get_batch_data():
    # Load data
    X, X_length, Y, YTWD, Y_DI, TW, Sources, Targets = load_train_data()

    # calc total batch count
    num_batch = len(X) // hp.batch_size
    
    # Convert to tensor
    #X = tf.convert_to_tensor(X, tf.int32)
    #X_length = tf.convert_to_tensor(X_length,tf.int32)
    #Y = tf.convert_to_tensor(Y, tf.int32)
    #YTWD = tf.convert_to_tensor(YTWD, tf.int32)
    #Y_DI = tf.convert_to_tensor(Y_DI, tf.int32)
    #TW = tf.convert_to_tensor(TW, tf.int32)

    # Create Queues
    #input_queues = tf.train.slice_input_producer([X, X_length, Y, YTWD, Y_DI, TW])
            
    # create batch queues
    #x, x_length, y, y_twrp, y_decoder_input, y_tw = tf.train.shuffle_batch(input_queues,
    #                            num_threads=8,
    #                           batch_size=hp.batch_size,
    #                            capacity=hp.batch_size*64,
    #                            min_after_dequeue=hp.batch_size*32,
    #                            allow_smaller_final_batch=False)

    return X, X_length, Y, YTWD, Y_DI, TW, num_batch      # (N, T), (N, T), ()


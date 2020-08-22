# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
class Hyperparams:
    '''Hyperparameters'''
    # data
    source_target_vocab = '../cop/test_data_128_25/corpra_dev_test_final_turn_len50.txt'
    topic_vocab = '../cop/test_data_128_25/corpra_dev_test_final_tp.txt'

    source_train = '../cop/test_data_128_25/corpra_dev_test_final_s.txt'
    target_train = '../cop/test_data_128_25/corpra_dev_test_final_t.txt'
    topic_train = '../cop/test_data_128_25/corpra_dev_test_final_tp.txt'

    source_test = 'corpora/JD.test.query'
    target_test = 'corpora/JD.test.answer'
    topic_test = 'corpora/JD.test.tw'

    source_dev = 'corpora/JD.dev.query'
    target_dev = 'corpora/JD.dev.answer'

    # training
    batch_size = 128 # alias = N
    lr = 0.0003 # learning rate. In paper, learning rate is adjusted to the global step.
    warmup_steps = 4000
    logdir = 'JDlogdir1129' # log directory
    
    # model
    maxlen = 52 # Maximum number of words in a sentence. alias = T.
                # Feel free to increase this if you are ambitious.
    tw_maxlen = 50
    min_cnt = 1 # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 512 # alias = C
    num_blocks = 6 # number of encoder/decoder blocks
    num_epochs = 50
    num_heads = 8
    dropout_rate = 0.1
    num_layers=1
    max_turn=16

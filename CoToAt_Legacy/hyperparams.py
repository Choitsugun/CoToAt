class Hyperparams:
    '''Hyperparameters'''
    # data
    source_target_vocab = '../cop/train_data/corpra_dev_test_final_turn_len50_turn15.txt'
    topic_vocab = '../cop/train_data/corpra_dev_test_final_tp.txt'


    source_train = '../cop/train_data/corpra_dev_test_final_s.txt'
    target_train = '../cop/train_data/corpra_dev_test_final_t.txt'
    topic_train = '../cop/train_data/corpra_dev_test_final_tp.txt'

    #source_train = '../cop/test_data_128_25/corpra_dev_test_final_s.txt'
    #target_train = '../cop/test_data_128_25/corpra_dev_test_final_t.txt'
    #topic_train = '../cop/test_data_128_25/corpra_dev_test_final_tp.txt'


    source_test = '../cop/test_data/corpra_dev_test_final_s.txt'
    target_test = '../cop/test_data/corpra_dev_test_final_t.txt'
    topic_test = '../cop/test_data/corpra_dev_test_final_tp.txt'


    # training
    batch_size = 64 # alias = N
    batch_size_eval = 2
    lr = 0.0003 # learning rate. In paper, learning rate is adjusted to the global step.
    warmup_steps = 4000
    logdir = 'Cho' # log directory
    
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
    max_turn=15

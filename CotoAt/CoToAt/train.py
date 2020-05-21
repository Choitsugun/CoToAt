
# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
from __future__ import print_function
import tensorflow as tf
import numpy
from hyperparams import Hyperparams as hp
from data_load import get_batch_data, load_de_en_vocab, load_tw_vocab
from modules import *
import os, codecs
from tqdm import tqdm

class Graph():
    def __init__(self, is_training=True, vocab_len=None, tw_vocab_len=None):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                self.x, self.x_length, self.y, self.y_twrp, self.y_decoder_input, self.y_tw, self.num_batch = get_batch_data()  # (N, T)
            else: # inference
                self.x = tf.placeholder(tf.int32, shape=(None,hp.max_turn,hp.maxlen))
                self.x_length = tf.placeholder(tf.int32,shape=(None,hp.max_turn))
                self.y = tf.placeholder(tf.int32, shape=(None, hp.maxlen))

            # define decoder inputs
            self.decoder_inputs = tf.concat((tf.ones_like(self.y_decoder_input[:, :1])*2, self.y_decoder_input[:, :-1]), -1) # 2:<S>
            self.tw_vocab_overlap = tf.constant(vocab_overlap, name='Const', dtype='float32')




            # Encoder
            with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
                ## Word Embedding
                self.enc_embed = get_token_embeddings(tf.reshape(self.x,[-1,hp.maxlen]),
                                      vocab_size=vocab_len,
                                      num_units=hp.hidden_units)

                ## Sentence Embedding(LSTM, Word Positional Encoding include)
                single_cell = tf.nn.rnn_cell.GRUCell(hp.hidden_units)
                self.rnn_cell = tf.nn.rnn_cell.MultiRNNCell([single_cell]*hp.num_layers)
                #print (self.enc_embed.get_shape())
                self.sequence_length=tf.reshape(self.x_length,[-1])
                #print(self.sequence_length.get_shape())
                self.uttn_outputs, self.uttn_states = tf.nn.dynamic_rnn(cell=self.rnn_cell,inputs=self.enc_embed,sequence_length=self.sequence_length, dtype=tf.float32,swap_memory=True)
                self.enc = tf.reshape(self.uttn_states,[hp.batch_size,hp.max_turn,hp.hidden_units])

                # src_masks
                src_masks = tf.math.equal(self.x_length, 0)  # (N, max_turn)

                ## Sentence Positional Encoding
                self.enc += positional_encoding(self.enc, hp.max_turn)
                self.enc = tf.layers.dropout(self.enc, hp.dropout_rate, training=is_training)

                ## Topic word Encoding
                self.tw_embed = get_token_embeddings(self.y_tw,
                                                      vocab_size=vocab_len,
                                                      num_units=hp.hidden_units)

                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                        # self-attention
                        self.enc = multihead_attention(queries=self.enc,
                                                  keys=self.enc,
                                                  values=self.enc,
                                                  key_masks=src_masks,
                                                  num_heads=hp.num_heads,
                                                  dropout_rate=hp.dropout_rate,
                                                  training=is_training,
                                                  causality=False)
                        # feed forward
                        self.enc = ff(self.enc, num_units=[4*hp.hidden_units, hp.hidden_units])



            # Decoder
            with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
                # tgt_masks
                tgt_masks = tf.math.equal(self.decoder_inputs, 0)  # (N, T2)

                ## Word Embedding
                self.dec = get_token_embeddings(self.decoder_inputs,
                                                vocab_size=vocab_len,
                                                num_units=hp.hidden_units)

                ## Positional Encoding
                self.dec += positional_encoding(self.dec, hp.maxlen)
                self.dec = tf.layers.dropout(self.dec, hp.dropout_rate, training=is_training)

                # Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                        # Masked self-attention (Note that causality is True at this time)
                        self.dec = multihead_attention(queries=self.dec,
                                                  keys=self.dec,
                                                  values=self.dec,
                                                  key_masks=tgt_masks,
                                                  num_heads=hp.num_heads,
                                                  dropout_rate=hp.dropout_rate,
                                                  training=is_training,
                                                  causality=True,
                                                  scope="self_attention")

                        # Vanilla attention
                        self.dec = multihead_attention(queries=self.dec,
                                                  keys=self.enc,
                                                  values=self.enc,
                                                  key_masks=src_masks,
                                                  num_heads=hp.num_heads,
                                                  dropout_rate=hp.dropout_rate,
                                                  training=is_training,
                                                  causality=False,
                                                  scope="vanilla_attention")
                        ### Feed Forward
                        self.dec = ff(self.dec, num_units=[4*hp.hidden_units, hp.hidden_units])

                ## Topic Word Attention
                self.future_blindness = multihead_attention(queries=self.dec,
                                                 keys=self.dec,
                                                 values=self.dec,
                                                 key_masks=tgt_masks,
                                                 num_heads=hp.num_heads,
                                                 dropout_rate=hp.dropout_rate,
                                                 training=is_training,
                                                 causality=True,
                                                 scope="self_attention")

                self.twdec = topic_word_attention(queries_hidden=self.future_blindness,
                                                  queries_context=self.enc,
                                                  keys=self.tw_embed,
                                                  dropout_rate=hp.dropout_rate,
                                                  training=is_training,
                                                  scope="topic_word_attention")
                ### Feed Forward
                self.twdec = ff(self.twdec, num_units=[4 * hp.hidden_units, hp.hidden_units],
                                                              scope="topic_word_feedforward")

                self.ct_tw_dec = self.dec + self.twdec

                ### Feed Forward
                self.ct_tw_dec = ff(self.ct_tw_dec, num_units=[4 * hp.hidden_units, hp.hidden_units],
                                scope="tw_context_feedforward")



            # get vocab embedding
            self.embeddings = get_token_embeddings(inputs=None,
                                                   vocab_size=vocab_len,
                                                   num_units=hp.hidden_units,
                                                   get_embedtable=True)

            # Final linear projection (embedding weights are shared)
            self.weights = tf.transpose(self.embeddings)                      # (d_model, vocab_size)
            self.logits_c = tf.einsum('ntd,dk->ntk', self.dec, self.weights)  # (N, T_q, vocab_size)
            self.logits_t = tf.layers.dense(self.ct_tw_dec, tw_vocab_len)     # (N, T_q, tw_vocab_size)

            self.prob_c = tf.nn.softmax(self.logits_c)                        # (N, T_q, vocab_size)
            self.prob_t = tf.nn.softmax(self.logits_t)                        # (N, T_q, tw_vocab_size)
            self.prob_t = tf.einsum('nlt,tv->nlv', self.prob_t, self.tw_vocab_overlap)  # (N, T_q, vocab_size)
            self.prob = self.prob_c + self.prob_t
            self.preds = tf.to_int32(tf.argmax(self.prob, axis=-1))



            if is_training:  
                # Loss_context
                self.y_smoothed_c = label_smoothing(tf.one_hot(self.y, depth=vocab_len))  # (N, T_q, vocab_size)
                self.ce_c = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits_c, labels=self.y_smoothed_c)  # (N, T_q)
                self.nonpadding_c = tf.to_float(tf.not_equal(self.y, 0))  # 0: <pad>　#(N,T_q)
                self.loss_c = tf.reduce_sum(self.ce_c * self.nonpadding_c) / (tf.reduce_sum(self.nonpadding_c) + 1e-7)

                # Loss_topic
                self.y_smoothed_t = label_smoothing(tf.one_hot(self.y_twrp, depth=tw_vocab_len))  # (N, T_q, tw_vocab_size)
                self.ce_t = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits_t, labels=self.y_smoothed_t)  # (N, T_q)
                self.noncost_unk = tf.to_float(tf.not_equal(self.y_twrp, 1))  # 1: <unk>
                self.noncost_pad = tf.to_float(tf.not_equal(self.y_twrp, 0))  # 0: <pad>
                self.noncost_t = self.noncost_unk * self.noncost_pad
                self.loss_t = tf.reduce_sum(self.ce_t * self.noncost_t) / (tf.reduce_sum(self.noncost_t) + 1e-7)

                # Loss
                self.loss = self.loss_t + self.loss_c
                self.global_step = tf.train.get_or_create_global_step()
                self.lr = noam_scheme(hp.lr, self.global_step, hp.warmup_steps)
                self.optimizer = tf.train.AdamOptimizer(self.lr)
                self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)



if __name__ == '__main__':
    # Load vocabulary
    token2idx, idx2token = load_de_en_vocab()
    tw2idx, idx2tw = load_tw_vocab()
    token2idx_len = len(token2idx)
    tw2idx_len = len(tw2idx)

    # Load vocab_overlap
    token_idx_list = []
    con_list = numpy.zeros([4, token2idx_len],dtype='float32')
    for i in range(4, tw2idx_len):
        tw = idx2tw[i]
        token_idx_list.append(token2idx[tw])

    vocab_overlap = numpy.append(con_list, numpy.eye(token2idx_len, dtype='float32')[token_idx_list], axis=0)


    # Construct graph
    g = Graph(True, token2idx_len, tw2idx_len); print("Graph loaded")

    # Start session
    sv = tf.train.Supervisor(graph=g.graph, 
                             logdir=hp.logdir,
                             save_model_secs=0)

    with sv.managed_session() as sess:
        for epoch in range(1, hp.num_epochs+1): 
            if sv.should_stop(): break
            loss=[]

            for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, unit='b'):
                _,loss_step = sess.run([g.train_op, g.loss])
                loss.append(loss_step)

            print("epoch:%03d train_loss:%.5lf\n"%(epoch, np.mean(loss)))

        gs = sess.run(g.global_step)
        sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))

    print("Train Done")
    


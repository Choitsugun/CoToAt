from __future__ import print_function
import tensorflow as tf
import numpy
from hyperparams import Hyperparams as hp
from data_load import get_batch_data, load_de_en_vocab, load_tw_vocab
from modules import *
import os, codecs
from tqdm import tqdm

class Graph():
    def __init__(self, is_training=True, vocab_len=None, tw_vocab_len=None, vocab_overlap=None):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                self.x        = tf.placeholder(tf.int32, shape=(hp.batch_size, hp.max_turn,hp.maxlen))
                self.x_length = tf.placeholder(tf.int32, shape=(hp.batch_size, hp.max_turn))
                self.y        = tf.placeholder(tf.int32, shape=(hp.batch_size, hp.maxlen))
                self.y_twrp   = tf.placeholder(tf.int32, shape=(hp.batch_size, hp.maxlen))
                self.y_tw     = tf.placeholder(tf.int32, shape=(hp.batch_size, hp.tw_maxlen))
                self.y_decoder_input = tf.placeholder(tf.int32, shape=(hp.batch_size, hp.maxlen))
            else:
                # inference
                self.x        = tf.placeholder(tf.int32, shape=(hp.batch_size_eval, hp.max_turn,hp.maxlen))
                self.x_length = tf.placeholder(tf.int32, shape=(hp.batch_size_eval, hp.max_turn))
                self.y        = tf.placeholder(tf.int32, shape=(hp.batch_size_eval, hp.maxlen))
                self.y_tw     = tf.placeholder(tf.int32, shape=(hp.batch_size_eval, hp.tw_maxlen))
                self.y_decoder_input = tf.placeholder(tf.int32, shape=(hp.batch_size_eval, hp.maxlen))
                self.tw_vocab_overlap = tf.constant(vocab_overlap, name='Const', dtype='float32')

            # define decoder inputs
            self.decoder_inputs = tf.concat((tf.ones_like(self.y_decoder_input[:, :1])*2, self.y_decoder_input[:, :-1]), -1) # 2:<S>

            ## Word Embedding
            self.enc_embed = get_token_embeddings(tf.reshape(self.x, [-1, hp.maxlen]),
                                                  vocab_size=vocab_len,
                                                  num_units=hp.hidden_units)

            ## Topic Word Embedding
            self.tw_embed = get_token_embeddings(self.y_tw,
                                                 vocab_size=vocab_len,
                                                 num_units=hp.hidden_units)

            ## Word Embedding
            self.dec_embed = get_token_embeddings(self.decoder_inputs,
                                                  vocab_size=vocab_len,
                                                  num_units=hp.hidden_units)

            # Get Vocab Embedding
            self.embeddings = get_token_embeddings(inputs=None,
                                                   vocab_size=vocab_len,
                                                   num_units=hp.hidden_units,
                                                   get_embedtable=True)



            # Encoder
            with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
                # Hierarchical Self-Attention reshape
                self.x_resha = tf.reshape(self.x, [-1, hp.maxlen]) # (N, S_maxlen)

                # Word src_masks
                src_masks_w = tf.math.equal(self.x_resha, 0)  # (N, S_maxlen)

                ## Word Positional Encoding
                self.enc = self.enc_embed + positional_encoding(self.enc_embed, hp.maxlen)
                self.enc = tf.layers.dropout(self.enc, hp.dropout_rate, training=is_training)

                ## Word Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_w{}".format(i), reuse=tf.AUTO_REUSE):
                        # self-attention
                        self.enc, self.att_w = multihead_attention(queries=self.enc,
                                                       keys=self.enc,
                                                       values=self.enc,
                                                       key_masks=src_masks_w,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       training=is_training,
                                                       causality=False)
                        # feed forward
                        self.enc = ff(self.enc, num_units=[4 * hp.hidden_units, hp.hidden_units])

                # Hierarchical Self-Attention reshape
                self.enc = tf.reshape(self.enc, [hp.batch_size, hp.max_turn, hp.maxlen, hp.hidden_units])
                self.enc = tf.reduce_sum(self.enc, axis=2)  # (N,max_turn,C)

                # Utterance which has been padded makes the Utterance vector for 0 regardless of self-attention
                x_length_mat = tf.not_equal(self.x_length, 0)  # (N, max_turn)
                x_length_mat = tf.expand_dims(x_length_mat, -1)  # (N, max_turn, 1)
                x_length_mat = tf.tile(x_length_mat, multiples=[1, 1, hp.hidden_units])  # (N, max_turn, C)
                zeros_mat = tf.zeros([hp.batch_size, hp.max_turn, hp.hidden_units], dtype=tf.float32)
                self.enc = tf.where(x_length_mat, self.enc, zeros_mat)


                # Uatterance src_masks
                src_masks_u = tf.math.equal(self.x_length, 0)  # (N, max_turn)

                ## Uatterance Positional Encoding
                self.enc_embed = tf.reshape(self.enc_embed, [hp.batch_size, hp.max_turn, hp.maxlen, hp.hidden_units])
                self.enc_embed = tf.reduce_sum(self.enc_embed, axis=2)  # (N,max_turn,C)
                self.enc = self.enc + positional_encoding(self.enc_embed, hp.max_turn)
                self.enc = tf.layers.dropout(self.enc, hp.dropout_rate, training=is_training)

                ## Uatterance Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_u{}".format(i), reuse=tf.AUTO_REUSE):
                        # self-attention
                        self.enc, self.att_u = multihead_attention(queries=self.enc,
                                                  keys=self.enc,
                                                  values=self.enc,
                                                  key_masks=src_masks_u,
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

                ## Positional Encoding
                self.dec = self.dec_embed + positional_encoding(self.dec_embed, hp.maxlen)
                self.dec = tf.layers.dropout(self.dec, hp.dropout_rate, training=is_training)

                # Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                        # Masked self-attention (Note that causality is True at this time)
                        self.dec, _ = multihead_attention(queries=self.dec,
                                                  keys=self.dec,
                                                  values=self.dec,
                                                  key_masks=tgt_masks,
                                                  num_heads=hp.num_heads,
                                                  dropout_rate=hp.dropout_rate,
                                                  training=is_training,
                                                  causality=True,
                                                  scope="self_attention")

                        # Vanilla attention
                        self.dec, self.att_v = multihead_attention(queries=self.dec,
                                                  keys=self.enc,
                                                  values=self.enc,
                                                  key_masks=src_masks_u,
                                                  num_heads=hp.num_heads,
                                                  dropout_rate=hp.dropout_rate,
                                                  training=is_training,
                                                  causality=False,
                                                  scope="vanilla_attention")
                        ### Feed Forward
                        self.dec = ff(self.dec, num_units=[4*hp.hidden_units, hp.hidden_units])

                ## Topic Word Attention
                self.future_blindness, _ = multihead_attention(queries=self.dec,
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



            # Final linear projection (embedding weights are shared)
            self.weights = tf.transpose(self.embeddings)                      # (d_model, vocab_size)
            self.logits_c = tf.einsum('ntd,dk->ntk', self.dec, self.weights)  # (N, T_q, vocab_size)
            self.logits_t = tf.layers.dense(self.ct_tw_dec, tw_vocab_len)     # (N, T_q, tw_vocab_size)



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
            else:
                # inference
                self.prob_c = tf.nn.softmax(self.logits_c)  # (N, T_q, vocab_size)
                self.prob_t = tf.nn.softmax(self.logits_t)  # (N, T_q, tw_vocab_size)
                self.prob_t = tf.einsum('nlt,tv->nlv', self.prob_t, self.tw_vocab_overlap)  # (N, T_q, vocab_size)
                self.prob = self.prob_c + self.prob_t  # (N, T_q, vocab_size)
                self.preds = tf.to_int32(tf.argmax(self.prob, axis=-1))  # (N, T_q)

                self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=vocab_len))  # (N, T_q, vocab_size)
                self.ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.prob,labels=self.y_smoothed)  # (N, T_q)
                self.ppl_step = tf.exp(self.ce)  # (N, T_q)



if __name__ == '__main__':
    # Load vocabulary
    token2idx, idx2token = load_de_en_vocab()
    tw2idx, idx2tw = load_tw_vocab()
    token2idx_len = len(token2idx)
    tw2idx_len = len(tw2idx)

    X, X_length, Y, YTWD, Y_DI, TW, num_batch = get_batch_data()

    # Construct graph
    g = Graph(True, token2idx_len, tw2idx_len, None)
    print("Graph loaded")

    # Start session
    sv = tf.train.Supervisor(graph=g.graph, 
                             logdir=hp.logdir,
                             save_model_secs=0)


    with sv.managed_session() as sess:
        for epoch in range(1, hp.num_epochs+1): 
            if sv.should_stop(): break
            loss=[]

            for step in tqdm(range(num_batch), total=num_batch, ncols=100, unit='b'):
                x =               X[step * hp.batch_size: (step + 1) * hp.batch_size]
                x_length = X_length[step * hp.batch_size: (step + 1) * hp.batch_size]
                y =               Y[step * hp.batch_size: (step + 1) * hp.batch_size]
                y_twrp =       YTWD[step * hp.batch_size: (step + 1) * hp.batch_size]
                y_tw =           TW[step * hp.batch_size: (step + 1) * hp.batch_size]
                y_decoder_input = Y_DI[step * hp.batch_size: (step + 1) * hp.batch_size]

                _,loss_step = sess.run([g.train_op, g.loss],
                                       {g.x:x, g.x_length:x_length, g.y:y, g.y_twrp:y_twrp, g.y_tw:y_tw, g.y_decoder_input:y_decoder_input})
                loss.append(loss_step)

            print("epoch:%03d train_loss:%.5lf\n"%(epoch, np.mean(loss)))

            gs = sess.run(g.global_step)
            sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))

    print("Train Done")
    


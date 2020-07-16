from __future__ import print_function
import codecs
import os
import tensorflow as tf
import numpy as np
from hyperparams import Hyperparams as hp
from data_load import load_test_data, load_de_en_vocab, load_tw_vocab
from train import Graph
from nltk.translate.bleu_score import corpus_bleu

def eval():
    # Load vocabulary
    token2idx, idx2token = load_de_en_vocab()
    tw2idx, idx2tw = load_tw_vocab()
    token2idx_len = len(token2idx)
    tw2idx_len = len(tw2idx)

    # Load vocab_overlap
    token_idx_list = []
    con_list = np.zeros([4, token2idx_len],dtype='float32')
    for i in range(4, tw2idx_len):
        tw = idx2tw[i]
        token_idx_list.append(token2idx[tw])

    vocab_overlap = np.append(con_list, np.eye(token2idx_len, dtype='float32')[token_idx_list], axis=0)

    # Load graph
    g = Graph(False, token2idx_len, tw2idx_len, vocab_overlap)
    print("Graph loaded")
    
    # Load data
    X, X_length, Y, TW, Sources, Targets = load_test_data()
     
    # Start session         
    with g.graph.as_default():    
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ## Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")
              
            ## Get model name
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1] # model name
            #fftmp=open("tmp.txt","w")

            ## Inference
            if not os.path.exists('results'): os.mkdir('results')

            with codecs.open("results/" + mname, "w", "utf-8") as fout:
                list_of_refs, hypotheses = [], []
                for i in range(len(X) // hp.batch_size_eval):
                    ### Get mini-batches
                    x =              X[i * hp.batch_size_eval: (i + 1) * hp.batch_size_eval]
                    x_length= X_length[i * hp.batch_size_eval: (i + 1) * hp.batch_size_eval]
                    y =              Y[i * hp.batch_size_eval: (i + 1) * hp.batch_size_eval]
                    y_tw =          TW[i * hp.batch_size_eval: (i + 1) * hp.batch_size_eval]
                    sources =  Sources[i * hp.batch_size_eval: (i + 1) * hp.batch_size_eval]
                    targets =  Targets[i * hp.batch_size_eval: (i + 1) * hp.batch_size_eval]
                    #fftmp.write("%s\n"%(" ".join(str(w) for w in x[0][0]).encode("utf-8")))
                    #fftmp.write("%s\n"%(sources[0].encode("utf-8")))
                    #fftmp.write("%s\n"%(' '.join(str(w) for w in x_length)))
                    #print (sources)
                    #print (targets) 
                    ### Autoregressive inference
                    preds = np.zeros((hp.batch_size_eval, hp.maxlen), np.int32)
                    ppls = np.zeros((hp.batch_size_eval, hp.maxlen), np.int32)
                    for j in range(hp.maxlen):
                        _preds, ppl_step, att_w, att_u, att_v = sess.run([g.preds, g.ppl_step, g.att_w, g.att_u, g.att_v],
                                                    {g.x:x, g.x_length:x_length, g.y:y, g.y_tw:y_tw, g.y_decoder_input:preds})
                        preds[:, j] = _preds[:, j]
                        ppls[:, j] = ppl_step[:, j]
                     
                    ### Write to file
                    for source, target, pred, ppl in zip(sources, targets, preds, ppls): # sentence-wise
                        got = " ".join(idx2token[idx] for idx in pred).split("</S>")[0].strip()
                        ppl_score = " ".join(score for score in ppl).strip()
                        fout.write("- source: " + source +"\n")
                        fout.write("- expected: " + target + "\n")
                        fout.write("- got: " + got + "\n\n")
                        fout.write("- ppl_score: " + ppl_score + "\n\n")
                        fout.flush()
                          
                        # bleu score
                        ref = target.split()
                        hypothesis = got.split()
                        if len(ref) > 3 and len(hypothesis) > 3:
                            list_of_refs.append([ref])
                            hypotheses.append(hypothesis)

                    ## Calculate bleu score
                    fout.write("- att_w: " + att_w + "\n")
                    fout.write("- att_u: " + att_u + "\n")
                    fout.write("- att_v: " + att_v + "\n")
                    fout.flush()

                ## Calculate bleu score
                score = corpus_bleu(list_of_refs, hypotheses)
                fout.write("Bleu Score = " + str(100*score))


if __name__ == '__main__':
    eval()
    print("Done")
    
    

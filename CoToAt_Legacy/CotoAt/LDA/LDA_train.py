import codecs
from gensim import corpora, models, similarities
import re



f = codecs.open('../cop/corpra_dev_test_final_turn_len50.txt','r','utf-8')
STOP_WORDS = set([w.strip() for w in codecs.open('../cop/characters/stop_words.txt','r','utf-8').readlines()])

# 过滤词长，过滤停用词，只保留中文
def is_fine_word(word, min_length=2):
    rule = re.compile(r"^[\u4e00-\u9fa5]+$")
    if len(word) >= min_length and word not in STOP_WORDS and re.search(rule, word):
        return True
    else:
        return False


line_list = []
texts = []
for line in f:
    line_list = line.split()
    txt = []
    for i in line_list:
       if is_fine_word(i):
           txt.append(i)
    texts.append(txt)

# load id->word mapping (the dictionary)
dictionary = corpora.Dictionary(texts)

# no more than 5% documents
#dictionary.filter_extremes(no_above=0.05)

# save dictionary
dictionary.save('dict_v1.dict')

# load corpus
corpus = [dictionary.doc2bow(text) for text in texts]

# extract 100 LDA topics, using 1 pass and updating once every 1 chunk (300000 documents), using 150 iterations
lda = models.LdaModel(corpus=corpus, id2word=dictionary, chunksize=2000, num_topics=100, iterations=700)

# save model to files
lda.save('mylda_v1.pkl')

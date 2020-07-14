import codecs
from gensim import corpora, models, similarities
from gensim.models import CoherenceModel, LdaModel
import re

f = codecs.open('../cop/corpra_dev_test_final_turn_len50.txt','r','utf-8')
STOP_WORDS = set([w.strip() for w in codecs.open('../cop/characters/stop_words.txt','r','utf-8').readlines()])

lda = models.LdaModel.load('mylda_v1.pkl')
lda100 = models.LdaModel.load('./topic100/mylda_v1.pkl')
dictionary = corpora.Dictionary.load('dict_v1.dict')


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


corpus = [dictionary.doc2bow(text) for text in texts]

lda_100 = CoherenceModel(model=lda100, corpus=corpus,
       dictionary=dictionary, coherence='u_mass')
lda_200 = CoherenceModel(model=lda, corpus=corpus,
       dictionary=dictionary, coherence='u_mass')

print(lda_100.get_coherence())
print(lda_200.get_coherence())

lda_100 = CoherenceModel(model=lda100, texts=texts,
    dictionary=dictionary,  coherence='c_v')
lda_200 = CoherenceModel(model=lda, texts=texts,
    dictionary=dictionary,  coherence='c_v')

print(lda_100.get_coherence())
print(lda_200.get_coherence())

lda_100 = CoherenceModel(model=lda100, texts=texts,
    dictionary=dictionary,  coherence='c_uci')
lda_200 = CoherenceModel(model=lda, texts=texts,
    dictionary=dictionary,  coherence='c_uci')

print(lda_100.get_coherence())
print(lda_200.get_coherence())

lda_100 = CoherenceModel(model=lda100, texts=texts,
    dictionary=dictionary,  coherence='c_npmi')
lda_200 = CoherenceModel(model=lda, texts=texts,
    dictionary=dictionary,  coherence='c_npmi')

print(lda_100.get_coherence())
print(lda_200.get_coherence())

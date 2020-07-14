from gensim import corpora, models, similarities
import codecs
import re

# load model and dictionary
dictionary = corpora.Dictionary.load('./topic700/dict_v1.dict')
lda700 = models.LdaModel.load('./topic700/mylda_v1.pkl')

# print the most contributing words for 100 randomly selected topics
#print(lda700.show_topic(8,200))

for i in range(100):
    print ('主题:',i)
    for word, prob in lda700.show_topic(i, 50):
        print (word, prob)

f = codecs.open('../cop/corpra_dev_test_final_turn_len50.txt','r','utf-8')
STOP_WORDS = set([w.strip() for w in codecs.open('../cop/characters/stop_words.txt','r','utf-8').readlines()])
fhightopic = codecs.open('./topic700/topic.txt','w','utf-8')

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

#print(lda700[corpus[1158]])
#print(lda700.get_document_topics(corpus[1158]))

#print(lda700[corpus[1159]])
#print(lda700.get_document_topics(corpus[1159]))

#print(lda700.get_document_topics(corpus))

for i in lda700.get_document_topics(corpus)[:]:
    listj=[]

    for j in i:
        listj.append(j[1])
    if len(i) > 0:
        bz=listj.index(max(listj))
        fhightopic.write(str(i[bz][0]))
        fhightopic.write("\r\n")
    else:
        fhightopic.write("UNK")
        fhightopic.write("\r\n")
    #print(i[bz][0])


import json
import operator
import re
import glob
import os
from gensim.models import word2vec
from gensim.models.phrases import Phraser
from gensim.models import Phrases
import multiprocessing
import string
import word2phrase

#initialize an empty list
full_content=list()

for filename in glob.glob('*.json'):
        data=[json.loads(line) for line in open(filename,'r')]
        full_content+=data

#############
def get_sentences_list(full_content):
        raw_data=list()
        for block in full_content:
                if'abstract' in block:
                        raw_data.append(block['abstract'].strip().lower())
        return raw_data

#############
def sentence_to_worldlist(string):
        temp =re.sub("[^a-zA-Z]"," ",string)
        return temp.split()

#############
#extract abstract content
abstract_text = get_sentences_list(full_content)
#convert a sentence into worldlist
sentences=[]
for sent in abstract_text:
        if len(sent)>0:
                sentences.append(sentence_to_worldlist(sent))

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s: %(message)s',level=logging.INF
O)

num_features =300
min_word_count =3
num_workers = multiprocessing.cpu_count()
context_size=10
downsampling =1e-3
seed =1
"""
modelx= word2vec.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)
modelx.build_vocab(sentences)
modelx.train(sentences,total_examples=modelx.corpus_count,epochs=15)
modelx.save('trained_model_epoch15')"""
print("Bigram Model Start Training..........")

phrase=Phrases(sentences, min_count=2, threshold=3, delimiter=b' ')
bigram=Phraser(phrase)

model_bigram=word2vec.Word2Vec(

    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling



)
model_bigram.build_vocab(bigram[sentences])
model_bigram.train(bigram[sentences],total_examples=model_bigram.corpus_count,epochs=5)
model_bigram.save("lowercase_bigram_model_epoch5")

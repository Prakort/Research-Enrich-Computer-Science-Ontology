#!/usr/bin/env python
# coding: utf-8

# In[13]:


from gensim.models import Word2Vec as w2v


# In[14]:


# load bigram model
model = w2v.load('lowercase_bigram_model_epoch5')


# In[15]:


# get a topic and its result as dictionary
file = open('CSO-clean.txt','r')
topic_result=dict()
for line in file:
    w = line.split()
    #topic predicate object
    #w[0] = topic
    #w[2] = object: result
    if w[0] in topic_result:
        topic_result[w[0]].append(w[2])
    else:
        topic_result[w[0]]=[w[2]]

#topic_result


# In[12]:


len(topic_result)


# In[16]:


# iterate over topic dictionary
# search each topic in bigram model
# add the results to a new Dictionary
def w2vResult(topN,model):
    w2v_result=dict()
    for topic in topic_result:
        #word = topic.replace('_',' ')
        word = topic
        result=[]
        try:
            result = model.wv.most_similar(positive=[word],topn=topN)
        except:
            result=[] # no result found
        w2v_result[topic]=result
    return w2v_result


# In[34]:


# compare two Dicts where they have the same key but different values
# write result into a new file
def compareResult(filename,topN,w2v_result,topic_result):
    compare=open(filename,'a+')
    sum=0
    for topic in topic_result:
        # write header
        compare.write('CSO: '+topic+'\t\t\t'+'WORD2VEC: '+topic+'\n\n')
        print('CSO: '+topic+'\t\t\t'+'WORD2VEC: '+topic+'\n\n')
        
        # get value of the topic from CSO dict
        cso=topic_result[topic]
        
        # get value of the topic from word2vec dict
        w2v=w2v_result[topic]
        
        # word2vec result format is a turple 
        # we will change it to a list easy to iterate
        wordlist=[]
        for re in w2v:
            word=re[0]
            wordlist.append(word)
        counter =0
        import itertools
        # use itertools.zip to print out both data as 2 column
        for a, b in itertools.zip_longest(cso, wordlist, fillvalue=''):   
            tempA = a
            tempB = b
            if tempA in wordlist:
                a+=' Matched found'
            if tempB in cso:
                counter+=1
                b+=' Matched found'
            compare.write("{0:50s}{1:25s}\n".format(a, b))
            #print("{0:50s}{1:25s}\n".format(a, b))
        sum+=counter
        compare.write('RESULT of PRECISON:'+str(counter)+'/'+str(topN))
        compare.write('\n\n')
        #print('RESULT of PRECISON:'+str(counter)+'/'+str(topN))
        #print('\n\n')
    #print('Overall Precision is :'+str(sum/len(w2v_result))+'/'+str(topN))
    compare.write('Overall Precision is :'+str(sum/len(w2v_result))+'/'+str(topN))


# In[24]:


#top 5
w2v_topn5 = w2vResult(5,model)
#print(w2v_topn5)
#compareResult('precisionTOP5.txt',5,w2v_topn5,topic_result)
compareResult('precisionTOP5.txt',5,w2v_topn5,topic_result)


# In[ ]:


topn =[200,300,400]
for n in topn:
    filename = 'precision'+str(n)+'.txt'
    w2v_result = w2vResult(n,model)
    compareResult(filename,n,w2v_result,topic_result)


# In[ ]:





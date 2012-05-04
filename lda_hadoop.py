'''
Created on 2011-5-30
@author: cyzhang9@mail.ustc.edu.cn 
http://code.google.com/p/semi-supervised-lda/
'''

#!/usr/bin/python
# coding=utf-8
import os
import sys
import dumbo
from dumbo.lib import identitymapper, identityreducer
from math import *
import random


def setdict2d(dic,key1,key2,value):
    if not dic.has_key(key1):
        dic[key1]={}
    dic[key1][key2]=value

def load_ldamodel(file_name):
    global num_topics
    global wordmap
    global word_sum
    global sum
    for line in open(file_name):
        sep = line.split("\t")
        if 2 != len(sep):
            continue
        word = sep[0]
        sep = sep[1].split()
        if num_topics == 0:
            num_topics = len(sep)
            for i in range(num_topics):
                sum.append(0.0)
        for i in range(len(sep)):
            if float(sep[i]) > 0:
                if not wordmap.has_key(word):
                    wordmap[word] = []
                wordmap[word].append((i,float(sep[i])))
                word_sum[word] = word_sum.get(word,0) + float(sep[i])
                sum[i] += float(sep[i])

filetype = int(os.environ.get('FILETYPE',0))
total_iterations = int(os.environ.get('TITER',20))
burn_in_iterations = int(os.environ.get('BITER',10))
beta = float(os.environ.get('BETA',0.01))
alpha = float(os.environ.get('ALPHA',0.1))
ldamodelfile = os.environ.get('LDAMODELFILE')

num_topics = 0
num_word = 0
wordmap = {}
sum = []
word_sum = {}


if ldamodelfile:
    load_ldamodel(ldamodelfile)
    num_word = len(wordmap)
    Vbeta = num_word * beta
    Kalpha = num_topics * alpha

def mapper(key, value):
    id_felist=value.strip().split('\t')
    if 2 != len(id_felist):
        return
    felist = id_felist[1].split()
    if len(felist) == 0:
        return

    j = 0
    doc_term = []
    while j < len(felist):
        if 0 == filetype:
            for n in range(int(felist[j + 1])):
                if wordmap.has_key(felist[j]):
                    doc_term.append((felist[j],-1))
            j += 2
        else:
            if wordmap.has_key(felist[j]):
                doc_term.append((felist[j],-1))
            j += 1

    if len(doc_term) == 0:
        return

    doc2topic_plsa = {}
    topicsum_plsa = 0.0
    for (k, v) in doc_term:
        l = wordmap[k]
        for (tp, sc) in wordmap[k]:
            topicsum_plsa += sc
            doc2topic_plsa[tp] = doc2topic_plsa.get(tp,0) + sc / word_sum[k]

    term2topic = {}
    doc2topic = {}
    for m,(k, v) in enumerate(doc_term):
        distribution_sum = 0.0
        for (tp, sc) in wordmap[k]:
            if doc2topic_plsa.has_key(tp):
                term2topic[tp] = ((sc + beta) / (sum[tp] + Vbeta)) * (doc2topic_plsa[tp] + alpha)
            else:
                term2topic[tp] = ((sc + beta) / (sum[tp] + Vbeta)) * alpha
            distribution_sum += term2topic[tp]
        choice = random.uniform(0, distribution_sum)
        sum_so_far = 0.0
        for (tp, sc) in term2topic.items():
            sum_so_far += sc
            if sum_so_far >= choice:
                doc_term[m] = (k, tp)
                doc2topic[tp] = doc2topic.get(tp,0) + 1
                break

    sumdoc2topic = 0.0
    prob_dist={}
    for iter in range(total_iterations):
        for m,(k, v) in enumerate(doc_term):
            doc2topic[v] -= 1
            doc_term[m] = (k, -1)
            term2topic = {}
            distribution_sum = 0.0
            for (tp, sc) in wordmap[k]:
                if doc2topic.has_key(tp):
                    term2topic[tp] = ((sc + beta) / (sum[tp] + Vbeta)) * (doc2topic[tp] + alpha)
                else:
                    term2topic[tp] = ((sc + beta) / (sum[tp] + Vbeta)) * alpha
                distribution_sum += term2topic[tp]
            choice = random.uniform(0, distribution_sum)
            sum_so_far = 0.0
            for (tp, sc) in term2topic.items():
                sum_so_far += sc
                if sum_so_far >= choice:
                    doc_term[m] = (k, tp)
                    doc2topic[tp] = doc2topic.get(tp,0) + 1
                    sum[tp] += 1
                    break

        if iter >= burn_in_iterations:
            for (tp, sc) in doc2topic.items():
                sumdoc2topic += sc
                if prob_dist.has_key(tp):
                    prob_dist[tp] += sc
                else:
                    prob_dist[tp] = sc

    docintopic_distrib_lst=[]
    for (tp, sc) in prob_dist.items():
        if sc > 0:
            docintopic_distrib_lst.append(str(tp) + ":" + str(sc/sumdoc2topic))

    yield key," ".join(docintopic_distrib_lst)


def runner(job):
    job.additer(mapper, identityreducer)

if __name__ == "__main__":
    dumbo.main(runner)

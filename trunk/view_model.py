#!/usr/bin/python2.4
# Print a readable text of the model
# ./view_model.py model_file viewable_file

import sys, os
num_topics = 0
map = []
tagmap = []
sum = []
tagsum=[]
word_sum = {}
for line in open(sys.argv[1]):
    sep = line.split("\t")
    word = sep[0]
    sep = sep[1].split()
    if num_topics == 0:
        num_topics = len(sep)
        for i in range(num_topics):
            map.append({})
            tagmap.append({})
            sum.append(0.0)
            tagsum.append(0.0)
    for i in range(len(sep)):
        if float(sep[i]) > 1:
            if word.startswith('tag_'):
                tagmap[i][word] = float(sep[i])
                tagsum[i] += float(sep[i])
            else:
                map[i][word] = float(sep[i])
                sum[i] += float(sep[i])

            if word_sum.has_key(word):
                word_sum[word] += float(sep[i])
            else:
                word_sum[word] = float(sep[i])


for i in range(len(map)):
    x = sorted(map[i].items(), key=lambda(k, v):(v, k), reverse = True)
    y = sorted(tagmap[i].items(), key=lambda(k, v):(v, k), reverse = True)

    print

    ctr=tagsum[i] * 10000/(sum[i] + tagsum[i])
    print "TOPIC: ", i, "feature:",int(sum[i]),"click:",int(tagsum[i]),"ctr*1W:", "%.5f " % ctr
    print

    for key in y:
        print key[0], i, int(key[1]), int(word_sum[key[0]]), "%.5f " % (key[1]/word_sum[key[0]]), "%.5f " % (key[1] * ctr/tagsum[i])

    print

    for key in x:
        print key[0], i,int(key[1]), int(word_sum[key[0]]), "%.5f " % (key[1]/word_sum[key[0]]), "%.5f " % (key[1] * ctr/sum[i])

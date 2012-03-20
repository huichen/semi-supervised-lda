'''
Created on 2011-5-30
@author: cyzhang
'''

import re
import sys,os

def dofeaindex(file,filetype):
    feamap = {}
    linenum = 0
    for line in open(file):
        line = line.strip()
        content = line.split('\t')
        if len(content) != 2:
            continue
        felist=content[1].split(' ')
        if len(felist) == 0:
            continue
        j = 0
        while j < len(felist):
            if 0 == filetype:
                feamap[felist[j]] = feamap.get(felist[j],0) + int(felist[j + 1])
                j += 2
            else:
                feamap[felist[j]] = feamap.get(felist[j],0) + 1
                j += 1
        linenum += 1
    return feamap,linenum
        
if __name__ == "__main__":
    inputfile = sys.argv[1]
    outputfile = sys.argv[2]
    filetype = int(sys.argv[3])
    cutoff = int(sys.argv[4])

    myfile = open(outputfile , 'w')

    feamap,linenum = dofeaindex(inputfile,filetype)
    id = 0
    for fea,num in feamap.items():
        if num > cutoff:
            myfile.write(str(id) + ' ' + fea + '\n')
            id += 1

    myfile.close()
cp ./test/lda_model_0-final.txt ./
ldamodelfile=lda_model_0-final.txt
hadoop fs -copyFromLocal test/test.dat ./hadooptest.txt
hadoop fs -rmr ./testout
dumbo start lda_hadoop.py -hadoop /usr/lib/hadoop -input hadooptest.txt -output ./testout -file ${ldamodelfile} -cmdenv "LDAMODELFILE=${ldamodelfile}"

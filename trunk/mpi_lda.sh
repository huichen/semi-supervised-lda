ldapath=/home/ldap/zhang/plda
testpath=/home/ldap/zhang/plda/test
python getwordindexfile.py $testpath/test.dat $testpath/wordindex.txt 0 20
mpiexec -n 8 $ldapath/mpi_lda --num_topics 100 --alpha 0.1 --beta 0.01 --training_data_file $testpath/test.dat --topic_distribution_file $testpath/topic_ --topic_assignments_file $testpath/assignments_ --model_file $testpath/lda_model_ --word_index_file $testpath/wordindex.txt --total_iterations 2000 --save_step 500 --file_type 0
python ./view_model.py $testpath/lda_model_0-final.txt > $testpath/viewable_file.txt
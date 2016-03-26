本工具修改自PLDA (code.google.com/p/plda)：增加了在种子词基础上半监督学习的功能，可以实现对已有的topic基于规则的拆分，同时优化了原先工具的内存占用。<br />
## 增加功能 ##
### 增加功能1： ###
> 使用人工定义的种子词，拆分原先的topic结果，实现半监督学习。<br />
> LDA是个非监督的学习工具，最终得到的topic结果，在实际使用中常需要对部分TOPIC进行更细拆分，并希望在拆分过程中可以通过加入种子词实现半监督的学习。<br />
> 本工具可以在非监督学习完成模型后，使用TOPIC规则文件在原先的模型基础上实现半监督学习。<br />
> TOPIC规则格式：<br />
> Topic id1 -> id2 种子词1 种子词2 种子词3 .... 种子词n<br />
> > id1：需要拆分的TOPIC的id<br />
> > id2：拆分后对应的TOPIC的id<br />

> 注意:种子词的个数可以任意(包括0)<br />
> > 拆分后原先TOPIC已经不复存在。在定义id2时需要保证拆分后TOPIC id的连续。<br />

> 如：<br />
> 在无监督的LDA结果中 TOPIC 12 是数码类，<br />
> 现在希望对这个TOPIC拆分成 TOPIC a： 手机， TOPIC b： 电脑， TOPIC c： 其他<br />
> 原先共100个TOPIC<br />
> TOPIC规则文件样例：<br />
> Topic 12 -> 12 手机 3G服务 移动通信<br />
> Topic 12 -> 100 台式机 平板电脑 笔记本电脑<br />
> Topic 12 -> 101<br />
> 半监督学习后，有102个Topic，即将原先Topic 12 拆分为  Topic 12，Topic 100，Topic 101。<br />
### 增加功能2： ###
> 在获得新的训练数据后，可以根据已有模型将新语料中新出现的term，分配到原先模型的各个topic上，并保持原先的term的分布不发生变化。需要传入 新词文件，格式为每行一个词<br />
### 增加功能3： ###
> 结合使用TOPIC规则文件和新词文件，可以实现基于种子词的半监督学习。<br />
> 方法：在TOPIC规则文件中为每个topic定义种子词，将种子词之外的词全部加入新词文件。通过合适定义TOPIC规则文件就可以学习出自己想要的topic<br />
> 如：为每个topic定义TOPIC规则：<br />
> Topic 0 -> 0 考试 成绩 试题 英语 报名<br />
> Topic 1 -> 1 直播 卫视 在线 电视 cctv 新闻<br />
> Topic 2 -> 2 团购 拉手	套餐 优惠<br />
> ....<br />
> 新词文件：<br />
> 中心<br />
> 国庆<br />
> 个人<br />
> 最好<br />
> ....<br />
### 增加功能4： ###
> 在实际使用中的，更多的情况是利用一个已有的模型对一批海量文本进行topic预测。这里给出了一个在hadoop下使用MapReduce程序进行分布式计算的python脚本，需要安装hadoop，dumbo和python工具。<br />

## 优化了内存 ##
> 对文档中的term做了索引，降低了内存占有(根据实际情况有5倍到20倍的优化)，因此在训练时需要传入term的索引文件。并可以通过这个索引文件实现特征的灵活选取如过滤停用词或设置cutoff<br />
## 输入文件格式和输出文件有改动。(见下) ##
<br />

# 使用方法： #
## 生成term索引文件 ##
python getwordindexfile.py $testpath/test.dat $testpath/wordindex.txt 0 20<br />
四个参数分别为：输入文件，输出文件，文件格式，cutoff值<br />
## 非监督学习 ##
示例脚本：mpi\_lda.sh<br />
mpiexec -n 8 $ldapath/mpi\_lda --num\_topics 100 --alpha 0.1 --beta 0.01 --training\_data\_file $testpath/test.dat --topic\_distribution\_file $testpath/topic_--topic\_assignments\_file $testpath/assignments_ --model\_file $testpath/lda\_model_--word\_index\_file $testpath/wordindex.txt --total\_iterations 2000 --save\_step 500 --file\_type 0_<br />
python ./view\_model.py $testpath/lda\_model\_0-final.txt > $testpath/viewable\_file.txt<br />
变量<br />
$ldapath: mpi\_lda程序存放的路径<br />
$testpath: 工作目录，训练数据和输出结果都放在这个目录。<br />
输入文件：<br />
--training\_data\_file: 训练数据<br />
> docname word1 word1\_count word2 word2\_count word3 word3\_count<br />
> docname为该行文档对于的文档名，可以是id号，与后面的内容用table分割。这个改动更加易于模型结果的分析。<br />
--word\_index\_file：term的索引文件<br />
> id term<br />
输出文件：<br />
--topic\_distribution\_file：文档上各个topic的分布,这是文件名的前缀,文件名如topic\_0-001000.txt，表示第0个进程上第1000次迭代的结果。<br />
--topic\_assignments\_file：文档中各个term被分配的topic<br />
--model\_file：模型文件,term上各个topic的分布<br />
参数：<br />
--alpha：LDA模型的超参数<br />
--beta：LDA模型的超参数<br />
--num\_topics: topic的数量<br />
--total\_iterations: 迭代次数<br />
--save\_step：迭代保存<br />
## 半监督学习 ##
示例脚本：mpi\_estc.sh<br />
mpiexec -n 8 $ldapath/mpi\_estc\_lda --num\_topics 104 --alpha 0.1 --beta 0.01 --rule\_file $testpath/rule.txt --training\_data\_file $testpath/test.dat --new\_model\_file $testpath/new\_lda\_model_--topic\_distribution\_file $testpath/new\_topic_ --topic\_assignments\_file $testpath/new\_assignments_--burn\_in\_iterations 25 --total\_iterations 2000 --file\_type 0 --save\_step 500 --new\_word\_file $testpath/newwords.txt --model\_file $testpath/lda\_model\_0-final.txt_<br />
变量<br />
$ldapath: mpi\_estc\_lda程序存放的路径<br />
$testpath: 工作目录，训练数据，规则文件和输出结果都放在这个目录。<br />
输入文件：<br />
rule.txt: TOPIC规则文件(见上)<br />
training.data: 训练数据<br />
newwords.txt: 新词文件<br />
lda\_model\_0-00500.txt： 原LDA模型<br />
输出文件：<br />
--topic\_distribution\_file：文档上各个topic的分布<br />
--topic\_assignments\_file：文档中各个term被分配的topic<br />
--model\_file：模型文件,term上各个topic的分布<br />
参数：<br />
--alpha：LDA模型的超参数<br />
--beta：LDA模型的超参数<br />
--num\_topics: 注意，这个topic数量是拆分后topic的总数量<br />
--burn\_in\_iterations：价值原始模型迭代的次数<br />
--total\_iterations: 半监督的迭代次数<br />
--save\_step：迭代保存<br />
## hadoop上运行topic模型进行inference ##
示例脚本：hadoop\_infer\_test.sh<br />
dumbo start lda\_hadoop.py -hadoop /usr/lib/hadoop -input hadooptest.txt -output ./testout -file ${ldamodelfile} -cmdenv "LDAMODELFILE=${ldamodelfile}"<br />
$ldamodelfile为模型文件<br />
可以根据需要加入beta，alpha等参数详见lda\_hadoop.py<br />

# 以上程序已全部在20台Linux服务器，160个CPU上测试通过 #
测试环境为<br />
Python 2.6<br />
dumbo 0.21.31<br />
Hadoop 0.20.2<br />

联系方式：cyzhang9@mail.ustc.edu.cn
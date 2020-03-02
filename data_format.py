# _*_ coding:utf-8 _*_

import numpy as np
import os
import jieba
from sklearn.model_selection import train_test_split
import tensorflow as tf

class DATA():
    def __init__(self,START_TOKEN = 0,END_TOKEN = 1,
                 data_path='./chinese',top_k=5000,read_order='./order.txt'):
        self.START_TOKEN=START_TOKEN    #开始标志
        self.END_TOKEN=END_TOKEN        #结束标志
        self.path=data_path
        self.top_k=top_k                #字典最大长度,选出最常用的
        self.read_order=read_order
        #获取所有文本
        self.alltext=self.get_text()
        #字典
        self.word_index,self.index_word,tokenizer=self.get_dictionary()
        #数据集和划分
        self.in_max_length,self.out_max_length, self.input_vector_train, \
        self.input_vector_val, self.output_vector_train, self.output_vector_val=self.getAndSplitDataset(tokenizer)
    #读取问答文件,进行分词后放入列表返回
    def get_text(self):
        with open(self.read_order, 'r+', encoding='utf-8')as file:
            content=file.readlines()
        content=[i[:-1] for i in content]
        alltext = []
        #for file in os.listdir(self.path):
        for file in content:
            #print(os.path.join(self.path,file))
            with open(file, 'r', encoding='utf-8') as f:
                # 按行读取，变为列表
                strtext = f.read().split('\n')
                # 过滤空行
                strtext = list(filter(lambda x: len(x) > 0, strtext))
                #从第四(index==3)行开始删去断横线和空格 并用jieba工具进行分词处理
                strtext = list(map(lambda x: " ".join(jieba.cut(x.replace('-', '').replace(' ', ''))), strtext[3:]))
                #分词部分结果
                #print(file, strtext[:2])
                #列表相接 列表长度必为双数(问答形式数据)
                alltext = alltext + strtext
                assert len(alltext)%2==0
                #print(len(alltext))

        return alltext

    def get_dictionary(self):
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.top_k, oov_token="<unk>")
        tokenizer.fit_on_texts(self.alltext)

        # 构造字典 选出最常用的top_k个词
        tokenizer.word_index = {key: value for key, value in tokenizer.word_index.items() if value <= self.top_k}
        tokenizer.word_index[tokenizer.oov_token] = self.top_k + 1
        tokenizer.word_index['<start>'] = self.START_TOKEN
        tokenizer.word_index['<end>'] = self.END_TOKEN
        #print(tokenizer.word_index)

        # 反向字典
        index_word = {value: key for key, value in tokenizer.word_index.items()}
        #print(len(index_word))
        return tokenizer.word_index,index_word,tokenizer

    def getAndSplitDataset(self,tokenizer):
        # 变为向量
        train_seqs = tokenizer.texts_to_sequences(self.alltext)
        # print(train_seqs[0:5])
        # print(len(train_seqs))
        # 裁分成问题与答案  train_seqs[0::2]==[train_seqs[i] if i%2==0  for i in range(len(train_seqs))]
        inputseq, outseq = train_seqs[0::2], train_seqs[1::2]
        #print(len(inputseq))
        #print(len(outseq))
        # 按照最长的句子对齐。不足的在其后面补0
        input_vector = tf.keras.preprocessing.sequence.pad_sequences(inputseq, padding='post', value=self.END_TOKEN)
        output_vector = tf.keras.preprocessing.sequence.pad_sequences(outseq, padding='post', value=self.END_TOKEN)
        #补齐后的shape==>(433,22)
        print('input_vector:',input_vector.shape)
        print('output_vecotr:',output_vector.shape)
        start = np.zeros_like(input_vector[:, 0])
        start = np.reshape(start, [-1, 1])
        end = np.ones_like(input_vector[:, 0])
        end = np.reshape(end, [-1, 1])
        #添加句子结束标识==>shape==>(433,23)
        input_vector = np.concatenate((input_vector, end), axis=1)
        output_vector = np.concatenate((output_vector, end), axis=1)
        print("in最大长度", len(input_vector[0]))
        print("out最大长度", len(output_vector[0]))
        in_max_length = len(input_vector[0])
        out_max_length = len(output_vector[0])

        input_vector_train, input_vector_val, output_vector_train, output_vector_val = train_test_split(input_vector,
                                                                                                        output_vector,
                                                                                                           test_size=0.2,
                                                                                                        random_state=1024)
        print('input_vector_train:',input_vector_train.shape,'output_vector_train:',output_vector_train.shape)
        print('input_vector_val:',input_vector_val.shape,'output_vector_val:',output_vector_val.shape)
        return in_max_length,out_max_length,input_vector_train,\
               input_vector_val,output_vector_train,output_vector_val


    # def get_privateValue(self):
    #     return


if __name__=='__main__':
    dt=DATA()
    dt.get_dictionary()
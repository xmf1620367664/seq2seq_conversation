# _*_ coding:utf-8 _*_
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'  #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
from model import MODEL
import numpy as np
import os
import jieba


class TRAIN(MODEL):
    def __init__(self,batch_size=10,checkpoint_dir='./log/000',training_epochs=1000):
        super(TRAIN,self).__init__()

        self.batch_size=batch_size
        self.training_epochs=training_epochs
        self.params={
            'vocab_size': len(self.index_word),
            'batch_size': self.batch_size,
            'output_max_length': self.out_max_length,
            'embed_dim': 100,
            'num_units': 256
        }
        self.checkpoint_dir=checkpoint_dir
        #self.mode=mode
        self.features=tf.placeholder(dtype=tf.int32,shape=[self.batch_size,self.in_max_length],name='input_features')
        self.labels=tf.placeholder(dtype=tf.int32,shape=[self.batch_size,self.out_max_length],name='output_labels')
        # 迭代次数，不可训练
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op,self.pred_outputs,self.loss=self.seq2seq(self.features,self.labels,self.params)
        self.sess=tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True))
        #print(self.input_vector_val[:5])
    def train(self,print_num=1,save_num=10):
        init = tf.global_variables_initializer()
        batch_num = int(self.input_vector_train.shape[0] / self.batch_size)
        test_batch_num = self.input_vector_val.shape[0] // self.batch_size
        tf.summary.scalar('loss_function',self.loss)
        # 定义saver
        saver = tf.train.Saver(max_to_keep=20)

        self.sess.run(init)
        tf.summary.FileWriter(self.checkpoint_dir,self.sess.graph)
        if True:
            # 加载模型继续训练
            ckpt = tf.train.latest_checkpoint(self.checkpoint_dir)
            if ckpt:
                print("load model …………")
                saver.restore(self.sess, ckpt)
            else:
                pass
            for epoch in range(self.sess.run(self.global_step), self.training_epochs):
                # 定义平均损失值
                average_loss = 0
                # 定义随机索引
                index = np.random.permutation(self.input_vector_train.shape[0])
                # 迭代batch_size
                for i in range(batch_num):
                    # 当前batch_size索引
                    train_index = index[i * self.batch_size:(i + 1) * self.batch_size]
                    # 获取batch数据
                    batch_x, batch_y = self.input_vector_train[train_index], self.output_vector_train[train_index]
                    # 训练模型
                    self.sess.run(self.train_op, feed_dict={self.features: batch_x, self.labels: batch_y})
                    # 计算平均损失
                    average_loss += self.sess.run(self.loss, feed_dict={self.features: batch_x, self.labels: batch_y}) / batch_num
                    # 打印信息
                if (epoch + 1) % print_num == 0:
                    print("训练集：\n epoch:{},loss:{}".format(epoch + 1, average_loss))
                    test_loss = 0
                    predictions=[]
                    for test_epoch in range(test_batch_num):
                        batch_x, batch_y = self.input_vector_val[
                                           test_epoch * self.batch_size:(test_epoch + 1) * self.batch_size], \
                                           self.output_vector_val[
                                           test_epoch * self.batch_size:(test_epoch + 1) * self.batch_size]
                        test_loss_ ,pred_outputs_= self.sess.run([self.loss,self.pred_outputs], feed_dict={self.features: batch_x, self.labels: batch_y})
                        test_loss += test_loss_ / test_batch_num
                        for i in range(len(pred_outputs_)):
                            predictions.append(pred_outputs_[i])

                    print("测试集：\n loss:{}".format(test_loss))
                    index=np.random.randint(0,len(predictions))
                    show_label=''
                    for i in self.output_vector_val[index]:
                        if i!=self.END_TOKEN:
                            show_label+=self.index_word[i]+' '
                        else:
                            break
                    show_predict=''
                    for i in predictions[index]:
                        if i!=self.END_TOKEN:
                            show_predict+=self.index_word[i]+' '
                        else:
                            break
                    print('True Label:',show_label,'\n',
                          'Predict Value:',show_predict)
                # global_step累加
                self.sess.run(tf.assign(self.global_step, epoch + 1))
                # 保存模型
                if epoch % save_num == 0:
                    print('save model …………')
                    saver.save(self.sess, os.path.join(self.checkpoint_dir, 'model.ckpt'), global_step=self.global_step)
            saver.save(self.sess, os.path.join(self.checkpoint_dir, 'model.ckpt'), global_step=self.global_step)
            print("训练完成")

    def predict(self,input_data):
        init = tf.global_variables_initializer()
        # 定义saver
        saver = tf.train.Saver(max_to_keep=20)
        self.sess.run(init)
        if True:
            # 加载模型继续训练
            ckpt = tf.train.latest_checkpoint(self.checkpoint_dir)
            if ckpt:
                print("load model …………")
                saver.restore(self.sess, ckpt)
            else:
                pass

        batch_x=np.concatenate([input_data,self.input_vector_val[0: self.batch_size-1]],axis=0)
        #print(batch_x.shape)
        #batch_x[0]=input_data
        predictions = self.sess.run([self.pred_outputs],
                                                  feed_dict={self.features: batch_x})[0]
        index = 0
        show_feature = ''
        #print(batch_x)
        for i in batch_x[0]:
            if i != self.END_TOKEN:
                show_feature += self.index_word[i] +''
            else:
                break
        show_predict = ''
        #print(predictions[index])
        for i in predictions[index]:
            if i != self.END_TOKEN:
                show_predict += self.index_word[i] + ''
            else:
                break
        print('Question Value:', show_feature, '\n',
              'Predict Value:', show_predict)
        return show_predict



if __name__=='__main__':
    tr=TRAIN()
    #tr.train()
    #test=np.ones([1,23])
    test='python好用吗'
    test_list=jieba.lcut(test)
    test_list=[tr.word_index[i] for i in test_list]
    print('test',test_list)
    input_list=np.ones([1,23])
    for i in range(len(test_list)):
        input_list[0][i]=test_list[i]
    print('input_list',input_list)
    tr.predict(input_list)
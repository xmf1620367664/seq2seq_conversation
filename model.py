# _*_ coding:utf-8 _*_
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'  #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
#=================gpu动态占用================
from tensorflow.contrib import layers
from data_format import DATA

#继承自DATA类
class MODEL(DATA):
    def __init__(self,useScheduled=True):
        super(MODEL,self).__init__()
        # 设置训练过程中的采样方式
        self.useScheduled=useScheduled

    def seq2seq(self, features, labels, params):
        vocab_size = params['vocab_size']
        embed_dim = params['embed_dim']
        num_units = params['num_units']
        output_max_length = params['output_max_length']

        print("获得输入张量的名字",features.name,labels.name)
        #inp = tf.identity(features[0], 'input_0')
        #output = tf.identity(labels[0], 'output_0')
        #print(inp.name,output.name)#用于钩子函数显示

        batch_size = tf.shape(features)[0]
        start_tokens = tf.tile([self.START_TOKEN], [batch_size])#也可以使用tf.zeros([batch_size], dtype=tf.int32)
        train_output = tf.concat([tf.expand_dims(start_tokens, 1), labels], 1)#为其添加开始标志

        input_lengths = tf.reduce_sum(tf.cast(tf.not_equal(features, self.END_TOKEN),tf.int32), 1,name="len")
        output_lengths = tf.reduce_sum(tf.cast(tf.not_equal(train_output, self.END_TOKEN),tf.int32), 1,name="outlen")

        input_embed = layers.embed_sequence( features, vocab_size=vocab_size, embed_dim=embed_dim, scope='embed')
        output_embed = layers.embed_sequence( train_output, vocab_size=vocab_size, embed_dim=embed_dim, scope='embed', reuse=True)

        with tf.variable_scope('embed', reuse=True):
            embeddings = tf.get_variable('embeddings')
        Indcell = tf.nn.rnn_cell.DeviceWrapper(tf.contrib.rnn.IndRNNCell(num_units=num_units), "/device:GPU:0")
        IndyLSTM_cell = tf.nn.rnn_cell.DeviceWrapper(tf.contrib.rnn.IndyLSTMCell(num_units=num_units), "/device:GPU:0")
        multi_cell = tf.nn.rnn_cell.MultiRNNCell([Indcell, IndyLSTM_cell])
        encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(multi_cell, input_embed, sequence_length=input_lengths,
                                                                 dtype=tf.float32)

        if self.useScheduled:
            train_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(output_embed,
                                                                               tf.tile([output_max_length],
                                                                                       [batch_size]), embeddings, 0.3)
        else:
            train_helper = tf.contrib.seq2seq.TrainingHelper(output_embed, tf.tile([output_max_length], [batch_size]))

        pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embeddings, start_tokens=tf.tile([self.START_TOKEN], [batch_size]), end_token=self.END_TOKEN)

        def decode(helper, scope, reuse=None):
            with tf.variable_scope(scope, reuse=reuse):
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(#注意力模型
                    num_units=num_units, memory=encoder_outputs,memory_sequence_length=input_lengths)

                cell = tf.contrib.rnn.IndRNNCell(num_units=num_units)
                if reuse == None:
                    keep_prob=0.8
                else:
                    keep_prob=1
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)


                attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                    cell, attention_mechanism, attention_layer_size=num_units / 2)

                out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                    attn_cell, vocab_size, reuse=reuse
                )
                decoder = tf.contrib.seq2seq.BasicDecoder(cell=out_cell, helper=helper,
                    initial_state=out_cell.zero_state( dtype=tf.float32, batch_size=batch_size))

                outputs = tf.contrib.seq2seq.dynamic_decode(
                    decoder=decoder, output_time_major=False,
                    impute_finished=True, maximum_iterations=output_max_length
                )
                return outputs[0]

        train_outputs = decode(train_helper, 'decode')
        pred_outputs = decode(pred_helper, 'decode', reuse=True)

        #tf.identity(train_outputs.sample_id[0], name='train_pred')

        # weights = tf.cast(tf.not_equal(train_output[:, :-1], 0),tf.float32)#掩码
        masks = tf.sequence_mask(output_lengths, output_max_length, dtype=tf.float32, name="masks")

        loss = tf.contrib.seq2seq.sequence_loss(
            train_outputs.rnn_output, labels, weights=masks)

        train_op = layers.optimize_loss(loss, tf.train.get_global_step(), optimizer=params.get('optimizer', 'Adam'),
                                        learning_rate=params.get('learning_rate', 0.001),
                                        summaries=['loss', 'learning_rate'])

        #tf.identity(pred_outputs.sample_id[0], name='predictions')  # 用于钩子函数显示

        return  train_op,pred_outputs.sample_id,loss
        #return     tf.estimator.EstimatorSpec(mode=mode, predictions=pred_outputs.sample_id, loss=loss, train_op=train_op)
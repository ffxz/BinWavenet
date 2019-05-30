# -*- coding:utf-8 -*-
# A Wavenet For Speech Denoising - Dario Rethage - 19.05.2017
# Models.py
#放弃直接使用keras API
#import keras

#使用tf.keras的API进行开发
import tensorflow as tf
from tensorflow.keras import layers

import util
import os
import numpy as np
import BinLayers
import logging

#Speech Denoising Wavenet Model

class DenoisingWavenet():

    def __init__(self, config, load_checkpoint=None, input_length=None, target_field_length=None, print_model_summary=False):

        self.config = config
        self.verbosity = config['training']['verbosity']
        #堆叠三层(num_stacks==3)
        self.num_stacks = self.config['model']['num_stacks']
        if type(self.config['model']['dilations']) is int:
            #1,2,4,...,512重复三次
            self.dilations = [2 ** i for i in range(0, self.config['model']['dilations'] + 1)]
        elif type(self.config['model']['dilations']) is list:
            self.dilations = self.config['model']['dilations']
        #类别是29，应该是说话人鉴别，（一共28个说话人，加上一个unknown）
        #self.num_condition_classes = config['dataset']['num_condition_classes']
        #使用binary编码方式，这个长度计算出来是5，(因为29类用binary的编码方式5位就够了)        
        #self.condition_input_length = self.get_condition_input_length(self.config['model']['condition_encoding'])
        #计算出来就是论文第四页的 6139 samples（对应一个output samples的输入感受野大小）
        self.receptive_field_length = util.compute_receptive_field_length(config['model']['num_stacks'], self.dilations,
                                                                          config['model']['filters']['lengths']['res'],
                                                                          1)
        
        #这里的target field的大小是直接给出的，就是一次预测出多个samples（json中给的是1601）
        if input_length is not None:
            self.input_length = input_length
            self.target_field_length = self.input_length - (self.receptive_field_length - 1)
        if target_field_length is not None:
            self.target_field_length = target_field_length
            self.input_length = self.receptive_field_length + (self.target_field_length - 1)
        #这里的代码是进入else选项 target_field_length == 1601
        #receptive_field_length计算为6139(和论文中一致)
        # ==>input_length == 6139 + 1601 - 1
        #训练的时候使用的是最下面的这种模型
        else:
            self.target_field_length = config['model']['target_field_length']
            self.input_length = self.receptive_field_length + (self.target_field_length - 1)

        #padding是在数据的周围补零吧
        #padding的目的是让输入数据和输出数据的长度是相同的
        self.target_padding = config['model']['target_padding']
        self.padded_target_field_length = self.target_field_length + 2 * self.target_padding
        #python3中是//表示变为整数
        self.half_target_field_length = self.target_field_length // 2
        self.half_receptive_field_length = self.receptive_field_length // 2
        #residual block的数量
        self.num_residual_blocks = len(self.dilations) * self.num_stacks
        self.activation = layers.Activation('relu')
        #加了padding的target范围
        self.samples_of_interest_indices = self.get_padded_target_field_indices()
        #不加padding的target范围
        self.target_sample_indices = self.get_target_field_indices()

        self.optimizer = self.get_optimizer()
        self.out_1_loss = self.get_out_1_loss()
        self.out_2_loss = self.get_out_2_loss()
        self.metrics = self.get_metrics()
        self.epoch_num = 0
        self.checkpoints_path = ''
        self.samples_path = ''
        self.history_filename = ''

        #后面这些参数在完成训练之后，写入到json文件中
        self.config['model']['num_residual_blocks'] = self.num_residual_blocks
        self.config['model']['receptive_field_length'] = self.receptive_field_length
        self.config['model']['input_length'] = self.input_length
        self.config['model']['target_field_length'] = self.target_field_length

        self.model = self.setup_model(load_checkpoint, print_model_summary)

    def setup_model(self, load_checkpoint=None, print_model_summary=False):

        self.checkpoints_path = os.path.join(self.config['training']['path'], 'checkpoints')
        self.samples_path = os.path.join(self.config['training']['path'], 'samples')
        self.history_filename = 'history_' + self.config['training']['path'][
                                             self.config['training']['path'].rindex('/') + 1:] + '.csv'

        model = self.build_model()

        if os.path.exists(self.checkpoints_path) and util.dir_contains_files(self.checkpoints_path):

            if load_checkpoint is not None:
                last_checkpoint_path = load_checkpoint
                self.epoch_num = 0
            else:
                checkpoints = os.listdir(self.checkpoints_path)
                checkpoints.sort(key=lambda x: os.stat(os.path.join(self.checkpoints_path, x)).st_mtime)
                last_checkpoint = checkpoints[-1]
                last_checkpoint_path = os.path.join(self.checkpoints_path, last_checkpoint)
                #last_checkpoint_path = self.checkpoints_path
                self.epoch_num = int(last_checkpoint[11:16])
            print('Loading model from epoch: %d' % self.epoch_num)
            model.load_weights(last_checkpoint_path)

        else:
            print('Building new model...')

            if not os.path.exists(self.config['training']['path']):
                os.mkdir(self.config['training']['path'])

            if not os.path.exists(self.checkpoints_path):
                os.mkdir(self.checkpoints_path)

            self.epoch_num = 0

        if not os.path.exists(self.samples_path):
            os.mkdir(self.samples_path)

        if print_model_summary:
            model.summary()
        #前面都是模型保存或者加载的
        #这个是keras训练时用到的
        model.compile(optimizer=self.optimizer,
                      loss={'data_output_1': self.out_1_loss, 'data_output_2': self.out_2_loss}, metrics=self.metrics)
        self.config['model']['num_params'] = model.count_params()

        config_path = os.path.join(self.config['training']['path'], 'config.json')
        if not os.path.exists(config_path):
            util.pretty_json_dump(self.config, config_path)

        if print_model_summary:
            util.pretty_json_dump(self.config)
        return model

    def get_optimizer(self):

        return tf.keras.optimizers.Adam(lr=self.config['optimizer']['lr'], decay=self.config['optimizer']['decay'],
                                     epsilon=self.config['optimizer']['epsilon'])

    #在它的配置文件当中两个loss都设置为了l1 loss，看看是否需要自己去更改                                 
    def get_out_1_loss(self):

        if self.config['training']['loss']['out_1']['weight'] == 0:
            return lambda y_true, y_pred: y_true * 0

        return lambda y_true, y_pred: self.config['training']['loss']['out_1']['weight'] * util.l1_l2_loss(
            y_true, y_pred, self.config['training']['loss']['out_1']['l1'],
            self.config['training']['loss']['out_1']['l2'])

    def get_out_2_loss(self):

        if self.config['training']['loss']['out_2']['weight'] == 0:
            return lambda y_true, y_pred: y_true * 0

        return lambda y_true, y_pred: self.config['training']['loss']['out_2']['weight'] * util.l1_l2_loss(
            y_true, y_pred, self.config['training']['loss']['out_2']['l1'],
            self.config['training']['loss']['out_2']['l2'])

    def get_callbacks(self):

        return [
            tf.keras.callbacks.ReduceLROnPlateau(patience=self.config['training']['early_stopping_patience'] / 2,
                                              cooldown=self.config['training']['early_stopping_patience'] / 4,
                                              verbose=1),
            tf.keras.callbacks.EarlyStopping(patience=self.config['training']['early_stopping_patience'], verbose=1,
                                          monitor='loss'),
            tf.keras.callbacks.ModelCheckpoint(os.path.join(self.checkpoints_path, 'checkpoint.{epoch:05d}-{val_loss:.3f}.h5')),
            tf.keras.callbacks.CSVLogger(os.path.join(self.config['training']['path'], self.history_filename), append=True),
            tf.keras.callbacks.TensorBoard(log_dir=os.path.join(self.config['training']['path'], 'tensorboard'))
        ]

    def fit_model(self, train_set_generator, num_train_samples, test_set_generator, num_test_samples, num_epochs):

        print('Fitting model with %d training samples and %d test samples...' % (num_train_samples, num_test_samples))

        self.model.fit_generator(train_set_generator,
                                 num_train_samples,
                                 epochs=num_epochs,
                                 validation_data=test_set_generator,
                                 validation_steps=num_test_samples,
                                 callbacks=self.get_callbacks(),
                                 verbose=self.verbosity,
                                 initial_epoch=self.epoch_num)

    def denoise_batch(self, inputs):
        return self.model.predict_on_batch(inputs)

    def get_target_field_indices(self):

        target_sample_index = self.get_target_sample_index()

        return range(target_sample_index - self.half_target_field_length,
                     target_sample_index + self.half_target_field_length + 1)
    #这里的最大值超出了范围
    def get_padded_target_field_indices(self):

        target_sample_index = self.get_target_sample_index()
        #见论文 fig4 ，非常形象的对应图的输入输出位置
        return range(target_sample_index - self.half_target_field_length - self.target_padding,
                     target_sample_index + self.half_target_field_length + self.target_padding + 1)

    def get_target_sample_index(self):
        return int(np.floor(self.input_length / 2.0))

    def get_metrics(self):

        return [
            tf.keras.metrics.mean_absolute_error,
            self.valid_mean_absolute_error
        ]

    def valid_mean_absolute_error(self, y_true, y_pred):
        return tf.keras.backend.mean(
            tf.keras.backend.abs(y_true[:, 1:-2] - y_pred[:, 1:-2]))

    '''def get_condition_input_length(self, representation):

        if representation == 'binary':
            return int(np.max((np.ceil(np.log2(self.num_condition_classes)), 1)))
        else:
            return self.num_condition_classes'''

    def build_model(self):
        #here, it transfer to tensorflow form
        data_input = layers.Input(
                shape=(self.input_length,),
                name='data_input')

        #规定了输入condition数据的大小 (condition_input_length==5)
        #condition_input = layers.Input(shape=(self.condition_input_length,),
        #                                     name='condition_input')
        
        #这条语句应该就是维度扩展，但是我不太懂继承keras.layers的自建类，使用时直接调用哪个方法
        data_expanded = BinLayers.AddSingletonDepth()(data_input)
        
        #Ellipsis在python是一个常量，第一次遇到，哈哈
        #这里的指令的作用就是把输入的长度变成padded_target_field_length
        #这个到后面相减的时候才会再用到
        #我觉得如果想要直接使用tf.slice，也要封装在一个类之中
        data_input_target_field_length = BinLayers.Slice(
            (slice(self.samples_of_interest_indices[0], self.samples_of_interest_indices[-1] + 1, 1), Ellipsis),
            (self.padded_target_field_length,1),
            name='data_input_target_field_length')(data_expanded)
        #滤波器的长度是3，卷积核的输出数量为128
        data_out = layers.Convolution1D(self.config['model']['filters']['depths']['res'],
                                              self.config['model']['filters']['lengths']['res'], padding='same',
                                              use_bias=False,
                                              name='initial_causal_conv')(data_expanded)
        
        #condition_input 本来是用5位来表示的，然后为了把它和input融合也需要进行维度转换=>128
        #condition_out = layers.Dense(self.config['model']['filters']['depths']['res'],
        #                                   name='initial_dense_condition',
        #                                   use_bias=False)(condition_input)
        #在时长上面进行repeat使得与input的长度是相同的
        #condition_out = layers.RepeatVector(self.input_length,
        #                                          name='initial_condition_repeat')(condition_out)
        #下面这句话打印出来其实就是add这个node
        #data_out = layers.Merge(mode='sum', name='initial_data_condition_merge')(
        #    [data_out, condition_out])
        #data_out = layers.Add()([data_out, condition_out])

        skip_connections = []
        res_block_i = 0
        #外部的三个block
        #如果内存溢出，把stack或者是dilations改的小一点可能会有帮助
        #但是需要把输入和输出都对应的进行调整
        for stack_i in range(self.num_stacks):
            layer_in_stack = 0
            #内部的1,2,4,,,512
            for dilation in self.dilations:
                res_block_i += 1
                data_out, skip_out = self.dilated_residual_block(data_out, res_block_i, layer_in_stack, dilation, stack_i)#, condition_input
                if skip_out is not None:
                    skip_connections.append(skip_out)
                layer_in_stack += 1

        #data_out = layers.Merge(mode='sum')(skip_connections)
        data_out = layers.Add()(skip_connections)
        data_out = self.activation(data_out)
        #filter 3, output 2048
        data_out = layers.Convolution1D(self.config['model']['filters']['depths']['final'][0],
                                              self.config['model']['filters']['lengths']['final'][0],
                                              padding='same',
                                              use_bias=False)(data_out)
        #condition_input也需要转成2048维的
        #condition_out = layers.Dense(self.config['model']['filters']['depths']['final'][0],
        #                                   use_bias=False,
        #                                   name='penultimate_conv_1d_condition')(condition_input)
        
        #长度扩展成 padded_target_field_length
        #condition_out = layers.RepeatVector(self.padded_target_field_length,
        #                                          name='penultimate_conv_1d_condition_repeat')(condition_out)

        #data_out = layers.Merge(mode='sum', name='penultimate_conv_1d_condition_merge')([data_out, condition_out])
        #data_out = layers.Add()([data_out, condition_out])

        data_out = self.activation(data_out)
        # filter 3, output 256
        data_out = layers.Convolution1D(self.config['model']['filters']['depths']['final'][1],
                                              self.config['model']['filters']['lengths']['final'][1], padding='same',
                                              use_bias=False)(data_out)

        #condition_out = layers.Dense(self.config['model']['filters']['depths']['final'][1], use_bias=False,
        #                                   name='final_conv_1d_condition')(condition_input)

        #condition_out = layers.RepeatVector(self.padded_target_field_length,
        #                                          name='final_conv_1d_condition_repeat')(condition_out)

        #data_out = layers.Merge(mode='sum', name='final_conv_1d_condition_merge')([data_out, condition_out])
        #data_out = layers.Add()([data_out, condition_out])
        #维度变成1为(音频)
        data_out = layers.Convolution1D(1, 1)(data_out)

        data_out_speech = data_out
        data_out_noise = BinLayers.Subtract(name='subtract_layer')([data_input_target_field_length, data_out_speech])
        
        #改变一下形式
        data_out_speech = layers.Lambda(lambda x: tf.keras.backend.squeeze(x, 2),
                                              output_shape=lambda shape: (shape[0], shape[1]), name='data_output_1')(
            data_out_speech)

        data_out_noise = layers.Lambda(lambda x: tf.keras.backend.squeeze(x, 2),
                                              output_shape=lambda shape: (shape[0], shape[1]), name='data_output_2')(
            data_out_noise)

        return tf.keras.models.Model(inputs=[data_input], outputs=[data_out_speech, data_out_noise])#inputs=[data_input, condition_input]


    #def dilated_residual_block(self, data_x, condition_x, res_block_i, layer_i, dilation, stack_i):
    def dilated_residual_block(self, data_x, res_block_i, layer_i, dilation, stack_i):

        original_x = data_x

        # Data sub-block, 空洞卷积的方法
        '''data_out = layers.AtrousConvolution1D(2 * self.config['model']['filters']['depths']['res'],
                                                    self.config['model']['filters']['lengths']['res'],
                                                    atrous_rate=dilation, border_mode='same',
                                                    bias=False,
                                                    name='res_%d_dilated_conv_d%d_s%d' % (
                                                    res_block_i, dilation, stack_i),
                                                    activation=None)(data_x)'''
        #这个方法的前两个参数还没弄清他们的位置关系，如果训练报错可以查看一下这个隐患
        data_out = layers.Convolution1D(2 * self.config['model']['filters']['depths']['res'],
                                                    self.config['model']['filters']['lengths']['res'],
                                                    dilation_rate=dilation, padding='same',
                                                    use_bias=False)(data_x)

        data_out_1 = BinLayers.Slice(
            (Ellipsis, slice(0, self.config['model']['filters']['depths']['res'])),
            (self.input_length, self.config['model']['filters']['depths']['res']),
            name='res_%d_data_slice_1_d%d_s%d' % (self.num_residual_blocks, dilation, stack_i))(data_out)

        data_out_2 = BinLayers.Slice(
            (Ellipsis, slice(self.config['model']['filters']['depths']['res'],
                             2 * self.config['model']['filters']['depths']['res'])),
            (self.input_length, self.config['model']['filters']['depths']['res']),
            name='res_%d_data_slice_2_d%d_s%d' % (self.num_residual_blocks, dilation, stack_i))(data_out)

        # Condition sub-block
        #condition_out = tf.keras.layers.Dense(2 * self.config['model']['filters']['depths']['res'],
        #                                   name='res_%d_dense_condition_%d_s%d' % (res_block_i, layer_i, stack_i),
        #                                   use_bias=False)(condition_x)

        #condition_out = tf.keras.layers.Reshape((self.config['model']['filters']['depths']['res'], 2),
        #                                     name='res_%d_condition_reshape_d%d_s%d' % (
        #                                         res_block_i, dilation, stack_i))(condition_out)

        #condition_out_1 = BinLayers.Slice((Ellipsis, 0), (self.config['model']['filters']['depths']['res'],),
        #                                      name='res_%d_condition_slice_1_d%d_s%d' % (
        #                                          res_block_i, dilation, stack_i))(condition_out)

        #condition_out_2 = BinLayers.Slice((Ellipsis, 1), (self.config['model']['filters']['depths']['res'],),
        #                                      name='res_%d_condition_slice_2_d%d_s%d' % (
        #                                          res_block_i, dilation, stack_i))(condition_out)

        #condition_out_1 = tf.keras.layers.RepeatVector(self.input_length, name='res_%d_condition_repeat_1_d%d_s%d' % (
        #                                                res_block_i, dilation, stack_i))(condition_out_1)
        #condition_out_2 = tf.keras.layers.RepeatVector(self.input_length, name='res_%d_condition_repeat_2_d%d_s%d' % (
        #                                                res_block_i, dilation, stack_i))(condition_out_2)

        #data_out_1 = tf.keras.layers.Merge(mode='sum', name='res_%d_merge_1_d%d_s%d' %
        #                                                 (res_block_i, dilation, stack_i))([data_out_1, condition_out_1])
        #data_out_2 = tf.keras.layers.Merge(mode='sum', name='res_%d_merge_2_d%d_s%d' % (res_block_i, dilation, stack_i))\
        #    ([data_out_2, condition_out_2])
        
        #data_out_1 = tf.keras.layers.Add()([data_out_1, condition_out_1])
        #data_out_2 = tf.keras.layers.Add()([data_out_2, condition_out_2])

        tanh_out = tf.keras.layers.Activation('tanh')(data_out_1)
        sigm_out = tf.keras.layers.Activation('sigmoid')(data_out_2)

        #这个操作就是相乘
        #data_x = keras.layers.Merge(mode='mul', name='res_%d_gated_activation_%d_s%d' % (res_block_i, layer_i, stack_i))(
        #    [tanh_out, sigm_out])
        data_x = layers.Multiply()([tanh_out, sigm_out])

        data_x = tf.keras.layers.Convolution1D(
            self.config['model']['filters']['depths']['res'] + self.config['model']['filters']['depths']['skip'], 1,
            padding='same', use_bias=False)(data_x)

        res_x = BinLayers.Slice((Ellipsis, slice(0, self.config['model']['filters']['depths']['res'])),
                             (self.input_length, self.config['model']['filters']['depths']['res']),
                             name='res_%d_data_slice_3_d%d_s%d' % (res_block_i, dilation, stack_i))(data_x)

        skip_x = BinLayers.Slice((Ellipsis, slice(self.config['model']['filters']['depths']['res'],
                                               self.config['model']['filters']['depths']['res'] +
                                               self.config['model']['filters']['depths']['skip'])),
                              (self.input_length, self.config['model']['filters']['depths']['skip']),
                              name='res_%d_data_slice_4_d%d_s%d' % (res_block_i, dilation, stack_i))(data_x)

        skip_x = BinLayers.Slice((slice(self.samples_of_interest_indices[0], self.samples_of_interest_indices[-1] + 1, 1),
                               Ellipsis), (self.padded_target_field_length, self.config['model']['filters']['depths']['skip']),
                              name='res_%d_keep_samples_of_interest_d%d_s%d' % (res_block_i, dilation, stack_i))(skip_x)

        #res_x = keras.layers.Merge(mode='sum')([original_x, res_x])
        res_x = layers.Add()([original_x, res_x])

        return res_x, skip_x

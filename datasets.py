# -*- coding:utf-8 -*-
# A Wavenet For Speech Denoising - Dario Rethage - 19.05.2017
# Datasets.py

import util
import os
import numpy as np
import logging

class NSDTSEADataset():

    def __init__(self, config, model):

        self.model = model
        self.path = config['dataset']['path']
        self.sample_rate = config['dataset']['sample_rate']
        self.file_paths = {'train': {'clean': [], 'noisy': []}, 'test': {'clean': [], 'noisy': []}}
        self.sequences = {'train': {'clean': [], 'noisy': []}, 'test': {'clean': [], 'noisy': []}}
        self.voice_indices = {'train': [], 'test': []}
        self.regain_factors = {'train': [], 'test': []}
        self.speakers = {'train': [], 'test': []}
        self.speaker_mapping = {}
        self.batch_size = config['training']['batch_size']
        self.noise_only_percent = config['dataset']['noise_only_percent']
        self.regain = config['dataset']['regain']
        self.extract_voice = config['dataset']['extract_voice']
        self.in_memory_percentage = config['dataset']['in_memory_percentage']
        self.num_sequences_in_memory = 0
        #self.condition_encode_function = util.get_condition_input_encode_func(config['model']['condition_encoding'])

    def load_dataset(self):

        print('Loading NSDTSEA dataset...')

        for set in ['train', 'test']:
            for condition in ['clean', 'noisy']:
                #一共有四个目录，都要把其中的wav遍历
                current_directory = os.path.join(self.path, condition+'_'+set+'set_wav')

                sequences, file_paths, speakers, speech_onset_offset_indices, regain_factors = \
                    self.load_directory(current_directory, condition)
                
                self.file_paths[set][condition] = file_paths
                self.speakers[set] = speakers
                self.sequences[set][condition] = sequences

                if condition == 'clean':
                    self.voice_indices[set] = speech_onset_offset_indices
                    self.regain_factors[set] = regain_factors

        return self

    def load_directory(self, directory_path, condition):

        filenames = [filename for filename in os.listdir(directory_path) if filename.endswith('.wav')]

        speakers = []
        file_paths = []
        speech_onset_offset_indices = []
        regain_factors = []
        sequences = []
        for filename in filenames:

            speaker_name = filename[0:4]
            speakers.append(speaker_name)

            filepath = os.path.join(directory_path, filename)

            if condition == 'clean':

                sequence = util.load_wav(filepath, self.sample_rate)
                sequences.append(sequence)
                self.num_sequences_in_memory += 1
                regain_factors.append(self.regain / util.rms(sequence))
                #如果extract_voice为true，则需要进行去除前后静音操作
                if self.extract_voice:
                    #speech_onset_offset_indices是非静音段的起止点
                    speech_onset_offset_indices.append(util.get_subsequence_with_speech_indices(sequence))
            else:
                if self.in_memory_percentage == 1 or np.random.uniform(0, 1) <= (self.in_memory_percentage-0.5)*2:
                    sequence = util.load_wav(filepath, self.sample_rate)
                    sequences.append(sequence)
                    self.num_sequences_in_memory += 1
                else:
                    sequences.append([-1])

            if speaker_name not in self.speaker_mapping:
                self.speaker_mapping[speaker_name] = len(self.speaker_mapping) + 1

            file_paths.append(filepath)

        return sequences, file_paths, speakers, speech_onset_offset_indices, regain_factors

    def get_num_sequences_in_dataset(self):
        return len(self.sequences['train']['clean']) + len(self.sequences['train']['noisy']) + len(self.sequences['test']['clean']) + len(self.sequences['test']['noisy'])

    def retrieve_sequence(self, set, condition, sequence_num):

        if len(self.sequences[set][condition][sequence_num]) == 1:
            sequence = util.load_wav(self.file_paths[set][condition][sequence_num], self.sample_rate)

            if (float(self.num_sequences_in_memory) / self.get_num_sequences_in_dataset()) < self.in_memory_percentage:
                self.sequences[set][condition][sequence_num] = sequence
                self.num_sequences_in_memory += 1
        else:
            sequence = self.sequences[set][condition][sequence_num]

        return np.array(sequence)

    def get_random_batch_generator(self, set):

        if set not in ['train', 'test']:
            raise ValueError("Argument SET must be either 'train' or 'test'")

        while True:
            #取batch个语音
            sample_indices = np.random.randint(0, len(self.sequences[set]['clean']), self.batch_size)
            condition_inputs = []
            batch_inputs = []
            batch_outputs_1 = []
            batch_outputs_2 = []

            for i, sample_i in enumerate(sample_indices):

                while True:

                    speech = self.retrieve_sequence(set, 'clean', sample_i)
                    noisy = self.retrieve_sequence(set, 'noisy', sample_i)
                    noise = noisy - speech
                    
                    #去除前后的静音段
                    if self.extract_voice:
                        speech = speech[self.voice_indices[set][sample_i][0]:self.voice_indices[set][sample_i][1]]
                    #要乘一个增益因子
                    speech_regained = speech * self.regain_factors[set][sample_i]
                    noise_regained = noise * self.regain_factors[set][sample_i]

                    #这句话的意思是如果随机取出的语音时长太短，则需要换一个（因为对输入的长度有要求）
                    if len(speech_regained) < self.model.input_length:
                        sample_i = np.random.randint(0, len(self.sequences[set]['clean']))
                    else:
                        break
                #输入的语音长度要比定义的模型输入要长，这时候要截取
                #np.squeeze是将list外边的括号去掉，只剩下里面的数值
                offset = np.squeeze(np.random.randint(0, len(speech_regained) - self.model.input_length, 1))
                #这里可以看出训练使用的只是其中的语音片段，不是用整段语音进行训练的
                speech_fragment = speech_regained[offset:offset + self.model.input_length]
                noise_fragment = noise_regained[offset:offset + self.model.input_length]

                #输入数据是含有噪声的语音
                input = noise_fragment + speech_fragment
                output_speech = speech_fragment
                output_noise = noise_fragment

                # **很重要** 这里就是在训练时加入10%的only noise作为训练
                if self.noise_only_percent > 0:
                    if np.random.uniform(0, 1) <= self.noise_only_percent:
                        input = output_noise #Noise only
                        output_speech = np.array([0] * self.model.input_length) #Silence

                batch_inputs.append(input)
                batch_outputs_1.append(output_speech)
                batch_outputs_2.append(output_noise)

                #如果概率小于1/29, 就没有条件输入, 否则加入条件
                if np.random.uniform(0, 1) <= 1.0 / self.get_num_condition_classes():
                    condition_input = 0
                else:
                    condition_input = self.speaker_mapping[self.speakers[set][sample_i]]
                    if condition_input > 28: #If speaker is in test set, use wildcard condition class 0
                        condition_input = 0

                condition_inputs.append(condition_input)

            batch_inputs = np.array(batch_inputs, dtype='float32')
            batch_outputs_1 = np.array(batch_outputs_1, dtype='float32')
            batch_outputs_2 = np.array(batch_outputs_2, dtype='float32')
            #这里是输入输出的对应关系，目前还没弄懂？？？
            #输出数据的范围，因为需要标注数据，它的位置需要是对应感受野的那块区域
            batch_outputs_1 = batch_outputs_1[:, self.model.get_padded_target_field_indices()]
            batch_outputs_2 = batch_outputs_2[:, self.model.get_padded_target_field_indices()]
            #要进行 binary 编码了
            #condition_inputs = self.condition_encode_function(np.array(condition_inputs, dtype='uint8'), self.model.num_condition_classes)

            #batch = {'data_input': batch_inputs, 'condition_input': condition_inputs}, {
            #    'data_output_1': batch_outputs_1, 'data_output_2': batch_outputs_2}

            batch = {'data_input': batch_inputs}, {
                'data_output_1': batch_outputs_1, 'data_output_2': batch_outputs_2}
            yield batch

    def get_condition_input_encode_func(self, representation):

        if representation == 'binary':
            return util.binary_encode
        else:
            return util.one_hot_encode

    def get_num_condition_classes(self):
        return 29

    def get_target_sample_index(self):
        return int(np.floor(self.fragment_length / 2.0))

    def get_samples_of_interest_indices(self, causal=False):

        if causal:
            return -1
        else:
            target_sample_index = self.get_target_sample_index()
            return range(target_sample_index - self.half_target_field_length - self.target_padding,
                         target_sample_index + self.half_target_field_length + self.target_padding + 1)

    def get_sample_weight_vector_length(self):
        if self.samples_of_interest_only:
            return len(self.get_samples_of_interest_indices())
        else:
            return self.fragment_length

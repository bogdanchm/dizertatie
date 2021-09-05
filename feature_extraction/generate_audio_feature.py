#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:18:00 2018
generate opensmile emotion based features on utterance level
@author: ddeng
"""
"""
///////// > openSMILE configuration file for emotion features <      //////////////////
/////////   Based on INTERSPEECH 2010 paralinguistics challenge      //////////////////
/////////   Pitch, Loudness, Jitter, MFCC, MFB, LSP and functionals  //////////////////
/////////                                                            //////////////////
/////////   1582 1st level functionals:                              //////////////////
/////////     (34 LLD + 34 delta) * 21 functionals                   //////////////////
/////////     +(4 LLD + 4 delta) * 19 functionals                    //////////////////
/////////     + 1 x Num. pitch onsets (pseudo syllables)             //////////////////
/////////      + 1 x turn duration in seconds                        //////////////////
"""

import csv
import glob
import os
import os.path
import subprocess
import tqdm
import numpy as np
from tqdm import tqdm
import pickle
import librosa
import librosa.display
from subprocess import call
import pdb
import argparse

main_Dir = '../Audio_Feature'

def padding(array, xx, yy):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """
    h = array.shape[0]
    w = array.shape[1]
    a = max((xx - h) // 2,0)
    aa = max(0,xx - a - h)
    b = max(0,(yy - w) // 2)
    bb = max(yy - b - w,0)
    return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')

def clean_data(data):
    data = data.split(',')[1:] #'noname' deleted
    data[-1] = data[-1][0:-1] #'\n' deleted
    length = len(data)
    new_data = np.zeros(length)
    for i, item in enumerate(data):
        new_data[i] = float(item)
    return new_data[0:1582]

def save_as_pkl(folder, save_file_name, audio_image_dict):
    """
    main_Dic: {State_name:{Video_Name: {Utterance_index:[ feature_vector ]}}}

    """
    main_Dic = {}
    main_Dic = {}
    video_path = glob.glob(os.path.join(main_Dir, folder, '*'))
    for video in tqdm(video_path):
        pass
        video_name = video.split('/')[-1]
        main_Dic[video_name]={}

        utterance_path = glob.glob(os.path.join(video, '*.txt'))
        for utterance in utterance_path:
            utterance_index = utterance.split('/')[-1].split('.')[0].split('_')[-1]
            main_Dic[video_name][utterance_index]={}

            # Uncomment this in order to extract features using opensmile
            # file = open(utterance, 'r')
            # while True:
            #     line = file.readline()
            #     if line:
            #         if line.startswith('@data'):
            #             line = file.readline()
            #             line = file.readline()
            #             data = line
            #             if data: #sometimes , the data might be empty
            #                 data = clean_data(data)
            #                 # main_Dic[video_name][utterance_index]=data
            #             break
            #     else:
            #         break

            # This uses the features extracted with librosa (mfcc, centroids, freq bandiwdth) concatenated as images over 3 channels
            main_Dic[video_name][utterance_index] = audio_image_dict[video_name][utterance_index]

    with open(save_file_name ,'wb') as fout:
        pickle.dump(main_Dic, fout)

def check_already_extracted(feature_dir):
    """Check to see if we created the -0001 frame of this file."""

    return bool(os.path.exists(os.path.join(feature_dir ,'*.txt')))



def generate_aud_rep(folder, file_path, audio_image_dict):
    """
    extract 1582 features using OpenSmile Toolkit, the command line should be:
    ./SMILExtract -C ./config/emobase2010.conf -I /newdisk/test/000046280.wav -O /newdisk/test/000046280.txt
    -C : the configure file's path
    -I: the input .wav file
    -O: the output txt file
    Noting that SMILExtract executable file is in /openSMILE-2.1.0 in the home directory
    """
    opensmile_script_path = '../../../opensmile/build/progsrc/smilextract/SMILExtract'
    # opensmile_conf = '../../../opensmile/config/emobase/emobase2010.conf'
    # opensmile_conf = '../../../opensmile/config/is09-13/IS09_emotion.conf'
    opensmile_conf = '../../../opensmile/config/is09-13/IS13_ComParE_Voc.conf'


    video_folders = glob.glob(os.path.join('../Audio_Frames', folder, '*'))
    print(folder + " videos extraction ...\n")

    max_dim = 0

    for vid in tqdm(video_folders):
        vid_name = vid.split('/')[-1]

        original_uttr_wav = glob.glob(os.path.join('../Audio_Frames', folder, vid_name, '*.wav'))
        for utter_wav in original_uttr_wav:
            samples, sample_rate = librosa.load(utter_wav)
            mel_sgram = mfcc = librosa.feature.mfcc(samples, sr=sample_rate)
            spec_centroid = librosa.feature.spectral_centroid(y=samples, sr=sample_rate)
            chroma_stft = librosa.feature.chroma_stft(y=samples, sr=sample_rate)

            max_dim = max(max(mel_sgram.shape[1], spec_centroid.shape[1], chroma_stft.shape[1]), max_dim)

    for vid in tqdm(video_folders):
        print ("Video Path: "+ vid+'\n')
        Train_Valid_orTest = vid.split('/')[1]
        vid_name = vid.split('/')[-1]


        original_uttr_wav = glob.glob(os.path.join('../Audio_Frames', folder, vid_name, '*.wav'))
        for utter_wav in original_uttr_wav:
            utterance_name = utter_wav.split('/')[-1].split('.')[0]
            utterance_index = utter_wav.split('/')[-1].split('.')[0].split('_')[-1]
            feature_path = os.path.join(main_Dir, folder, vid_name, utterance_name+'.txt')
            if not os.path.exists(main_Dir):
                os.mkdir(main_Dir)
            if not os.path.exists(os.path.join(main_Dir, folder)):
                os.mkdir(os.path.join(main_Dir, folder))
            if not os.path.exists(os.path.join(main_Dir, folder, vid_name)):
                os.mkdir(os.path.join(main_Dir, folder, vid_name))

            create_audio_feature_images(utter_wav, utterance_index, vid_name, audio_image_dict, max_dim)

            if not os.path.exists(feature_path):
                des = feature_path
                cmd = opensmile_script_path + ' -C ' + opensmile_conf + ' -I ' + utter_wav +' -O '+ des
                process = subprocess.Popen(cmd.split(),stdout=subprocess.PIPE, stderr= subprocess.STDOUT)
                out, err = process.communicate()
                print(out)
            else:
                print(vid + " feature has been extracted already.\n")

def create_audio_feature_images(utter_wav, utterance_index, vid_name, audio_image_dict, max_dim):
    samples, sample_rate = librosa.load(utter_wav)
        
    mel_sgram = mfcc = librosa.feature.mfcc(samples, sr=sample_rate)
    spec_centroid = librosa.feature.spectral_centroid(y=samples, sr=sample_rate)
    chroma_stft = librosa.feature.chroma_stft(y=samples, sr=sample_rate)

    mel_sgram = np.reshape(padding(mel_sgram, 20, max_dim), (1, 20, max_dim))
    spec_centroid = np.reshape(padding(spec_centroid, 20, max_dim), (1, 20, max_dim))
    chroma_stft = np.reshape(padding(chroma_stft, 20, max_dim), (1, 20, max_dim))

    if vid_name not in audio_image_dict.keys():
        audio_image_dict[vid_name] = {}
    audio_image_dict[vid_name][utterance_index] = np.concatenate((mel_sgram, spec_centroid, chroma_stft), axis=0)

def join_data():
    train_path = 'Audio_Feature_Train.pkl'
    validation_path = 'Audio_Feature_Validation.pkl'
    output_path = 'Audio_Feature.pkl'
    if os.path.exists(train_path) and os.path.exists(validation_path):
        data = {}
        data['Train'] = pickle.load(open(train_path,'rb'))     
        data['Validation'] = pickle.load(open(validation_path,'rb'))
        pickle.dump(data, open(output_path,'wb'))

def fix_data():
    output_path = 'Audio_Feature.pkl'
    data = {}
    data = pickle.load(open(output_path,'rb'))
    data['Train'] = data['Train']['Train']
    data['Validation'] = data['Validation']['Validation']
    pickle.dump(data, open(output_path,'wb'))

def main(task):
    if task == 'All':
        tasks = ['Train', 'Validation', 'Test']
    elif ',' in task:
        tasks = task.split(',')
    else:
        tasks = [task]
    for task in tasks:
        audio_image_dict = {}
        save_file_name =  'Audio_Feature_' + task + '.pkl'
        generate_aud_rep(task, save_file_name, audio_image_dict)
        # pdb.set_trace()
        save_as_pkl(task, save_file_name, audio_image_dict)
    join_data()
    """
    folders = ['Train', 'Validation']
    save_file_name =  'Audio_Feature_uttr_level.pkl' 
    generate_aud_rep(folders)
    save_as_pkl(folders, save_file_name)
    """
    #folders = ['Train', 'Validation', 'Test']
    #for folder in folders:
    #    save_file_name =  'Audio_Feature_' + folder + '.pkl'
    #    generate_aud_rep(folder, save_file_name)
    # save_as_pkl(folder, save_file_name)
    # join_data()
    #fix_data()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script is used to extract features from audio.')
    parser.add_argument('--task', default='All', help="This value can be Train, Validation, Test, or All.")
    args = parser.parse_args()
    # pdb.set_trace()
    main(args.task)


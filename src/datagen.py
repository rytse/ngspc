import os

import glob
import random

import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display

import augment

id_n = 0
aug_ratio = 3

for filename in glob.iglob('../data/raw/' + '**/*.wav', recursive=True):
    cat = filename.split('/')[3][:-5]
    id_s = "{:04d}".format(id_n)

    try:
        y, sr = librosa.load(filename)
    except Exception as e:
        print(f'Failed to load {filename}')
        print('\n\n~~~~~~~~\n\n')
        print(e)
        print('\n\n~~~~~~~~\n\n')
        print('Continuing\n\n\n')
    
    if not os.path.exists('../data/gen/mel/train/' + cat + '/'):
        os.makedirs('../data/gen/mel/train/' + cat + '/')
    if not os.path.exists('../data/gen/mel/valid/' + cat + '/'):
        os.makedirs('../data/gen/mel/valid/' + cat + '/')
    if not os.path.exists('../data/gen/ftemp/train/' + cat + '/'):
        os.makedirs('../data/gen/ftemp/train/' + cat + '/')
    if not os.path.exists('../data/gen/ftemp/valid/' + cat + '/'):
        os.makedirs('../data/gen/ftemp/valid/' + cat + '/')
        
    # if not os.path.exists('../data/gen/sflat/train/' + cat + '/'):
    #     os.makedirs('../data/gen/sflat/train/' + cat + '/')
    # if not os.path.exists('../data/gen/sflat/valid/' + cat + '/'):
    #     os.makedirs('../data/gen/sflat/valid/' + cat + '/')

    all_sigs = []
    all_sigs.append(y)
    for i in range(aug_ratio - 1):
        all_sigs.append(augment.augment(y, sr))

    try:
        for i in range(len(all_sigs)):
            sig = all_sigs[i]

            tv = 'valid' if random.random() >= 0.9 else 'train'
            i_s = '{:02d}'.format(i)
            ppath = f'/{tv}/{cat}/{cat}_{id_s}_{i_s}.png'
            
            mel_spec = librosa.feature.melspectrogram(sig, sr)
            plt.figure()
            librosa.display.specshow(librosa.power_to_db(mel_spec, ref=np.max))
            plt.savefig('../data/gen/mel' + ppath)
            plt.close()

            f_temp = librosa.feature.fourier_tempogram(sig, sr)
            plt.figure()
            librosa.display.specshow(f_temp, sr=sr)
            plt.savefig('../data/gen/ftemp' + ppath)
            plt.close()

            # s_flat = librosa.feature.spectral_flatness(sig).T
            # plt.figure()
            # librosa.display.specshow(s_flat, sr=sr)
            # plt.savefig('../data/gen/sflat' + ppath)
            # plt.close()

    except Exception as e:
        print(f'Error on {id_s}!')
        print('\n\n~~~~~~~~~~~~~~~~~~~~~~\n\n')
        print(e)
        print('\n\n\n')

    id_n += 1
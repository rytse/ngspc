import glob
import os

import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display

id_n = 0

for filename in glob.iglob('./'+ '**/*.wav', recursive=True):
    cat = filename.split('/')[3][:-5]
    y, sr = librosa.load(filename)

    id_s = "{:04d}".format(id_n)
    if not os.path.exists('./data/mel/' + cat + '/'):
        os.makedirs('./data/mel/' + cat + '/')
    if not os.path.exists('./data/ftemp/' + cat + '/'):
        os.makedirs('./data/ftemp/' + cat + '/')
    if not os.path.exists('./data/sflat/' + cat + '/'):
        os.makedirs('./data/sflat/' + cat + '/')

    try:
        mel_spec = librosa.feature.melspectrogram(y, sr)
        plt.figure()
        librosa.display.specshow(librosa.power_to_db(mel_spec, ref=np.max))
        plt.savefig('./data/mel/' + cat + '/' + cat + '_' + id_s + '.png')
        plt.close()

        f_temp = librosa.feature.fourier_tempogram(y, sr)
        plt.figure()
        librosa.display.specshow(f_temp, sr=sr)
        plt.savefig('./data/ftemp/' + cat + '/' + cat + '_' + id_s + '.png')
        plt.close()

        s_flat = librosa.feature.spectral_flatness(y).T
        plt.figure()
        librosa.display.specshow(s_flat, sr=sr)
        plt.savefig('./data/sflat/' + cat + '/' + cat + '_' + id_s + '.png')
        plt.close()

    except:
        print(f'Error on {id_s}!')

    id_n += 1

import os
import librosa
import scipy
import numpy as np
from tqdm import tqdm

def convert_folder_to_extension(folder_path, extension=  'wav'):
    for root, dirs, files in os.walk(folder_path):
        for name in tqdm(files):
            if ".mp3" in name:
                try:
                    new_name = name[:-3]+extension
                    if not os.path.exists(os.path.join(root,new_name)):
                        y,sr =  librosa.load(os.path.join(root,name))
                        
                        if extension =='wav':
                            scipy.io.wavfile.write(os.path.join(root,new_name), sr, y)
                        if extension == 'npy':
                            np.save(os.path.join(root,new_name), y)
                except:
                    print(os.path.join(root,name))
                    os.remove(os.path.join(root,name))
                


def conver_folder_to_wav(folder_path):
    convert_folder_to_extension(folder_path=folder_path,extension='wav')


def conver_folder_to_npy(folder_path):
    convert_folder_to_extension(folder_path=folder_path,extension='npy')
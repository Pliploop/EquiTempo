import os
import librosa
import scipy
import numpy as np
from tqdm import tqdm

def convert_folder_to_extension(folder_path, new_folder = None, extension=  'wav', delete = False, sr = 16000):
    for root, dirs, files in os.walk(folder_path):
        if new_folder is None:
            folder = root
        else:
            folder = root.replace(folder_path,new_folder)

        for name in tqdm(files):
            if ".mp3" in name:
                try:
                    new_name = name[:-3]+extension
                    if not os.path.exists(os.path.join(folder,new_name)):
                        y,sr =  librosa.load(os.path.join(root,name),sr=sr)
                        if not os.path.exists(folder):
                            os.makedirs(folder)
                        if extension =='wav':
                            scipy.io.wavfile.write(os.path.join(folder,new_name), sr, y)
                        if extension == 'npy':
                            np.save(os.path.join(folder,new_name), y)
                            
                except Exception as e:
                    print(e)
                    if delete:
                        os.remove(os.path.join(root,name))
                


def convert_folder_to_wav(folder_path, new_folder = None, delete = False, sr = 16000):
    convert_folder_to_extension(folder_path=folder_path,extension='wav', new_folder=new_folder, delete=delete, sr = sr)


def convert_folder_to_npy(folder_path, new_folder = None, delete = False, sr = 16000):
    convert_folder_to_extension(folder_path=folder_path,extension='npy', new_folder=new_folder, delete=delete, sr =sr)
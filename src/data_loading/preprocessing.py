import torch

def pad_or_truncate(input_audio,target_len):
    new = torch.zeros((1,target_len))
    if input_audio.shape[1] > target_len:
        new = input_audio[0,:target_len]
    else:
        new[0,:input_audio.shape[1]] = input_audio
    return new

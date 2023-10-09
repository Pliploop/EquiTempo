import torch

def pad_or_truncate(input_audio,target_len):

    # if input_audio.dim() == 2:
    #     input_audio=input_audio.squeeze()

    new = torch.zeros((1,target_len))
    if input_audio.shape[-1] > target_len:
        new = input_audio[:,:target_len]
    else:
        new[:,:input_audio.shape[-1]] = input_audio
    return new

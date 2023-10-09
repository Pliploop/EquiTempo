import torch

def pad_or_truncate(input_audio,target_len):

    if input_audio.dim() == 2:
        input_audio=input_audio.squeeze()

    new = torch.zeros((1,target_len))
    if input_audio.shape[0] > target_len:
        new = torch.Tensor(input_audio).unsqueeze(0)[0,:target_len]
    else:
        new[0,:input_audio.shape[0]] = input_audio
    return new

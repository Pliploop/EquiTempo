import torch

def pad_or_truncate(input_audio,target_len):

    new = torch.zeros((target_len,))
    if input_audio.shape[-1] > target_len:
        new = input_audio[:target_len]
    else:
        new[:input_audio.shape[-1]] = input_audio
    return new


def power2db(S, amin=1e-10, top_db=80.0):
    log_spec = 10.0 * torch.log10(torch.maximum(torch.ones_like(S)*amin, S))
    if top_db is not None:
        log_spec = torch.maximum(log_spec, torch.amax(log_spec, (-2,-1), keepdim=True) - top_db)
    return log_spec
import math
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def intersperse(lst, item):
    result = [item] * (2 * len(lst) - 1)
    result[0::2] = lst
    return result

def load_wav_to_torch(full_path):
    from scipy.io.wavfile import read
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate

def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text

def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.LongTensor(max_len)).to(lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask

class HParams():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getattr__(self, item):
        return None

    def set_hparam(self, k, v):
        setattr(self, k, v)

def save_spectrogram(mel_spectrogram, path):
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spectrogram, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title('Mel Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.savefig(path)
    plt.close()

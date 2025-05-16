import os
import torch
import soundfile as sf
import numpy as np
from scipy import signal
from gtcrn import GTCRN

## load model
device = torch.device("cpu")
model = GTCRN().eval()
ckpt = torch.load(os.path.join('checkpoints', 'model_trained_on_dns3.tar'), map_location=device)
model.load_state_dict(ckpt['model'])

## load data
mix, fs = sf.read(os.path.join('test_wavs', 'mix.wav'), dtype='float32')
target_fs = 16000
if fs != target_fs:
    print(f"原始采样率: {fs}Hz, 重采样到: {target_fs}Hz")
    # 使用scipy进行重采样，避免librosa依赖
    num_samples = round(len(mix) * target_fs / fs)
    mix = signal.resample(mix, num_samples)
    fs = target_fs

## inference
input = torch.stft(torch.from_numpy(mix), 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)
with torch.no_grad():
    output = model(input[None])[0]
enh = torch.istft(output, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)

## save enhanced wav
sf.write(os.path.join('test_wavs', 'enh.wav'), enh.detach().cpu().numpy(), fs)

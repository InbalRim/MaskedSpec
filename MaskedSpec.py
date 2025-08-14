
import numpy as np
import torch
import torch.nn as nn
import random

class MaskedSpec(nn.Module):
    def __init__(self, device='cuda:0',patch_shape='bands',win_size=400):

        self.device = device
        self.window = torch.hamming_window(win_size).to(device)
        self.patch_shape = patch_shape
        self.initialize_patch_func()

    def create_gaussian_patch(self,feature_shape, center, sigma, threshold=0.1):
        rows, cols = feature_shape
        self.mesh_x, self.mesh_y = np.meshgrid(np.linspace(0, cols - 1, cols), np.linspace(0, rows - 1, rows))

        x0, y0 = center
        gaussian = np.exp(-((self.mesh_x - x0) ** 2 + (self.mesh_y - y0) ** 2) / (2 * sigma ** 2))
        patch_mask = gaussian > threshold

        return patch_mask

    def patch_square(self, feature, val, f, f0, f_mask):
        t = np.random.randint(10, f_mask)
        t0 = random.randint(0, feature.shape[-1] - t)
        feature[f0:f0 + f, t0:t0 + t] = val
        return feature
    def patch_bands(self, feature, val, f, f0, f_mask):
        feature[f0:f0 + f, :] = val
        return feature
    def patch_singles(self, feature, val, f, f0, f_mask):
        feature[f0:f0 + 1, :] = val
        return feature
    def patch_gauss(self, feature, val, f, f0, f_mask):
        t0 = random.randint(0, feature.shape[-1])
        patch_mask = self.create_gaussian_patch(feature_shape=feature.shape, center=[t0, f0], sigma=f)
        feature[patch_mask] = val

        return feature
    def patch_wn(self, feature, val, f, f0, f_mask):
        wn = np.random.normal(0, 1, feature.shape)
        patch_mask = wn > 0.9
        feature[patch_mask] = val

        return feature

    def initialize_patch_func(self):
        # Define the patch function based on self.patch_shape
        if self.patch_shape == 'squer':
            self.patch_func = self.patch_square
        elif self.patch_shape == 'bands':
            self.patch_func = self.patch_bands
        elif self.patch_shape == 'singles':
            self.patch_func = self.patch_singles
        elif self.patch_shape == 'gauss':
            self.patch_func = self.patch_gauss
        elif self.patch_shape == 'wn':
            self.patch_func = self.patch_wn
        elif self.patch_shape == 'random':
            self.patch_func = [self.patch_square,self.patch_bands,self.patch_singles,self.patch_gauss]
        else:
            raise ValueError('Choose a valid patch shape')
    def spec_masking(self,utt, masks_num, f_mask, nfft=1024, win_length=400, hop_length=160):
        '''
        Applies spectral masking to the given audio waveform to augment data by randomly masking
        frequency bands in the STFT domain.

        Parameters
        ----------
        utt : torch.Tensor
            1D tensor containing the audio waveform (time-domain samples).

        masks_num : int
            Number of frequency masks to apply to the spectrogram.

        f_mask : int
            Maximum width (in frequency bins) of each frequency mask.
            The actual width for each mask is randomly selected between 1 and `f_mask`.

        nfft : int, optional (default=1024)
            Number of FFT points used in the Short-Time Fourier Transform (STFT).

        win_length : int, optional (default=400)
            Window size (in samples) for each STFT frame.

        hop_length : int, optional (default=160)
            Hop length (in samples) between successive STFT frames.

        Returns
        -------
        torch.Tensor
        The audio waveform (time-domain) after applying spectral masking.
        '''

        feature = torch.stft(utt, n_fft=nfft, hop_length=hop_length,
                             win_length=win_length, window=self.window,
                             return_complex=True)

        #
        mean_magnitude = torch.mean(torch.abs(feature))
        mean_phase = torch.mean(torch.angle(feature))
        masking_value = mean_magnitude * torch.complex(torch.cos(mean_phase), torch.sin(mean_phase))

        max_f = feature.shape[-2]

        for i in range(masks_num):
            f = np.random.randint(1, f_mask)
            f0 = random.randint(0, max_f - f)
            try:
                feature = self.patch_func(feature, masking_value, f, f0)
            except:
                feature = random.choice(self.patch_func)(feature, masking_value, f, f0, f_mask)


        return torch.istft(feature, n_fft=nfft, hop_length=hop_length, win_length=win_length, window=self.window,
                           length=len(utt))

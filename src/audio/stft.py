import torch

from config import N_FFT, HOP_LENGTH, WIN_LENGTH


def create_window(device=None):
    """
    Create the Hann window used for STFT and inverse STFT.
    """
    return torch.hann_window(
        WIN_LENGTH,
        device=device
    )


def compute_stft(audio):
    """
    Convert an audio waveform into a complex STFT.

    Parameters
    ----------
    audio : torch.Tensor
        Audio waveform.

        Supported shapes:
        (samples,)
        (channels, samples)
        (batch, channels, samples)

    Returns
    -------
    torch.Tensor
        Complex STFT representation.
    """

    window = create_window(audio.device)

    stft = torch.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        return_complex=True
    )

    return stft


def compute_magnitude(stft):
    """
    Calculate the magnitude spectrogram from a complex STFT.
    """
    return torch.abs(stft)


def compute_phase(stft):
    """
    Calculate the phase of a complex STFT.
    """
    return torch.angle(stft)


def compute_istft(stft, length=None):
    """
    Reconstruct a waveform from a complex STFT.

    Parameters
    ----------
    stft : torch.Tensor
        Complex STFT representation.

    length : int, optional
        Desired number of output samples.

    Returns
    -------
    torch.Tensor
        Reconstructed waveform.
    """

    window = create_window(stft.device)

    audio = torch.istft(
        stft,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        length=length
    )

    return audio
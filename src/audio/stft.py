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

    Supported shapes:
    (samples,)
    (channels, samples)
    (batch, channels, samples)
    """

    window = create_window(audio.device)

    original_shape = audio.shape

    # Batched stereo audio:
    # (batch, channels, samples)
    if audio.ndim == 3:
        batch_size, channels, samples = audio.shape

        audio = audio.reshape(
            batch_size * channels,
            samples
        )

    stft = torch.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        return_complex=True
    )

    # Restore batch and channel dimensions
    if len(original_shape) == 3:
        stft = stft.reshape(
            batch_size,
            channels,
            stft.shape[-2],
            stft.shape[-1]
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
    Reconstruct waveform from a complex STFT.

    Supported shapes:
    (freq, time)
    (channels, freq, time)
    (batch, channels, freq, time)
    """

    window = create_window(stft.device)

    original_shape = stft.shape

    if stft.ndim == 4:
        batch_size, channels, freq, time = stft.shape

        stft = stft.reshape(
            batch_size * channels,
            freq,
            time
        )

    audio = torch.istft(
        stft,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        length=length
    )

    if len(original_shape) == 4:
        audio = audio.reshape(
            batch_size,
            channels,
            audio.shape[-1]
        )

    return audio
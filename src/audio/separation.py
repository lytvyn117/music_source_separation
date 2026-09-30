from src.audio.stft import compute_istft


def apply_mask(mixture_stft, mask):
    """
    Apply a time-frequency mask to a complex mixture STFT.

    Parameters
    ----------
    mixture_stft : torch.Tensor
        Complex STFT of the mixture.

    mask : torch.Tensor
        Real-valued mask with the same shape as the magnitude spectrogram.

    Returns
    -------
    torch.Tensor
        Masked complex STFT.
    """

    return mixture_stft * mask


def reconstruct_source(mixture_stft, mask, length=None):
    """
    Reconstruct a separated audio source from a mixture STFT and mask.

    Parameters
    ----------
    mixture_stft : torch.Tensor
        Complex STFT of the mixture.

    mask : torch.Tensor
        Predicted source mask.

    length : int, optional
        Desired waveform length.

    Returns
    -------
    torch.Tensor
        Reconstructed waveform.
    """

    source_stft = apply_mask(
        mixture_stft,
        mask
    )

    source_audio = compute_istft(
        source_stft,
        length=length
    )

    return source_audio
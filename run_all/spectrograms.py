import numpy as np
from tqdm.auto import tqdm
from pathlib2 import Path
from tqdm.autonotebook import tqdm
from avgn.utils.paths import ensure_dir
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import librosa
from avgn.visualization.spectrogram import draw_spec_set
from avgn.utils.audio import float32_to_int16, int16_to_float32
from scipy import signal


def make_spec(
    syll_wav,
    fs,
    hparams,
    mel_matrix=None,
    use_tensorflow=False,
    use_mel=True,
    return_tensor=False,
    norm_uint8=False,
):
    """
    """
      # convert to float
    if type(syll_wav[0]) == int:
        syll_wav = int16_to_float32(syll_wav)
        
    stft = librosa.stft(
        y=signal.lfilter([1, -hparams.preemphasis], [1], syll_wav),
        n_fft=hparams.n_fft,
        hop_length=int(hparams.hop_length_ms / 1000 * fs),
        win_length=int(hparams.win_length_ms / 1000 * fs),
    )
    S = 20 * np.log10(np.maximum(1e-5, np.abs(stft))) - hparams.ref_level_db
    spec = np.clip((S - hparams.min_level_db) / -hparams.min_level_db, 0, 1)
    spec = np.dot(spec.T, mel_matrix).T

    return spec


def create_spectrograms(dataset, syllable_df, n_jobs, verbosity, FIGs_LOC):
    from avgn.signalprocessing.create_spectrogram_dataset import make_spec
    with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
        # create spectrograms
        syllables_spec = parallel(
            delayed(make_spec)(
                syllable,
                rate,
                hparams=dataset.hparams,
                mel_matrix=dataset.mel_matrix,
                use_mel=True,
                use_tensorflow=False,
            )
            for syllable, rate in tqdm(
                zip(syllable_df.audio.values, syllable_df.rate.values),
                total=len(syllable_df),
                desc="getting syllable spectrograms",
                leave=False,
            )
        )
        # plot an example syllable
    # plt.matshow(syllables_spec[10])
    ensure_dir(FIGs_LOC)
    fig_out = Path(FIGs_LOC, "spectrograms.png")
    plt.savefig(fig_out)
    plt.imsave(arr=syllables_spec[10],fname=fig_out)
    return syllables_spec, syllable_df

def norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def normalize_spectrograms(syllables_spec):
    syllables_spec = [(norm(i)*255).astype('uint8') for i in tqdm(syllables_spec)]
    return syllables_spec

def pad_spectrograms(syllables_spec, n_jobs, verbosity, figures_loc):
    from avgn.signalprocessing.create_spectrogram_dataset import pad_spectrogram
    syll_lens = [np.shape(i)[1] for i in syllables_spec]
    plt.hist(syll_lens)
    pad_length = np.max(syll_lens)
    with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
        syllables_spec = parallel(
            delayed(pad_spectrogram)(spec, pad_length)
            for spec in tqdm(
                syllables_spec, desc="padding spectrograms", leave=False
            )
        )
    # lets take a look at these spectrograms
    draw_spec_set(syllables_spec, zoom=1, maxrows=10, colsize=25, fig_loc=figures_loc/"pad_spectrograms.png")
    return syllables_spec

def rescale_spectrograms(syllables_spec, n_jobs, verbosity, log_scaling_factor, figures_loc):
    from avgn.signalprocessing.create_spectrogram_dataset import log_resize_spec
    with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
        syllables_spec = parallel(
            delayed(log_resize_spec)(spec, scaling_factor=log_scaling_factor)
            for spec in tqdm(syllables_spec, desc="scaling spectrograms", leave=False)
        )
    # lets take a look at these spectrograms
    draw_spec_set(syllables_spec, zoom=1, maxrows=10, colsize=25, fig_loc=figures_loc/"rescale_spectrograms.png")
    return syllables_spec

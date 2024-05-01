import pathlib
from IPython import display


import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt


def squeeze(audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels


def main():
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    DATASET_PATH: str = 'data/mini_speech_commands'

    data_dir = pathlib.Path(DATASET_PATH)
    if not data_dir.exists():
        tf.keras.utils.get_file(
            'mini_speech_commands.zip',
            origin="https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
            extract=True,
            cache_dir='.',
            cache_subdir='data')

    commands = np.array(tf.io.gfile.listdir(str(data_dir)))
    commands = commands[(commands != 'README.md') & (commands != '.DS_Store')]
    print('Commands:', commands, end='\n')

    train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
        directory=data_dir,
        batch_size=64,
        validation_split=0.2,
        seed=0,
        output_sequence_length=16000,
        subset='both')

    label_names = np.array(train_ds.class_names)
    print("label names:", label_names)

    print("sets 'commands' and 'labels' contain the same str values:", set(commands) == set(label_names))

    train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)  # single channel audio, drop the extra axis
    val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

    #  Split a test dataset from the validation dataset
    test_ds = val_ds.shard(num_shards=2, index=0)
    val_ds = val_ds.shard(num_shards=2, index=1)

    #  pick an element to show each shape:
    #  example_audio = (batch, count)
    #  example_labels.shape = (64,)
    for example_audio, example_labels in train_ds.take(1):
        print(example_audio.shape)
        print(example_labels.shape)

    print(label_names[[1, 1, 3, 0]])

    # plot labels (in steps of 1), and audio signal amplitude (-1.0 to 1.0) in TIME domain only
    plt.figure(figsize=(16, 10))
    rows = 3
    cols = 3
    n = rows * cols
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        audio_signal = example_audio[i]
        plt.plot(audio_signal)
        plt.title(label_names[example_labels[i]])
        plt.yticks(np.arange(-1.2, 1.2, 0.2))
        plt.ylim([-1.1, 1.1])

    plt.show()

    def get_spectrogram(waveform):
        # Convert the waveform to a spectrogram via a STFT.

        # signals: A[..., samples] float32 / float64 Tensor of real - valued signals.
        #
        # frame_length: An integer scalar Tensor.The window length in samples.

        # frame_step: An integer scalar Tensor.The number of samples to step.

        # fft_length: An integer scalar Tensor.The size of the FFT to apply. If not provided, uses the smallest power of
        # 2 enclosing frame_length.

        # window_fn: A callable that takes a window length and a dtype keyword argument and returns a[window_length]
        # Tensor of samples in the provided datatype.If set to None, no windowing is used

        # pad_end: Whether to pad the end of signals with zeros when the provided frame length and step produces a
        # frame that lies partially past its end.

        # name: An optional name for the operation.

        # 1. Convert the waveform to a spectrogram via a STFT.
        spectrogram = tf.signal.stft(
            waveform, frame_length=255, frame_step=128)

        # 2.  Obtain the magnitude of the STFT.
        spectrogram = tf.abs(spectrogram)

        #3. Add a `channels` dimension, so that the spectrogram can be used as image-like input data with convolution
        # layers (which expect a shape (`batch_size`, `height`, `width`, `channels`).
        spectrogram = spectrogram[..., tf.newaxis]
        return spectrogram

    for i in range(3):
        label = label_names[example_labels[i]]
        waveform = example_audio[i]
        spectrogram = get_spectrogram(waveform)

        print('Label:', label)
        print('Waveform shape:', waveform.shape)
        print('Spectrogram shape:', spectrogram.shape)
        print('Audio playback')
        display.display(display.Audio(waveform, rate=16000))


if __name__ == '__main__':
    main()

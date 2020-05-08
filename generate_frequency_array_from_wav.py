from os import path
from math import floor

import click
import numpy as np
from scipy.io import wavfile
from scipy import signal


def aggressive_array_split(array, parts):
    # Potentially exclude some indeces to get equal bins
    end_index = len(array) - (len(array) % parts)
    return np.split(array[:end_index], parts)


def generate_frame_samples(samples, sample_rate, fps):
    samples_per_frame = int(sample_rate / fps)
    
    for frame_num in range(floor(len(samples) / samples_per_frame)):
        yield samples[frame_num * samples_per_frame: (frame_num + 1) * samples_per_frame]


def derive_output_filepath(input_filepath, number_of_bins, algorithm, extension):
    base_path, filename = path.split(input_filepath)
    name, _ = path.splitext(filename)
    
    new_name = f'{name}_freq_arrays_{number_of_bins}-bins_{algorithm}.{extension}'
    return path.join(base_path, new_name)


def simple_periodogram(samples, sample_rate, *args, **kwargs):
    _, spectral_density = signal.periodogram(samples, sample_rate, return_onesided=True)
    return spectral_density


def welch_periodogram(samples, sample_rate, bin_count):
    segment_size = int(samples.size / bin_count)
    _, spectral_density = signal.welch(
        samples,
        sample_rate,
        nperseg=segment_size,
        return_onesided=True
    )
    return spectral_density


PERIODOGRAM_FUNCTION_MAP = {
    'simple': simple_periodogram,
    'welch': welch_periodogram,
}


@click.command()
@click.argument('input_filepath')
@click.option('--fps', default=30, help='FPS determining how many frequency vectors per second.')
@click.option('-n', '--number-of-bins', default=3, help='The number of frequency bins per vector.')
@click.option(
    '-f',
    '--output-format',
    type=click.Choice(('WAV', 'NPY')),
    default='WAV',
    help='The format to save the spectral desnity array output.'
)
@click.option(
    '-a',
    '--algorithm',
    type=click.Choice(PERIODOGRAM_FUNCTION_MAP.keys()),
    required=True,
    help='Algorithm used to compute spectral density.'
)
@click.option(
    '-m',
    '--min-density',
    default=0,
    type=float,
    help='The lowest permissible maximum density for a frequency bin. Those lower are ommitted.'
)
def generate_array(input_filepath, fps, number_of_bins, output_format, algorithm, min_density):
    periodogram_function = PERIODOGRAM_FUNCTION_MAP[algorithm]
    output_filepath = derive_output_filepath(input_filepath, number_of_bins, algorithm, output_format.lower())
    sample_rate, samples = wavfile.read(input_filepath)

    all_bins = []
    for index, frame_samples in enumerate(generate_frame_samples(samples, sample_rate, fps)):
        spectral_density = periodogram_function(
            frame_samples, 
            sample_rate, 
            bin_count=number_of_bins
        )

        bin_arrays = aggressive_array_split(spectral_density, number_of_bins)
        bins = np.sum(bin_arrays, axis=1)
        all_bins.append(bins)

    all_bins_array = np.array(all_bins)
    normalised_bins = all_bins_array / np.max(all_bins_array) 
    pruned_bins = normalised_bins[:, np.max(normalised_bins, axis=0) >= min_density]
    print(f'Pruned {normalised_bins.shape[1]} bins down to {pruned_bins.shape[1]}')

    print(f'Saving to {output_filepath}')
    if output_format == 'NPY':
        np.save(output_filepath, pruned_bins)
    else:
        wavfile.write(output_filepath, fps, pruned_bins)



if __name__ == '__main__':
    generate_array()

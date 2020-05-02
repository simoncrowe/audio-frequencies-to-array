from os import path
from math import floor

import click
import numpy as np
from scipy.io.wavfile import read as wavread
from scipy.signal import welch


def aggressive_array_split(array, parts):
    # Potentially exclude some indeces to get equal bins
    end_index = len(array) - (len(array) % parts)
    return np.split(array[:end_index], parts)


def generate_frame_samples(samples, sample_rate, fps):
    samples_per_frame = int(sample_rate / fps)
    
    for frame_num in range(floor(len(samples) / samples_per_frame)):
        yield samples[frame_num * samples_per_frame: (frame_num + 1) * samples_per_frame]


def derive_output_filepath(input_filepath, number_of_bins):
    base_path, filename = path.split(input_filepath)
    name, _ = path.splitext(filename)
    
    new_name = f'{name}_freq_arrays_{number_of_bins}-bins.npy'
    return path.join(base_path, new_name)


@click.command()
@click.argument('input_filepath')
@click.option('--fps', default=30, help='FPS determining how many frequency vectors per second.')
@click.option('-n', '--number-of-bins', default=3, help='The number of frequency bins per vector.')
def generate_array(input_filepath, fps, number_of_bins):
    output_filepath = derive_output_filepath(input_filepath, number_of_bins)
    sample_rate, samples = wavread(input_filepath)

    all_bins = []
    for index, frame_samples in enumerate(generate_frame_samples(samples, sample_rate, fps)):
        segment_size = int(frame_samples.size / number_of_bins)
        _, spectral_density = welch(
            frame_samples, 
            sample_rate, 
            nperseg=segment_size, 
            return_onesided=True
        )
        print(spectral_density.shape) 
        bin_arrays = aggressive_array_split(spectral_density, number_of_bins)
        bins = np.sum(bin_arrays, axis=1)
        all_bins.append(bins)

    all_bins_array = np.array(all_bins)
    normalised_bins = all_bins_array / np.max(all_bins_array) 
    print('Normalised bins:')
    print(normalised_bins)
    print(f'Saving to {output_filepath}')
    np.save(output_filepath, normalised_bins)


if __name__ == '__main__':
    generate_array()

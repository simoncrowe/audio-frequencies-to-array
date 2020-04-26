from os import path
from math import floor

import click
import numpy as np
from scipy.io.wavfile import read as wavread

def aggressive_array_split(array, parts):
    # Potentially exclude some indeces to get equal bins
    end_index = len(array) - (len(array) % parts)
    return np.split(array[:end_index], parts)


def generate_frame_samples(samples, sample_rate, fps):
    samples_per_frame = int(sample_rate / fps)
    
    for frame_num in range(floor(len(samples) / samples_per_frame)):
        yield samples[frame_num * samples_per_frame: (frame_num + 1) * samples_per_frame]


def derive_output_filepath(input_filepath):
    base_path, filename = path.split(input_filepath)
    name, extension = path.splitext(filename)
    
    new_name = f'{name}_freq_arrays{extension}'
    return path.join(base_path, new_name)


@click.command()
@click.argument('input_filepath')
@click.option('--fps', default=30, help='FPS determining how many frequency vectors per second.')
@click.option('-n', '--number-of-bins', default=3, help='The number of frequency bins per vector.')
@click.option(
    '-o', 
    '--output-filepath', 
    default=None, 
    help='Path to save .npy array. If none is specified, it will be based on the input.'
)
def generate_array(input_filepath, fps, number_of_bins, output_filepath):
    if output_filepath is None:
        output_filepath = derive_output_filepath(input_filepath)

    sample_rate, samples = wavread(input_filepath)

    # Calcualte max "energy" (a bit like amplitude) for later scaling
    dft_energy_array = np.abs(np.fft.fft(samples))
    max_energy = np.max(dft_energy_array)

    all_bins = []
    for index, frame_samples in enumerate(generate_frame_samples(samples, sample_rate, fps)):
        dft_energy_array = np.abs(np.fft.fft(frame_samples))
        # Split and flip negative half of Discrete Fourier Transform array
        # (Absolute values are more-or-less mirrored.)
        energy_positive, energy_negative = aggressive_array_split(dft_energy_array, 2)
        energy_negative = np.flip(energy_negative)    
        combined_energy = energy_positive + energy_negative
        
        bin_arrays = aggressive_array_split(combined_energy, number_of_bins)
        bins = np.sum(bin_arrays, axis=1)
        all_bins.append(bins)

    all_bins_array = np.array(all_bins)
    normalised_bins = all_bins / np.max(all_bins)
    print('Normalised bins:')
    print(normalised_bins)
    print(f'Saving to {output_filepath}')
    np.save(output_filepath, normalised_bins)


        


if __name__ == '__main__':
    generate_array()

import numpy as np


def generate_sine_wave(freq, duration, gain=0.5, phase_offset=0, sample_rate=44100):
    phase_duration = freq / sample_rate
    samples_per_phase = sample_rate / freq
    num_samples = (sample_rate * duration) + samples_per_phase
    
    sine_samples = np.sin(2 * np.pi * np.arange(num_samples) * phase_duration)
    sine_samples *= gain
    
    samples_per_phase = sample_rate / freq
    start_slice = int(samples_per_phase* phase_offset)
    end_slice = int(samples_per_phase * (1 - phase_offset))
    
    if phase_offset not in (1, 0):
        end_slice += 1

    if end_slice == 0:
        return sine_samples.astype(np.float32)[start_slice:]
    else:
        return sine_samples.astype(np.float32)[start_slice: - end_slice]



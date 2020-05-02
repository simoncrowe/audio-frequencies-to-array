import numpy as np


def generate_sine_wave(freq, duration, amp=0.5, phase_offset=0, sample_rate=44100):
    phase_duration = freq / sample_rate
    samples_per_phase = sample_rate / freq
    num_samples = (sample_rate * duration) + (3 * samples_per_phase)
    
    sine_samples = np.sin(2 * np.pi * np.arange(num_samples) * phase_duration)
    sine_samples *= amp
    
    samples_per_phase = sample_rate / freq
    start_slice = int(samples_per_phase + (samples_per_phase * phase_offset))
    end_slice = int(samples_per_phase + (samples_per_phase * (1 - phase_offset)))
    
    return sine_samples.astype(np.float32)[start_slice: - end_slice]



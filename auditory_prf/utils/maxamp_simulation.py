import thorns.waves
from auditory_prf.stimuli.soundgen import SoundGen
import numpy as np
from tqdm import tqdm

# Initialize the generator
sample_rate = 44100
tau = 0.005
sound_maker = SoundGen(sample_rate=sample_rate, tau=tau)
# Parameters for simulation
frequencies = np.arange(125, 2501, 1)  # Frequencies from 200 Hz to 2000 Hz
harmonic_factors = np.arange(0.1, 1.1, 0.1)  # Harmonic factors from 0.1 to 1.0
tone_duration = 0.2  # 200 ms
max_amplitudes = []
# Simulate to find max amplitude
max_s = 0 # Initialize max amplitude tracker

# Calculate total iterations for progress bar
total_iterations = 20 * len(frequencies) * len(harmonic_factors)

for num_harmonic in tqdm(range(1, 21), desc="Harmonics"):  # 1 to 20 harmonics
    for freq in frequencies:
        for harmonic_factor in harmonic_factors:
            sound = sound_maker.sound_maker(freq=freq, num_harmonics=num_harmonic,
                                        tone_duration=tone_duration,
                                        harmonic_factor=harmonic_factor)
            max_amplitudes.append(np.max(np.abs(sound)))
            max_s = max(max_s, np.max(np.abs(sound)))


# Determine the overall maximum amplitude
overall_max_amplitude = np.max(max_amplitudes)
print(f"Calculated maximum amplitude across all sounds: {overall_max_amplitude:.4f}")
print(f"Tracked maximum amplitude using max_s: {max_s:.4f}")



# Now you can use overall_max_amplitude in your other scripts to normalize sounds accordingly.
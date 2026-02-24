
# Created by: Ekim Celikay

import numpy as np
import thorns
from scipy.signal import firwin, filtfilt



class SoundGen:

    def __init__(self, sample_rate, tau):
        """
        Initialize the CreateSound instance.
        :param sample_rate: Sample rate of sounds ( per second).
        :param tau: The ramping window in seconds.
        """
        self.sample_rate = sample_rate
        self.tau = tau

    def sound_maker(self,
                    freq,
                    num_harmonics,
                    tone_duration,
                    harmonic_factor,
                    dbspl=60,
                    ):
        """
        Generate a harmonic complex tone.

        :param freq: Base frequency in Hz.
        :param num_harmonics: Number of harmonic tones.
        :param tone_duration: Duration of each tone in seconds.
        :param harmonic_factor: Harmonic amplitude decay factor for each tone.
        :param dbspl: Desired dbspl (loudness) level (default: 60 dB).
        :return: sound: np.ndarray: array of audio samples
                        representing the harmonic complex tone.
        """
        # Create the time array
        t = np.linspace(0,
                        tone_duration,
                        int(self.sample_rate * tone_duration),
                        endpoint=False)
        # Initialize the sound array
        sound = np.zeros_like(t)
        # Generate the harmonics
        for k in range(1, num_harmonics + 1):
            omega = 2 * np.pi * freq * k
            harmonic = np.sin(omega * t)
            amplitude = harmonic_factor ** (k - 1)
            sound = sound + amplitude * harmonic

        # Normalize to desired dbspl level
        normalized_sound = thorns.waves.set_dbspl(sound, dbspl)
        # found this value in simulation,`maxamp_simulation.py`
        max_amplitude = 0.2753
        normalized_sound = normalized_sound / (max_amplitude + 0.01)

        return normalized_sound

    def noise_maker(self, tone_duration, seed=None):
        if seed is not None:
            np.random.seed(seed)
        sound = np.random.normal(0, 1, tone_duration * self.sample_rate)
        return sound

    def bandpass_filter_FIR(self,
                            tone_duration,
                            dbspl,
                            lowcut,
                            highcut,
                            numtaps,
                            seed=None,
                            ):
        if seed is not None:
            np.random.seed(seed)

        # Generate the white noise using noise_maker method
        noise = self.noise_maker(tone_duration)
        # design FIR bandpass filter
        fir_coeffs = firwin(numtaps,
                            [lowcut, highcut],
                            pass_zero=True,
                            fs=self.sample_rate)

        # apply zero-phase FIR filter (no delay)
        filtered_noise = filtfilt(fir_coeffs, [1.0], noise)

        # set to desired dbspl using thorns
        normalized_noise = thorns.waves.set_dbspl(filtered_noise, dbspl)
        return normalized_noise

    def generate_multiple_band_limited_noises(self,
                                              n_trials,
                                              tone_duration,
                                              lowcut,
                                              highcut,
                                              numtaps,
                                              dbspl,
                                              base_seed=0
                                              ):
        """
        Generate multiple reproducible band-limited white noise stimuli.

        Parameters:
            n_trials (int): Number of stimuli to generate
            tone_duration (float): Duration in seconds
            lowcut (float): Lower cutoff frequency of bandpass filter (Hz)
            highcut (float): Upper cutoff frequency of bandpass filter (Hz)
            numtaps (int): FIR filter length
            dbspl (float): Desired dB SPL
            base_seed (int): Seed offset to allow reproducible variation

        Returns:
            list of numpy.ndarray: List of filtered, level-adjusted noise signals
        """
        fir_coeffs = firwin(numtaps,
                            [lowcut, highcut],
                            pass_zero=False,
                            fs=self.sample_rate)
        n_samples = int(tone_duration * self.sample_rate)
        stimuli = []

        for i in range(n_trials):
            seed = base_seed + i
            np.random.seed(seed)
            white_noise = np.random.normal(0, 1, n_samples)
            filtered = filtfilt(fir_coeffs, [1.0], white_noise)
            scaled = thorns.waves.set_dbspl(filtered, dbspl)
            stimuli.append(scaled)

        return stimuli

    def ramp_in_out(self, sound):
        sound = sound.copy()
        L = int(self.tau * self.sample_rate)
        hw = np.hamming(2*L)

        sound = sound.copy()
        sound[:L] *= hw[:L]
        sound[-L:] *= hw[L:]
        return sound

    def sine_ramp(self, sound):
        L = int(self.tau * self.sample_rate)
        t = np.linspace(0, L / self.sample_rate, L)
        sine_window = np.sin(np.pi * t / (2 * self.tau)) ** 2  # Sine fade-in

        sound = sound.copy()
        sound[:L] *= sine_window  # Apply fade-in
        sound[-L:] *= sine_window[::-1]  # Apply fade-out

        return sound

    def generate_sequence(self,
                          freq,
                          num_harmonics,
                          tone_duration,
                          harmonic_factor,
                          dbspl,
                          total_duration,
                          isi,
                          stereo=True,
                          ):

        # Generate the tone using the sound_maker method
        sound = self.sound_maker(freq,
                                 num_harmonics,
                                 tone_duration,
                                 harmonic_factor,
                                 dbspl)
        # Sine ramp
        ramped_sound = self.sine_ramp(sound)
        # Calculate the number of tones that can fit into the total duration
        # Call the function here
        num_tones, isi_samples, total_samples = \
            self.calculate_num_tones(tone_duration, isi, total_duration)

        # Generate the sequence with ISI gaps between each tone
        sequence = np.array([])

        for i in range(num_tones):
            sequence = np.concatenate((sequence, ramped_sound))

            # Add ISI (silent gap) between tones, but not after the last tone
            if i < num_tones - 1:
                sequence = np.concatenate((sequence, np.zeros(isi_samples)))

        # Pad or truncate to reach exactly total_duration
        if len(sequence) < total_samples:
            # Pad with zeros to reach exact duration
            padding = np.zeros(total_samples - len(sequence))
            sequence = np.concatenate((sequence, padding))
        elif len(sequence) > total_samples:
            # Truncate if somehow exceeded
            sequence = sequence[:total_samples]

        # If stereo is desired, duplicate the mono sequence into two channels
        if stereo:
            sequence = np.column_stack((sequence, sequence))

        return sequence

    def sample_frequencies_gaussian(self,
                                    freq_mean,
                                    freq_std,
                                    num_samples,
                                    freq_min=None,
                                    freq_max=None,
                                    seed=None
                                    ):

        if seed is not None:
            np.random.seed(seed)

        # Sample from Gaussian distribution
        frequencies = np.random.normal(freq_mean, freq_std, num_samples)

        # Apply min/max constraints if provided
        if freq_min is not None and freq_max is not None:
            frequencies = np.clip(frequencies, freq_min, freq_max)
        elif freq_min is not None:
            frequencies = np.maximum(frequencies, freq_min)
        elif freq_max is not None:
            frequencies = np.minimum(frequencies, freq_max)

        # Ensure no negative frequencies (physically meaningless)
        frequencies = np.abs(frequencies)

        return frequencies

    def calculate_num_tones(self,
                            tone_duration,
                            isi,
                            total_duration,
                            ):
        total_samples = int(total_duration * self.sample_rate)
        isi_samples = int(isi * self.sample_rate)
        tone_samples = int(tone_duration * self.sample_rate)
        num_tones = int((total_samples + isi_samples) //
                        (tone_samples + isi_samples))
        return num_tones, isi_samples, total_samples

    def generate_sequence_from_freq_array(self,
                                          frequencies,
                                          num_harmonics,
                                          tone_duration,
                                          harmonic_factor,
                                          dbspl,
                                          isi,
                                          total_duration=None,
                                          stereo=True,
                                          ):
        isi_samples = int(isi * self.sample_rate)
        sequence = np.array([])

        # Generate each tone at its specified frequency
        for freq in frequencies:
            # Generate tone at this frequency
            sound = self.sound_maker(freq, num_harmonics, tone_duration,
                                     harmonic_factor, dbspl)

            # Apply ramping
            ramped_sound = self.sine_ramp(sound)

            # Add tone to sequence
            sequence = np.concatenate((sequence, ramped_sound))

            # Add ISI (silence) after tone
            sequence = np.concatenate((sequence, np.zeros(isi_samples)))

        # Trim to exact total duration if specified
        if total_duration is not None:
            total_samples = int(total_duration * self.sample_rate)
            if len(sequence) > total_samples:
                sequence = sequence[:total_samples]

        # If stereo is desired, duplicate the mono sequence into two channels
        if stereo:
            sequence = np.column_stack((sequence, sequence))
        return sequence

    def generate_sequence_gaussian_freq(self,
                                        freq_mean,
                                        freq_std,
                                        num_harmonics,
                                        tone_duration,
                                        harmonic_factor,
                                        dbspl,
                                        total_duration,
                                        isi,
                                        freq_min=None,
                                        freq_max=None,
                                        seed=None,
                                        stereo=True
                                        ):

        """
        Convenience method: Generate sequence with Gaussian-distributed frequencies.

        This combines the modular methods above into one call.
        :param freq_mean: Mean frequency for Gaussian sampling.
        :param freq_std: Standard deviation for Gaussian sampling.
        :param num_harmonics: Number of harmonics for each tone.
        :param tone_duration: Duration of each tone in seconds.
        :param harmonic_factor: Amplitude decay factor for harmonics.
        :param total_duration: Total duration of the sequence in seconds.
        :param isi: Inter-stimulus interval in seconds.
        :param freq_min: Minimum frequency constraint (optional).
        :param freq_max: Maximum frequency constraint (optional).
        :param seed: Random seed for reproducibility (optional).
        :param stereo: Whether to generate stereo output (default True).

        :return: tuple: (sequence, frequencies)
            - sequence: np.ndarray: Generated sound sequence.
            - frequencies: np.ndarray: Array of sampled frequencies used in
            the sequence.

        """
        # Step 1: Calculate how many tones we need
        num_tones = self.calculate_num_tones(tone_duration, isi, total_duration)

        # Step 2: Sample frequencies
        frequencies = self.sample_frequencies_gaussian(
            freq_mean, freq_std, num_tones, freq_min, freq_max, seed
        )

        # Step 3: Generate sequence from frequency array
        sequence = self.generate_sequence_from_freq_array(
            frequencies,
            num_harmonics,
            tone_duration,
            harmonic_factor,
            dbspl,
            isi,
            total_duration,
            stereo=stereo,
        )

        return sequence, frequencies


from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ToneConfig:
    """Configuration for ONE tone stimulus generation."""

    # ================ Stimulus Parameters ================
    sample_rate: int = 100000
    tau_ramp: float = 0.005
    num_harmonics: int = 1
    tone_duration: float = 0.200
    harmonic_factor: float = 0.5
    db_range: List[int] = field(default_factory=lambda: [60])
    freq_range: Tuple[float, float, int] = (125.0, 2500.0, 40)

    def get_stimulus_params(self):
        """For SoundGen.sound_maker()."""
        return {
            'num_harmonics': self.num_harmonics,
            'duration': self.tone_duration,
            'harmonic_factor': self.harmonic_factor
        }

    def get_folder_params(self):
        """For folder naming."""
        num_tones = self.freq_range[2]
        return {
            'num_tones': num_tones,
            'db_range': self.db_range,
            'freq_range': self.freq_range
        }

# Usage:
# trains = cochlea.run_zilany2014(sound, fs, **config.get_cochlea_kwargs())
# tone = sound_gen.sound_maker(freq, **config.get_stimulus_params())
# folder_mgr.with_params(**config.get_folder_params())
# use the tone_generator directly

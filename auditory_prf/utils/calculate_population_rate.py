import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)

def calculate_population_rate(
        fiber_responses: Dict[str, np.ndarray],
        hsr_weight: float = 0.63,
        msr_weight: float = 0.25,
        lsr_weight: float = 0.12
        ) -> np.ndarray:
    """Calculate weighted population rate across fiber types.

    Typical auditory nerve fiber proportions:
    - 63% High Spontaneous Rate (HSR)
    - 25% Medium Spontaneous Rate (MSR)
    - 12% Low Spontaneous Rate (LSR)

    Works with both mean rates (1D) and PSTH data (2D) - output shape matches input shape.

    Args:
        fiber_responses: Dict[fiber_type, array] - responses per fiber type
            - For mean rates: Dict[fiber_type, array(num_cf)]
            - For PSTH: Dict[fiber_type, array(num_cf, n_bins)]
        hsr_weight: Weight for HSR fibers (default: 0.63)
        msr_weight: Weight for MSR fibers (default: 0.25)
        lsr_weight: Weight for LSR fibers (default: 0.12)

    Returns:
        np.ndarray - weighted population response (shape matches input arrays)
            - For mean rates: shape (num_cf,)
            - For PSTH: shape (num_cf, n_bins)
    """
    weights = {'hsr': hsr_weight, 'msr': msr_weight, 'lsr': lsr_weight}

    # Validate weights sum to 1
    weight_sum = sum(weights.values())
    if not np.isclose(weight_sum, 1.0):
        logger.warning(f"Weights sum to {weight_sum:.3f}, normalizing to 1.0")
        weights = {k: v/weight_sum for k, v in weights.items()}

    # Calculate weighted sum
    population_rate = np.zeros_like(fiber_responses['hsr'])

    for fiber_type, weight in weights.items():
        if fiber_type in fiber_responses:
            population_rate += weight * fiber_responses[fiber_type]
        else:
            logger.warning(f"Missing {fiber_type} in fiber_responses")

    return population_rate
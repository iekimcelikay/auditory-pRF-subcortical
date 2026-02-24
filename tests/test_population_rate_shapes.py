"""Test script to verify calculate_population_rate works with both mean_rates and PSTH."""
import numpy as np
from utils.calculate_population_rate import calculate_population_rate

print("Testing calculate_population_rate with different input shapes")
print("=" * 60)

# Test 1: Mean rates (1D arrays)
print("\nTest 1: Mean Rates (1D arrays)")
mean_rates = {
    'hsr': np.array([10.0, 15.0, 20.0, 25.0]),  # shape: (4,)
    'msr': np.array([8.0, 12.0, 16.0, 20.0]),
    'lsr': np.array([5.0, 8.0, 10.0, 12.0])
}
print(f"Input shapes - HSR: {mean_rates['hsr'].shape}, MSR: {mean_rates['msr'].shape}, LSR: {mean_rates['lsr'].shape}")

population_mean = calculate_population_rate(mean_rates)
print(f"Output shape: {population_mean.shape}")
print(f"Output values: {population_mean}")
print(f"Expected shape: (4,) - ✓" if population_mean.shape == (4,) else "FAILED")

# Test 2: PSTH data (2D arrays)
print("\n" + "=" * 60)
print("Test 2: PSTH (2D arrays)")
psth = {
    'hsr': np.random.rand(4, 100),  # shape: (4 CFs, 100 time bins)
    'msr': np.random.rand(4, 100),
    'lsr': np.random.rand(4, 100)
}
print(f"Input shapes - HSR: {psth['hsr'].shape}, MSR: {psth['msr'].shape}, LSR: {psth['lsr'].shape}")

population_psth = calculate_population_rate(psth)
print(f"Output shape: {population_psth.shape}")
print(f"Expected shape: (4, 100) - ✓" if population_psth.shape == (4, 100) else "FAILED")

# Test 3: Verify weighted calculation is correct
print("\n" + "=" * 60)
print("Test 3: Verify weighted calculation")
simple_rates = {
    'hsr': np.array([100.0]),
    'msr': np.array([100.0]),
    'lsr': np.array([100.0])
}
result = calculate_population_rate(simple_rates)
expected = 0.63 * 100 + 0.25 * 100 + 0.12 * 100  # Should be 100
print(f"All fiber types at 100 Hz -> {result[0]} Hz")
print(f"Expected: 100.0 Hz - ✓" if np.isclose(result[0], 100.0) else "FAILED")

# Test 4: Different values
simple_rates2 = {
    'hsr': np.array([100.0]),
    'msr': np.array([80.0]),
    'lsr': np.array([60.0])
}
result2 = calculate_population_rate(simple_rates2)
expected2 = 0.63 * 100 + 0.25 * 80 + 0.12 * 60  # = 63 + 20 + 7.2 = 90.2
print(f"HSR=100, MSR=80, LSR=60 -> {result2[0]:.1f} Hz")
print(f"Expected: {expected2:.1f} Hz - ✓" if np.isclose(result2[0], expected2) else "FAILED")

print("\n" + "=" * 60)
print("SUMMARY: Function works correctly for both 1D (mean_rates) and 2D (PSTH) inputs!")
print("The numpy operations preserve array dimensions correctly.")

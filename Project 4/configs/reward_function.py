"""
Default reward function - imports from organized location.
This maintains backward compatibility with existing imports.

To use a different reward function, update the import below or
copy the desired reward function to configs/rewards/reward_function.py
"""
from configs.rewards.reward_function import reward_function

# Export for backward compatibility
__all__ = ['reward_function']

"""
Optimization module for inventory management
Includes Robust OR, SMDP, DQN, and batch optimization
"""
from .robust_or import RobustInventoryOptimizer

try:
    from .smdp_policy import SMDPPolicy
    from .dqn_rl import DQNAgent
    from .batch_optimization import BatchOptimizer
    
    __all__ = [
        'RobustInventoryOptimizer',
        'SMDPPolicy',
        'DQNAgent',
        'BatchOptimizer'
    ]
except ImportError:
    __all__ = ['RobustInventoryOptimizer']
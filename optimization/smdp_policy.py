"""
Semi-Markov Decision Process (SMDP) policy for inventory control
"""
import numpy as np
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SMDPPolicy:
    """
    SMDP-based inventory policy
    States: inventory levels
    Actions: order quantities
    Rewards: negative costs
    """
    
    def __init__(self, max_inventory=1000, max_order=500):
        self.max_inventory = max_inventory
        self.max_order = max_order
        
        # Discretize state and action spaces
        self.states = np.arange(0, max_inventory + 1, 10)
        self.actions = np.arange(0, max_order + 1, 10)
        
        # Initialize value function and policy
        self.V = np.zeros(len(self.states))
        self.policy = np.zeros(len(self.states), dtype=int)
        
        # Parameters
        self.gamma = 0.95  # Discount factor
        self.holding_cost = 0.5
        self.shortage_cost = 50
        self.order_cost = 100
    
    def get_state_index(self, inventory_level: float) -> int:
        """Convert inventory level to state index"""
        idx = np.abs(self.states - inventory_level).argmin()
        return idx
    
    def transition_probability(self, state: int, action: int, 
                              next_state: int, demand: float) -> float:
        """Calculate transition probability"""
        current_inventory = self.states[state]
        order_qty = self.actions[action]
        next_inventory = self.states[next_state]
        
        # Next inventory = current + order - demand
        expected_next = current_inventory + order_qty - demand
        
        # Probability based on proximity to expected value
        distance = abs(next_inventory - expected_next)
        prob = np.exp(-distance / 10)  # Exponential decay
        
        return prob
    
    def reward(self, state: int, action: int, demand: float) -> float:
        """Calculate reward (negative cost)"""
        current_inventory = self.states[state]
        order_qty = self.actions[action]
        
        # Holding cost
        avg_inventory = (current_inventory + order_qty) / 2
        holding = -self.holding_cost * avg_inventory
        
        # Ordering cost
        ordering = -self.order_cost if order_qty > 0 else 0
        
        # Shortage cost
        net_inventory = current_inventory + order_qty - demand
        shortage = -self.shortage_cost * max(0, -net_inventory)
        
        return holding + ordering + shortage
    
    def value_iteration(self, demand_distribution: np.ndarray, 
                       iterations=100) -> None:
        """Perform value iteration to find optimal policy"""
        for iter in range(iterations):
            V_old = self.V.copy()
            
            for s in range(len(self.states)):
                max_value = float('-inf')
                best_action = 0
                
                for a in range(len(self.actions)):
                    # Expected value for this action
                    expected_value = 0
                    
                    for demand in demand_distribution:
                        # Calculate expected next state
                        next_inventory = (self.states[s] + 
                                        self.actions[a] - demand)
                        next_inventory = np.clip(next_inventory, 0, 
                                                self.max_inventory)
                        next_s = self.get_state_index(next_inventory)
                        
                        # Transition probability (simplified)
                        prob = 1.0 / len(demand_distribution)
                        
                        # Reward
                        r = self.reward(s, a, demand)
                        
                        # Bellman equation
                        expected_value += prob * (r + self.gamma * V_old[next_s])
                    
                    if expected_value > max_value:
                        max_value = expected_value
                        best_action = a
                
                self.V[s] = max_value
                self.policy[s] = best_action
            
            # Check convergence
            if np.max(np.abs(self.V - V_old)) < 0.01:
                logger.info(f"SMDP converged at iteration {iter}")
                break
    
    def get_action(self, inventory_level: float) -> float:
        """Get optimal action for current inventory level"""
        state_idx = self.get_state_index(inventory_level)
        action_idx = self.policy[state_idx]
        return self.actions[action_idx]
"""
Batch optimization for multiple medicines
Handles global constraints and prioritization
"""
import pulp
import numpy as np
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatchOptimizer:
    """
    Optimize orders for multiple medicines simultaneously
    Considers global constraints like budget and storage
    """
    
    def __init__(self, storage_capacity=10000, budget=100000):
        self.storage_capacity = storage_capacity
        self.budget = budget
    
    def optimize(self, medicines: List[Dict], 
                forecasts: Dict[str, np.ndarray]) -> List[Dict]:
        """
        Perform batch optimization
        
        Args:
            medicines: List of medicine data
            forecasts: Forecasts for each medicine
            
        Returns:
            Optimized order quantities
        """
        # Create LP problem
        prob = pulp.LpProblem("BatchInventoryOptimization", pulp.LpMinimize)
        
        # Decision variables
        order_vars = {}
        for med in medicines:
            med_id = med['medicine_id']
            max_order = med.get('max_order_quantity', 1000)
            
            order_vars[med_id] = pulp.LpVariable(
                f"order_{med_id}",
                lowBound=0,
                upBound=max_order,
                cat='Continuous'
            )
        
        # Objective: Minimize total cost
        total_cost = 0
        for med in medicines:
            med_id = med['medicine_id']
            
            # Holding cost
            unit_cost = med.get('unit_cost', 10)
            holding_cost = 0.5 * unit_cost * order_vars[med_id]
            
            # Fixed ordering cost
            # Use binary variable for fixed cost
            is_ordered = pulp.LpVariable(f"is_ordered_{med_id}", cat='Binary')
            prob += order_vars[med_id] <= 10000 * is_ordered  # Big M constraint
            ordering_cost = 100 * is_ordered
            
            total_cost += holding_cost + ordering_cost
        
        prob += total_cost
        
        # Constraint 1: Storage capacity
        total_inventory = sum(
            med.get('current_inventory', 0) + order_vars[med['medicine_id']]
            for med in medicines
        )
        prob += total_inventory <= self.storage_capacity, "StorageCapacity"
        
        # Constraint 2: Budget
        total_spending = sum(
            order_vars[med['medicine_id']] * med.get('unit_cost', 10)
            for med in medicines
        )
        prob += total_spending <= self.budget, "Budget"
        
        # Constraint 3: Service level (meet demand with high probability)
        for med in medicines:
            med_id = med['medicine_id']
            if med_id in forecasts:
                # Ensure inventory + order >= forecasted demand
                forecast_demand = np.mean(forecasts[med_id])
                current_inv = med.get('current_inventory', 0)
                
                prob += (current_inv + order_vars[med_id] >= 
                        forecast_demand * 0.95), f"ServiceLevel_{med_id}"
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Extract results
        results = []
        for med in medicines:
            med_id = med['medicine_id']
            order_qty = order_vars[med_id].varValue
            
            if order_qty is not None and order_qty > 0:
                results.append({
                    'medicine_id': med_id,
                    'current_inventory': med.get('current_inventory', 0),
                    'order_quantity': order_qty,
                    'total_cost': order_qty * med.get('unit_cost', 10),
                    'priority': self._calculate_priority(med, forecasts.get(med_id))
                })
        
        logger.info(f"Batch optimization complete: {len(results)} orders")
        
        return results
    
    def _calculate_priority(self, medicine: Dict, forecast: np.ndarray) -> float:
        """Calculate order priority"""
        current_inv = medicine.get('current_inventory', 0)
        reorder_point = medicine.get('reorder_point', 100)
        
        if forecast is not None:
            demand_rate = np.mean(forecast)
            days_of_supply = current_inv / max(demand_rate, 1)
        else:
            days_of_supply = 30
        
        # Priority based on days of supply
        if days_of_supply < 7:
            return 1.0  # Critical
        elif days_of_supply < 14:
            return 0.7  # High
        elif days_of_supply < 30:
            return 0.4  # Medium
        else:
            return 0.1  # Low
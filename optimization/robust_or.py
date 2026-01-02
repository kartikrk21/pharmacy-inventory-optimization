import pulp
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta
import sys
sys.path.append('..')

from config.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustInventoryOptimizer:
    """
    Robust Operations Research based inventory optimization
    Uses Linear Programming with uncertainty handling
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.optimization_results = {}
        
    def calculate_safety_stock(self, demand_forecast: np.ndarray,
                              demand_uncertainty: np.ndarray,
                              lead_time: int,
                              service_level: float) -> float:
        """
        Calculate safety stock using uncertainty quantification
        
        Safety Stock = Z * σ * √(Lead Time)
        where Z is the service level factor
        """
        # Service level to Z-score mapping
        z_scores = {
            0.90: 1.28,
            0.95: 1.65,
            0.99: 2.33,
            0.995: 2.58,
            0.9995: 3.29
        }
        
        z_score = z_scores.get(service_level, 1.96)
        
        # Average daily demand and standard deviation
        avg_daily_demand = np.mean(demand_forecast[:lead_time])
        demand_std = np.mean(demand_uncertainty[:lead_time])
        
        # Safety stock calculation
        safety_stock = z_score * demand_std * np.sqrt(lead_time)
        
        return max(0, safety_stock)
    
    def calculate_reorder_point(self, demand_forecast: np.ndarray,
                               lead_time: int,
                               safety_stock: float) -> float:
        """
        Calculate reorder point
        ROP = Lead Time Demand + Safety Stock
        """
        lead_time_demand = np.sum(demand_forecast[:lead_time])
        reorder_point = lead_time_demand + safety_stock
        
        return reorder_point
    
    def calculate_economic_order_quantity(self, annual_demand: float,
                                         ordering_cost: float,
                                         holding_cost: float) -> float:
        """
        Calculate Economic Order Quantity (EOQ)
        EOQ = √(2 * D * S / H)
        where D = annual demand, S = ordering cost, H = holding cost
        """
        if annual_demand <= 0 or holding_cost <= 0:
            return 0
        
        eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
        
        return eoq
    
    def optimize_single_medicine(self, medicine_id: str,
                                 current_inventory: float,
                                 demand_forecast: np.ndarray,
                                 demand_uncertainty: np.ndarray,
                                 unit_cost: float,
                                 shelf_life: int) -> Dict:
        """
        Optimize inventory for a single medicine using robust OR
        """
        # Calculate key metrics
        safety_stock = self.calculate_safety_stock(
            demand_forecast,
            demand_uncertainty,
            self.config.LEAD_TIME,
            self.config.SERVICE_LEVEL
        )
        
        reorder_point = self.calculate_reorder_point(
            demand_forecast,
            self.config.LEAD_TIME,
            safety_stock
        )
        
        # Annual demand estimation
        annual_demand = np.sum(demand_forecast) * (365 / len(demand_forecast))
        
        # Economic Order Quantity
        eoq = self.calculate_economic_order_quantity(
            annual_demand,
            self.config.ORDER_COST,
            self.config.HOLDING_COST_PER_UNIT * unit_cost
        )
        
        # Adjust for shelf life constraints
        max_order_by_shelf_life = (
            annual_demand / 365 * shelf_life * 0.8  # 80% of shelf life
        )
        
        optimal_order_qty = min(
            eoq,
            max_order_by_shelf_life,
            self.config.MAX_ORDER_QUANTITY
        )
        
        # Check if reorder is needed
        should_order = current_inventory < reorder_point
        
        if should_order:
            order_quantity = max(
                optimal_order_qty - current_inventory,
                self.config.MIN_ORDER_QUANTITY
            )
        else:
            order_quantity = 0
        
        # Calculate expected costs
        expected_holding_cost = (
            (current_inventory + order_quantity) / 2 * 
            self.config.HOLDING_COST_PER_UNIT * unit_cost * 30
        )
        
        expected_shortage_cost = self._calculate_shortage_cost(
            current_inventory + order_quantity,
            demand_forecast,
            demand_uncertainty
        )
        
        # Waste estimation
        expected_waste = self._calculate_expected_waste(
            current_inventory + order_quantity,
            demand_forecast,
            shelf_life
        )
        
        result = {
            'medicine_id': medicine_id,
            'current_inventory': current_inventory,
            'safety_stock': safety_stock,
            'reorder_point': reorder_point,
            'economic_order_quantity': eoq,
            'optimal_order_quantity': optimal_order_qty,
            'should_order': should_order,
            'order_quantity': order_quantity,
            'expected_holding_cost': expected_holding_cost,
            'expected_shortage_cost': expected_shortage_cost,
            'expected_waste': expected_waste,
            'total_expected_cost': (
                expected_holding_cost + 
                expected_shortage_cost + 
                expected_waste * unit_cost * self.config.WASTE_COST_MULTIPLIER
            ),
            'service_level': self.config.SERVICE_LEVEL,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def _calculate_shortage_cost(self, inventory_level: float,
                                 demand_forecast: np.ndarray,
                                 demand_uncertainty: np.ndarray) -> float:
        """Calculate expected shortage cost"""
        # Simulate demand scenarios
        num_scenarios = 1000
        total_shortage = 0
        
        for _ in range(num_scenarios):
            # Sample from uncertain demand
            scenario_demand = np.random.normal(
                demand_forecast,
                demand_uncertainty
            )
            scenario_demand = np.maximum(scenario_demand, 0)
            
            # Calculate shortage in this scenario
            cumulative_demand = 0
            remaining_inventory = inventory_level
            
            for daily_demand in scenario_demand[:30]:  # 30 days
                cumulative_demand += daily_demand
                if cumulative_demand > remaining_inventory:
                    total_shortage += (cumulative_demand - remaining_inventory)
                    break
        
        expected_shortage = total_shortage / num_scenarios
        expected_shortage_cost = expected_shortage * self.config.SHORTAGE_COST
        
        return expected_shortage_cost
    
    def _calculate_expected_waste(self, inventory_level: float,
                                  demand_forecast: np.ndarray,
                                  shelf_life: int) -> float:
        """Calculate expected waste due to expiration"""
        # Forecast demand over shelf life
        days_to_consider = min(shelf_life, len(demand_forecast))
        total_demand = np.sum(demand_forecast[:days_to_consider])
        
        # Waste is inventory that exceeds demand within shelf life
        expected_waste = max(0, inventory_level - total_demand * 1.1)  # 10% buffer
        
        return expected_waste
    
    def batch_optimize(self, medicines_data: List[Dict],
                      demand_forecasts: Dict[str, np.ndarray],
                      demand_uncertainties: Dict[str, np.ndarray]) -> List[Dict]:
        """
        Optimize inventory for multiple medicines with capacity constraints
        """
        # First, optimize each medicine individually
        individual_results = []
        
        for med in medicines_data:
            medicine_id = med['medicine_id']
            
            if medicine_id not in demand_forecasts:
                continue
            
            result = self.optimize_single_medicine(
                medicine_id,
                med['current_inventory'],
                demand_forecasts[medicine_id],
                demand_uncertainties[medicine_id],
                med['unit_cost'],
                med['shelf_life']
            )
            
            individual_results.append(result)
        
        # Apply global constraints using Linear Programming
        batch_optimized = self._apply_global_constraints(
            individual_results,
            medicines_data
        )
        
        return batch_optimized
    
    def _apply_global_constraints(self, individual_results: List[Dict],
                                 medicines_data: List[Dict]) -> List[Dict]:
        """
        Apply global constraints (storage capacity, budget) using LP
        """
        # Create LP problem
        prob = pulp.LpProblem("InventoryOptimization", pulp.LpMinimize)
        
        # Decision variables: order quantity for each medicine
        order_vars = {}
        for result in individual_results:
            med_id = result['medicine_id']
            order_vars[med_id] = pulp.LpVariable(
                f"order_{med_id}",
                lowBound=0,
                upBound=result['optimal_order_quantity'],
                cat='Continuous'
            )
        
        # Objective: minimize total cost
        total_cost = pulp.lpSum([
            result['expected_holding_cost'] +
            result['expected_shortage_cost'] +
            result['expected_waste'] * order_vars[result['medicine_id']]
            for result in individual_results
        ])
        
        prob += total_cost
        
        # Constraint 1: Storage capacity
        total_inventory = pulp.lpSum([
            result['current_inventory'] + order_vars[result['medicine_id']]
            for result in individual_results
        ])
        prob += total_inventory <= self.config.STORAGE_CAPACITY, "StorageCapacity"
        
        # Constraint 2: Ensure critical medicines are stocked
        for result in individual_results:
            if result['should_order']:
                prob += (
                    order_vars[result['medicine_id']] >= 
                    self.config.MIN_ORDER_QUANTITY
                ), f"MinOrder_{result['medicine_id']}"
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Update results with optimized quantities
        optimized_results = []
        for result in individual_results:
            med_id = result['medicine_id']
            result['optimized_order_quantity'] = order_vars[med_id].varValue
            optimized_results.append(result)
        
        logger.info(f"Batch optimization complete. Status: {pulp.LpStatus[prob.status]}")
        
        return optimized_results
    
    def generate_order_recommendations(self, optimization_results: List[Dict]) -> List[Dict]:
        """Generate actionable order recommendations"""
        recommendations = []
        
        for result in optimization_results:
            if result.get('optimized_order_quantity', 0) > 0:
                recommendation = {
                    'medicine_id': result['medicine_id'],
                    'order_quantity': result['optimized_order_quantity'],
                    'priority': self._calculate_priority(result),
                    'expected_cost_savings': self._calculate_savings(result),
                    'urgency': 'HIGH' if result['current_inventory'] < result['safety_stock'] else 'NORMAL',
                    'rationale': self._generate_rationale(result)
                }
                recommendations.append(recommendation)
        
        # Sort by priority
        recommendations.sort(key=lambda x: x['priority'], reverse=True)
        
        return recommendations
    
    def _calculate_priority(self, result: Dict) -> float:
        """Calculate order priority score"""
        # Factors: shortage risk, cost savings, service level
        shortage_risk = (
            1 - (result['current_inventory'] / 
                 max(result['reorder_point'], 1))
        )
        
        cost_benefit = 1 / (1 + result['total_expected_cost'])
        
        priority = shortage_risk * 0.6 + cost_benefit * 0.4
        
        return max(0, min(1, priority))
    
    def _calculate_savings(self, result: Dict) -> float:
        """Estimate cost savings from optimization"""
        # Compare optimized vs no-action scenario
        baseline_cost = (
            result['expected_shortage_cost'] * 2 +  # Worse shortages
            result['expected_waste'] * 1.5  # More waste
        )
        
        optimized_cost = result['total_expected_cost']
        
        savings = max(0, baseline_cost - optimized_cost)
        
        return savings
    
    def _generate_rationale(self, result: Dict) -> str:
        """Generate human-readable rationale"""
        reasons = []
        
        if result['current_inventory'] < result['safety_stock']:
            reasons.append("Inventory below safety stock")
        
        if result['current_inventory'] < result['reorder_point']:
            reasons.append("Below reorder point")
        
        if result['expected_shortage_cost'] > 100:
            reasons.append("High shortage risk")
        
        if not reasons:
            reasons.append("Optimal replenishment cycle")
        
        return "; ".join(reasons)

# Example usage
if __name__ == "__main__":
    optimizer = RobustInventoryOptimizer()
    
    # Sample data
    medicines = [
        {
            'medicine_id': 'MED001',
            'current_inventory': 50,
            'unit_cost': 10,
            'shelf_life': 365
        }
    ]
    
    demand_forecasts = {
        'MED001': np.array([10, 12, 11, 13, 10, 12] * 5)  # 30 days
    }
    
    demand_uncertainties = {
        'MED001': np.array([2, 2.5, 2, 2.5, 2, 2.5] * 5)
    }
    
    # Optimize
    results = optimizer.batch_optimize(
        medicines,
        demand_forecasts,
        demand_uncertainties
    )
    
    # Generate recommendations
    recommendations = optimizer.generate_order_recommendations(results)
    
    for rec in recommendations:
        print(f"\nRecommendation for {rec['medicine_id']}:")
        print(f"  Order Quantity: {rec['order_quantity']:.0f}")
        print(f"  Priority: {rec['priority']:.2f}")
        print(f"  Urgency: {rec['urgency']}")
        print(f"  Rationale: {rec['rationale']}")
"""
Order Actuator - Executes and validates orders
Handles batch optimization and order generation
"""
import logging
from typing import Dict, List
from datetime import datetime, timedelta
import sys
sys.path.append('..')

from config.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrderActuator:
    """
    Execute and validate inventory orders
    Implements batching optimization and PO generation
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.pending_orders = []
        self.validation_rules = self._load_validation_rules()
    
    def _load_validation_rules(self) -> Dict:
        """Load order validation rules"""
        return {
            'min_order_quantity': self.config.MIN_ORDER_QUANTITY,
            'max_order_quantity': self.config.MAX_ORDER_QUANTITY,
            'min_order_value': 100,  # Minimum order value in currency
            'max_order_value': 100000,
            'lead_time': self.config.LEAD_TIME,
            'allowed_suppliers': ['Supplier_A', 'Supplier_B', 'Supplier_C']
        }
    
    def validate_order(self, order: Dict) -> tuple:
        """
        Validate order against business rules
        
        Args:
            order: Order dictionary
            
        Returns:
            (is_valid, error_message)
        """
        # Check quantity
        if order['quantity'] < self.validation_rules['min_order_quantity']:
            return False, f"Order quantity below minimum: {self.validation_rules['min_order_quantity']}"
        
        if order['quantity'] > self.validation_rules['max_order_quantity']:
            return False, f"Order quantity exceeds maximum: {self.validation_rules['max_order_quantity']}"
        
        # Check order value
        total_value = order.get('quantity', 0) * order.get('unit_cost', 0)
        if total_value < self.validation_rules['min_order_value']:
            return False, f"Order value below minimum: ${self.validation_rules['min_order_value']}"
        
        if total_value > self.validation_rules['max_order_value']:
            return False, f"Order value exceeds maximum: ${self.validation_rules['max_order_value']}"
        
        # Check required fields
        required_fields = ['medicine_id', 'quantity', 'unit_cost']
        for field in required_fields:
            if field not in order:
                return False, f"Missing required field: {field}"
        
        return True, "Order validated successfully"
    
    def create_purchase_order(self, recommendation: Dict) -> Dict:
        """
        Create a purchase order from optimization recommendation
        
        Args:
            recommendation: Order recommendation from optimizer
            
        Returns:
            Purchase order dictionary
        """
        po_id = f"PO{int(datetime.now().timestamp()*1000)}"
        
        purchase_order = {
            'order_id': po_id,
            'medicine_id': recommendation['medicine_id'],
            'quantity': recommendation['order_quantity'],
            'unit_cost': recommendation.get('unit_cost', 0),
            'total_cost': recommendation['order_quantity'] * recommendation.get('unit_cost', 0),
            'priority': recommendation.get('priority', 0.5),
            'urgency': recommendation.get('urgency', 'NORMAL'),
            'status': 'PENDING',
            'expected_delivery_date': (
                datetime.now() + timedelta(days=self.config.LEAD_TIME)
            ).isoformat(),
            'supplier': self._select_supplier(recommendation),
            'rationale': recommendation.get('rationale', ''),
            'created_at': datetime.now().isoformat(),
            'created_by': 'system'
        }
        
        return purchase_order
    
    def _select_supplier(self, recommendation: Dict) -> str:
        """Select supplier based on medicine and urgency"""
        urgency = recommendation.get('urgency', 'NORMAL')
        
        # Simple supplier selection logic
        if urgency == 'HIGH':
            return 'Supplier_A'  # Fastest delivery
        elif urgency == 'NORMAL':
            return 'Supplier_B'  # Best balance
        else:
            return 'Supplier_C'  # Most economical
    
    def batch_orders(self, orders: List[Dict], 
                    batch_size: int = None) -> List[List[Dict]]:
        """
        Batch orders for efficient processing
        
        Args:
            orders: List of orders
            batch_size: Maximum orders per batch
            
        Returns:
            List of order batches
        """
        if batch_size is None:
            batch_size = self.config.BATCH_SIZE
        
        batches = []
        current_batch = []
        
        # Sort by priority
        sorted_orders = sorted(
            orders, 
            key=lambda x: x.get('priority', 0), 
            reverse=True
        )
        
        for order in sorted_orders:
            current_batch.append(order)
            
            if len(current_batch) >= batch_size:
                batches.append(current_batch)
                current_batch = []
        
        # Add remaining orders
        if current_batch:
            batches.append(current_batch)
        
        logger.info(f"Created {len(batches)} batches from {len(orders)} orders")
        
        return batches
    
    def consolidate_orders(self, orders: List[Dict]) -> List[Dict]:
        """
        Consolidate orders for the same medicine
        
        Args:
            orders: List of orders
            
        Returns:
            Consolidated orders
        """
        consolidated = {}
        
        for order in orders:
            medicine_id = order['medicine_id']
            
            if medicine_id in consolidated:
                # Combine quantities
                consolidated[medicine_id]['quantity'] += order['quantity']
                consolidated[medicine_id]['total_cost'] += order.get('total_cost', 0)
                
                # Keep highest priority
                consolidated[medicine_id]['priority'] = max(
                    consolidated[medicine_id].get('priority', 0),
                    order.get('priority', 0)
                )
            else:
                consolidated[medicine_id] = order.copy()
        
        result = list(consolidated.values())
        
        logger.info(f"Consolidated {len(orders)} orders into {len(result)} orders")
        
        return result
    
    def execute_order(self, order: Dict) -> Dict:
        """
        Execute a purchase order
        
        Args:
            order: Purchase order to execute
            
        Returns:
            Execution result
        """
        # Validate order
        is_valid, message = self.validate_order(order)
        
        if not is_valid:
            logger.error(f"Order validation failed: {message}")
            return {
                'success': False,
                'order_id': order.get('order_id'),
                'error': message
            }
        
        # Simulate order placement
        try:
            # In real implementation, this would:
            # 1. Send order to supplier via ERP API
            # 2. Update inventory system
            # 3. Create tracking record
            # 4. Send notifications
            
            logger.info(f"Executing order {order['order_id']} for {order['medicine_id']}")
            
            # Update order status
            order['status'] = 'APPROVED'
            order['approved_at'] = datetime.now().isoformat()
            
            # Add to pending orders
            self.pending_orders.append(order)
            
            return {
                'success': True,
                'order_id': order['order_id'],
                'message': 'Order executed successfully',
                'expected_delivery': order['expected_delivery_date']
            }
            
        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            return {
                'success': False,
                'order_id': order.get('order_id'),
                'error': str(e)
            }
    
    def execute_batch(self, orders: List[Dict]) -> Dict:
        """
        Execute a batch of orders
        
        Args:
            orders: List of orders to execute
            
        Returns:
            Batch execution results
        """
        results = {
            'total': len(orders),
            'successful': 0,
            'failed': 0,
            'details': []
        }
        
        for order in orders:
            result = self.execute_order(order)
            results['details'].append(result)
            
            if result['success']:
                results['successful'] += 1
            else:
                results['failed'] += 1
        
        logger.info(
            f"Batch execution complete: {results['successful']}/{results['total']} successful"
        )
        
        return results
    
    def cancel_order(self, order_id: str) -> Dict:
        """
        Cancel a pending order
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Cancellation result
        """
        for order in self.pending_orders:
            if order['order_id'] == order_id:
                if order['status'] in ['PENDING', 'APPROVED']:
                    order['status'] = 'CANCELLED'
                    order['cancelled_at'] = datetime.now().isoformat()
                    
                    logger.info(f"Order {order_id} cancelled")
                    
                    return {
                        'success': True,
                        'message': f"Order {order_id} cancelled successfully"
                    }
                else:
                    return {
                        'success': False,
                        'message': f"Order {order_id} cannot be cancelled (status: {order['status']})"
                    }
        
        return {
            'success': False,
            'message': f"Order {order_id} not found"
        }
    
    def track_order(self, order_id: str) -> Dict:
        """
        Track order status
        
        Args:
            order_id: Order ID to track
            
        Returns:
            Order tracking information
        """
        for order in self.pending_orders:
            if order['order_id'] == order_id:
                return {
                    'order_id': order_id,
                    'status': order['status'],
                    'medicine_id': order['medicine_id'],
                    'quantity': order['quantity'],
                    'expected_delivery': order.get('expected_delivery_date'),
                    'supplier': order.get('supplier')
                }
        
        return {
            'order_id': order_id,
            'status': 'NOT_FOUND'
        }
    
    def get_pending_orders(self) -> List[Dict]:
        """Get all pending orders"""
        return [
            order for order in self.pending_orders 
            if order['status'] in ['PENDING', 'APPROVED']
        ]
    
    def generate_order_report(self) -> Dict:
        """Generate order execution report"""
        total_orders = len(self.pending_orders)
        
        status_counts = {}
        for order in self.pending_orders:
            status = order['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        total_value = sum(
            order.get('total_cost', 0) 
            for order in self.pending_orders
        )
        
        return {
            'total_orders': total_orders,
            'status_breakdown': status_counts,
            'total_order_value': total_value,
            'generated_at': datetime.now().isoformat()
        }

# Example usage
if __name__ == "__main__":
    actuator = OrderActuator()
    
    # Sample recommendation
    recommendation = {
        'medicine_id': 'MED001',
        'order_quantity': 100,
        'unit_cost': 10.50,
        'priority': 0.8,
        'urgency': 'HIGH',
        'rationale': 'Inventory below safety stock'
    }
    
    # Create and execute order
    po = actuator.create_purchase_order(recommendation)
    print("\nPurchase Order Created:")
    print(f"  Order ID: {po['order_id']}")
    print(f"  Medicine: {po['medicine_id']}")
    print(f"  Quantity: {po['quantity']}")
    print(f"  Total Cost: ${po['total_cost']:.2f}")
    
    # Execute order
    result = actuator.execute_order(po)
    print(f"\nExecution Result: {result['message']}")
    
    # Generate report
    report = actuator.generate_order_report()
    print(f"\nOrder Report:")
    print(f"  Total Orders: {report['total_orders']}")
    print(f"  Total Value: ${report['total_order_value']:.2f}")
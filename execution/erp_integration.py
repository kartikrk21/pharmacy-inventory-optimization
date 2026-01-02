"""
ERP/DI Integration Module
Handles integration with external ERP systems and data interchange
"""
import requests
import json
import logging
from typing import Dict, List
from datetime import datetime
import sys
sys.path.append('..')

from config.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ERPIntegration:
    """
    Integration with Enterprise Resource Planning systems
    Supports order submission, inventory sync, and status tracking
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.api_url = self.config.ERP_API_URL
        self.api_key = self.config.ERP_API_KEY
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        })
    
    def test_connection(self) -> bool:
        """Test ERP API connection"""
        try:
            response = self.session.get(f"{self.api_url}/health")
            if response.status_code == 200:
                logger.info("ERP connection successful")
                return True
            else:
                logger.error(f"ERP connection failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"ERP connection error: {e}")
            return False
    
    def submit_order(self, order: Dict) -> Dict:
        """
        Submit purchase order to ERP system
        
        Args:
            order: Order details
            
        Returns:
            ERP response with order confirmation
        """
        try:
            # Format order for ERP
            erp_order = self._format_order_for_erp(order)
            
            # Submit to ERP
            response = self.session.post(
                f"{self.api_url}/orders",
                json=erp_order,
                timeout=30
            )
            
            if response.status_code in [200, 201]:
                result = response.json()
                logger.info(f"Order submitted to ERP: {result.get('erp_order_id')}")
                
                return {
                    'success': True,
                    'erp_order_id': result.get('erp_order_id'),
                    'order_id': order['order_id'],
                    'status': result.get('status'),
                    'message': 'Order submitted successfully'
                }
            else:
                logger.error(f"ERP order submission failed: {response.text}")
                return {
                    'success': False,
                    'error': response.text,
                    'status_code': response.status_code
                }
                
        except Exception as e:
            logger.error(f"ERP submission error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _format_order_for_erp(self, order: Dict) -> Dict:
        """Format order for ERP API"""
        return {
            'external_order_id': order['order_id'],
            'item_code': order['medicine_id'],
            'quantity': order['quantity'],
            'unit_price': order.get('unit_cost', 0),
            'total_amount': order.get('total_cost', 0),
            'priority': order.get('priority', 'NORMAL'),
            'delivery_date': order.get('expected_delivery_date'),
            'supplier_code': order.get('supplier', ''),
            'notes': order.get('rationale', ''),
            'created_at': order.get('created_at', datetime.now().isoformat())
        }
    
    def get_order_status(self, erp_order_id: str) -> Dict:
        """
        Get order status from ERP
        
        Args:
            erp_order_id: ERP order identifier
            
        Returns:
            Order status information
        """
        try:
            response = self.session.get(
                f"{self.api_url}/orders/{erp_order_id}",
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get order status: {response.status_code}")
                return {'status': 'UNKNOWN'}
                
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def sync_inventory(self, medicine_id: str) -> Dict:
        """
        Sync inventory levels with ERP
        
        Args:
            medicine_id: Medicine identifier
            
        Returns:
            Current inventory from ERP
        """
        try:
            response = self.session.get(
                f"{self.api_url}/inventory/{medicine_id}",
                timeout=30
            )
            
            if response.status_code == 200:
                inventory_data = response.json()
                logger.info(f"Synced inventory for {medicine_id}")
                return {
                    'success': True,
                    'medicine_id': medicine_id,
                    'quantity': inventory_data.get('quantity', 0),
                    'location': inventory_data.get('location'),
                    'last_updated': inventory_data.get('last_updated')
                }
            else:
                return {
                    'success': False,
                    'error': f"Status code: {response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"Inventory sync error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def update_inventory(self, medicine_id: str, quantity_change: float,
                        transaction_type: str) -> Dict:
        """
        Update inventory in ERP
        
        Args:
            medicine_id: Medicine identifier
            quantity_change: Change in quantity (positive or negative)
            transaction_type: Type of transaction
            
        Returns:
            Update result
        """
        try:
            payload = {
                'medicine_id': medicine_id,
                'quantity_change': quantity_change,
                'transaction_type': transaction_type,
                'timestamp': datetime.now().isoformat()
            }
            
            response = self.session.post(
                f"{self.api_url}/inventory/update",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info(f"Updated inventory for {medicine_id}: {quantity_change}")
                return {
                    'success': True,
                    'new_quantity': response.json().get('new_quantity')
                }
            else:
                return {
                    'success': False,
                    'error': response.text
                }
                
        except Exception as e:
            logger.error(f"Inventory update error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def batch_sync_inventory(self, medicine_ids: List[str]) -> Dict:
        """
        Sync inventory for multiple medicines
        
        Args:
            medicine_ids: List of medicine identifiers
            
        Returns:
            Batch sync results
        """
        results = {
            'total': len(medicine_ids),
            'successful': 0,
            'failed': 0,
            'details': []
        }
        
        for med_id in medicine_ids:
            result = self.sync_inventory(med_id)
            results['details'].append(result)
            
            if result.get('success'):
                results['successful'] += 1
            else:
                results['failed'] += 1
        
        logger.info(
            f"Batch inventory sync: {results['successful']}/{results['total']} successful"
        )
        
        return results

class DIIntegration:
    """
    Data Interchange Integration
    Handles EDI (Electronic Data Interchange) and other data formats
    """
    
    def __init__(self):
        self.supported_formats = ['JSON', 'XML', 'CSV', 'EDI']
    
    def export_order_to_edi(self, order: Dict) -> str:
        """
        Export order to EDI format
        
        Args:
            order: Order details
            
        Returns:
            EDI formatted string
        """
        # Simplified EDI 850 Purchase Order format
        edi = f"""ISA*00*          *00*          *ZZ*SENDER         *ZZ*RECEIVER       *{datetime.now().strftime('%y%m%d')}*{datetime.now().strftime('%H%M')}*U*00401*000000001*0*P*>~
GS*PO*SENDER*RECEIVER*{datetime.now().strftime('%Y%m%d')}*{datetime.now().strftime('%H%M')}*1*X*004010~
ST*850*0001~
BEG*00*SA*{order['order_id']}**{datetime.now().strftime('%Y%m%d')}~
REF*DP*{order.get('department', 'PHARMACY')}~
PER*BD*{order.get('contact_name', 'Pharmacy Manager')}*TE*{order.get('contact_phone', '000-000-0000')}~
PO1*1*{order['quantity']}*EA*{order.get('unit_cost', 0)}**VP*{order['medicine_id']}~
CTT*1~
SE*7*0001~
GE*1*1~
IEA*1*000000001~"""
        
        return edi
    
    def import_order_from_edi(self, edi_string: str) -> Dict:
        """
        Import order from EDI format
        
        Args:
            edi_string: EDI formatted string
            
        Returns:
            Order dictionary
        """
        # Simplified EDI parser
        lines = edi_string.split('~')
        order = {}
        
        for line in lines:
            if line.startswith('BEG'):
                parts = line.split('*')
                order['order_id'] = parts[3] if len(parts) > 3 else ''
            elif line.startswith('PO1'):
                parts = line.split('*')
                order['quantity'] = float(parts[2]) if len(parts) > 2 else 0
                order['unit_cost'] = float(parts[4]) if len(parts) > 4 else 0
                order['medicine_id'] = parts[7] if len(parts) > 7 else ''
        
        return order
    
    def export_to_json(self, data: Dict, filepath: str = None) -> str:
        """Export data to JSON"""
        json_str = json.dumps(data, indent=2)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
            logger.info(f"Data exported to {filepath}")
        
        return json_str
    
    def import_from_json(self, filepath: str) -> Dict:
        """Import data from JSON"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.info(f"Data imported from {filepath}")
        return data

# Mock ERP server for testing
class MockERPServer:
    """Mock ERP server for testing without real ERP connection"""
    
    def __init__(self):
        self.orders = {}
        self.inventory = {}
    
    def submit_order(self, order: Dict) -> Dict:
        """Mock order submission"""
        erp_order_id = f"ERP{len(self.orders):06d}"
        self.orders[erp_order_id] = order
        
        return {
            'erp_order_id': erp_order_id,
            'status': 'APPROVED',
            'message': 'Order received'
        }
    
    def get_order_status(self, erp_order_id: str) -> Dict:
        """Mock get order status"""
        if erp_order_id in self.orders:
            return {
                'erp_order_id': erp_order_id,
                'status': 'IN_TRANSIT',
                'estimated_delivery': '2024-12-31'
            }
        return {'status': 'NOT_FOUND'}
    
    def get_inventory(self, medicine_id: str) -> Dict:
        """Mock get inventory"""
        return {
            'medicine_id': medicine_id,
            'quantity': self.inventory.get(medicine_id, 0),
            'location': 'Warehouse_A',
            'last_updated': datetime.now().isoformat()
        }

# Example usage
if __name__ == "__main__":
    # Use mock ERP for testing
    mock_erp = MockERPServer()
    
    # Create sample order
    order = {
        'order_id': 'PO12345',
        'medicine_id': 'MED001',
        'quantity': 100,
        'unit_cost': 10.50,
        'total_cost': 1050.00,
        'priority': 'HIGH'
    }
    
    # Test EDI export
    di = DIIntegration()
    edi_data = di.export_order_to_edi(order)
    print("\nEDI Export:")
    print(edi_data[:200] + "...")
    
    # Test JSON export
    json_data = di.export_to_json(order)
    print("\nJSON Export:")
    print(json_data)
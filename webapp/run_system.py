#!/usr/bin/env python3
"""
Complete System Integration Script
Connects all components: Data Generation -> Model Training -> Web App
"""
import os
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_file_exists(filepath, description):
    """Check if required file exists"""
    if not os.path.exists(filepath):
        logger.error(f"{description} not found: {filepath}")
        return False
    logger.info(f"✓ Found {description}: {filepath}")
    return True

def verify_csv_data(csv_path):
    """Verify CSV data is valid"""
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        required_cols = ['prescription_id', 'medicine_id', 'medicine_name', 
                        'category', 'quantity', 'timestamp']
        
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            logger.error(f"CSV missing columns: {missing}")
            return False
        
        logger.info(f"✓ CSV valid: {len(df)} rows, {len(df['medicine_id'].unique())} medicines")
        return True
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        return False

def setup_directories():
    """Create required directories"""
    dirs = ['templates', 'static', 'static/css', 'static/js', 'trained_models']
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Directory ready: {d}")

def generate_data_if_missing():
    """Generate historical data if CSV doesn't exist"""
    csv_path = 'historical_prescriptions.csv'
    
    if os.path.exists(csv_path):
        logger.info(f"✓ Found existing data: {csv_path}")
        return True
    
    logger.info("Generating historical prescription data...")
    try:
        from prescription_generator import PrescriptionDataGenerator
        
        generator = PrescriptionDataGenerator(num_medicines=500)
        generator.save_historical_data(csv_path, num_days=365)
        
        logger.info(f"✓ Generated data: {csv_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to generate data: {e}")
        return False

def verify_templates():
    """Check if HTML templates exist"""
    templates = ['dashboard.html', 'analytics.html']
    missing = []
    
    for template in templates:
        path = f'templates/{template}'
        if not os.path.exists(path):
            missing.append(template)
    
    if missing:
        logger.warning(f"Missing templates: {missing}")
        logger.info("Templates will be created automatically")
    else:
        logger.info("✓ All templates found")
    
    return True

def create_index_html():
    """Create simple index.html if missing"""
    index_path = 'templates/index.html'
    if os.path.exists(index_path):
        return
    
    html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pharmacy Inventory System</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .container {
            text-align: center;
            background: white;
            padding: 50px;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            color: #333;
            margin-bottom: 30px;
        }
        .btn {
            display: inline-block;
            padding: 15px 40px;
            margin: 10px;
            background: #667eea;
            color: white;
            text-decoration: none;
            border-radius: 10px;
            font-weight: bold;
            transition: transform 0.2s;
        }
        .btn:hover {
            transform: scale(1.05);
            background: #5568d3;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏥 Pharmacy Inventory Optimization System</h1>
        <p>Real-time prescription tracking and intelligent inventory management</p>
        <div style="margin-top: 30px;">
            <a href="/dashboard" class="btn">📊 Open Dashboard</a>
            <a href="/analytics" class="btn">📈 View Analytics</a>
        </div>
    </div>
</body>
</html>
'''
    
    with open(index_path, 'w') as f:
        f.write(html)
    logger.info("✓ Created index.html")

def create_basic_css():
    """Create basic CSS if missing"""
    css_path = 'static/css/style.css'
    if os.path.exists(css_path):
        return
    
    css = '''* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: #f3f4f6;
    color: #1f2937;
}

.dashboard-container {
    padding: 20px;
    max-width: 1600px;
    margin: 0 auto;
}

.dashboard-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 30px;
    border-radius: 15px;
    margin-bottom: 30px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.header-controls {
    display: flex;
    gap: 10px;
}

.btn {
    padding: 10px 20px;
    border: none;
    border-radius: 8px;
    font-weight: bold;
    cursor: pointer;
    transition: all 0.2s;
    text-decoration: none;
    display: inline-block;
}

.btn-primary { background: #3b82f6; color: white; }
.btn-success { background: #10b981; color: white; }
.btn-danger { background: #ef4444; color: white; }
.btn-secondary { background: #6b7280; color: white; }

.btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}

.btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

.metrics-section {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
    margin-bottom: 30px;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.metric-card h3 {
    font-size: 14px;
    color: #6b7280;
    margin-bottom: 10px;
}

.metric-value {
    font-size: 28px;
    font-weight: bold;
    color: #1f2937;
    margin-bottom: 5px;
}

.metric-target {
    font-size: 12px;
    color: #9ca3af;
}

.content-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
    gap: 20px;
    margin-bottom: 30px;
}

.panel {
    background: white;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.panel h2 {
    margin-bottom: 20px;
    color: #1f2937;
}

table {
    width: 100%;
    border-collapse: collapse;
}

th, td {
    padding: 12px;
    text-align: left;
    border-bottom: 1px solid #e5e7eb;
}

th {
    background: #f9fafb;
    font-weight: bold;
    color: #374151;
}

.status-badge {
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: bold;
}

.status-normal { background: #d1fae5; color: #065f46; }
.status-high { background: #fef3c7; color: #92400e; }
.status-low { background: #fee2e2; color: #991b1b; }

.loading {
    border: 4px solid #f3f4f6;
    border-top: 4px solid #3b82f6;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    animation: spin 1s linear infinite;
    margin: 20px auto;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
'''
    
    with open(css_path, 'w') as f:
        f.write(css)
    logger.info("✓ Created style.css")

def run_checks():
    """Run all system checks"""
    logger.info("="*60)
    logger.info("PHARMACY INVENTORY SYSTEM - INITIALIZATION")
    logger.info("="*60)
    
    # Check Python version
    py_version = sys.version_info
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 8):
        logger.error(f"Python 3.8+ required, found {py_version.major}.{py_version.minor}")
        return False
    logger.info(f"✓ Python version: {py_version.major}.{py_version.minor}.{py_version.micro}")
    
    # Setup directories
    setup_directories()
    
    # Generate data if missing
    if not generate_data_if_missing():
        logger.error("Failed to generate or find prescription data")
        return False
    
    # Verify CSV
    if not verify_csv_data('historical_prescriptions.csv'):
        return False
    
    # Create basic templates
    create_index_html()
    create_basic_css()
    
    # Verify templates
    verify_templates()
    
    # Check if app.py exists
    if not check_file_exists('app.py', 'Flask application'):
        return False
    
    logger.info("="*60)
    logger.info("✅ ALL CHECKS PASSED - SYSTEM READY")
    logger.info("="*60)
    return True

def check_ml_files():
    """Check if ML model files exist"""
    ml_files = [
        'demand_forecasting.py',
        'arima_lstm_model.py',
        'uncertainty_quantification.py',
        'ml_integration.py'
    ]
    
    missing = []
    for f in ml_files:
        if not os.path.exists(f):
            missing.append(f)
    
    if missing:
        logger.warning(f"⚠️  Missing ML files: {missing}")
        logger.warning("ML forecasting will use fallback methods")
    else:
        logger.info("✅ All ML model files found")
    
    return len(missing) == 0

def start_application():
    """Start the Flask application"""
    logger.info("\n🚀 Starting application...")
    logger.info("📊 Dashboard: http://localhost:5001/dashboard")
    logger.info("📈 Analytics: http://localhost:5001/analytics")
    logger.info("🤖 ML Models: Training top 20 medicines on startup...")
    logger.info("\n⏳ This may take 1-2 minutes on first run...")
    logger.info("Press CTRL+C to stop the server\n")
    
    try:
        import app
        # App will start via socketio.run() in app.py
        # ML models are trained automatically in app.py
    except KeyboardInterrupt:
        logger.info("\n👋 Application stopped")
    except Exception as e:
        logger.error(f"Error starting application: {e}")
        raise

if __name__ == "__main__":
    try:
        if run_checks():
            start_application()
        else:
            logger.error("❌ System checks failed. Please fix errors and try again.")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
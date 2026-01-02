// Dashboard JavaScript - FIXED for Socket.IO live updates
const socket = io();

// State
let streamActive = false;
let prescriptionCount = 0;
let streamStartTime = null;

// Charts
let streamChart, latencyChart, throughputChart;

// Initialize on load
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Dashboard initializing...');
    initializeCharts();
    loadInitialData();
    setupEventListeners();
    setupSocketListeners();
});

// ============================================================
// SOCKET.IO LISTENERS - CRITICAL FIX
// ============================================================

function setupSocketListeners() {
    console.log('📡 Setting up Socket.IO listeners...');
    
    socket.on('connect', () => {
        console.log('✅ Socket.IO connected!');
        showNotification('Connected to server', 'success');
        
        // Request initial state
        socket.emit('request_update');
    });

    socket.on('disconnect', () => {
        console.log('⚠️  Socket.IO disconnected');
        showNotification('Disconnected from server', 'error');
    });

    socket.on('connection_status', (data) => {
        console.log('📊 Connection status:', data);
    });

    // ============================================================
    // MAIN STATE UPDATE HANDLER - This is where live updates happen
    // ============================================================
    socket.on('state_update', (state) => {
        console.log('📡 State update received:', state);
        
        // Update prescription count
        if (state.total_prescriptions !== undefined) {
            prescriptionCount = state.total_prescriptions;
            const totalEl = document.getElementById('totalPrescriptions');
            if (totalEl) {
                totalEl.textContent = prescriptionCount;
            }
        }
        
        // Update stream rate
        if (state.stream_rate !== undefined) {
            const rateEl = document.getElementById('streamRate');
            if (rateEl) {
                rateEl.textContent = state.stream_rate.toFixed(2) + '/sec';
            }
        }
        
        // Update metrics
        updateMetricsFromState(state);
        
        // Update inventory table if data present
        if (state.inventory && Object.keys(state.inventory).length > 0) {
            updateInventoryFromState(state.inventory);
        }
        
        // Update charts
        if (prescriptionCount % 10 === 0) {
            updateStreamChart(prescriptionCount);
        }
    });

    socket.on('metrics_update', (data) => {
        console.log('📊 Metrics update:', data);
        
        // Update latency chart
        if (latencyChart && data.latency) {
            addDataToChart(latencyChart, data.latency);
        }
        
        // Update throughput chart
        if (throughputChart && data.throughput) {
            addDataToChart(throughputChart, data.throughput);
        }
    });

    socket.on('new_order', async (data) => {
        console.log('📦 New order:', data);
        showNotification(`New order: ${data.medicine_id}`, 'info');
        await loadInitialData();
    });

    // Connection error handling
    socket.on('connect_error', (error) => {
        console.error('❌ Socket.IO connection error:', error);
        showNotification('Connection error', 'error');
    });

    console.log('✅ Socket.IO listeners configured');
}

// ============================================================
// UPDATE FUNCTIONS
// ============================================================

function updateMetricsFromState(state) {
    const fillRateEl = document.getElementById('fillRate');
    const wasteEl = document.getElementById('wastePercent');
    const costReductionEl = document.getElementById('costReduction');
    const latencyEl = document.getElementById('latency');
    const throughputEl = document.getElementById('throughput');
    const availabilityEl = document.getElementById('availability');

    if (fillRateEl && state.fill_rate) {
        fillRateEl.textContent = (state.fill_rate * 100).toFixed(2) + '%';
    }
    if (wasteEl && state.waste_percentage) {
        wasteEl.textContent = (state.waste_percentage * 100).toFixed(0) + '%';
    }
    if (costReductionEl && state.cost_reduction) {
        costReductionEl.textContent = (state.cost_reduction * 100).toFixed(0) + '%';
    }
    if (latencyEl && state.avg_latency_ms) {
        latencyEl.textContent = Math.round(state.avg_latency_ms) + 'ms';
    }
    if (throughputEl && state.throughput) {
        throughputEl.textContent = (state.throughput / 1000).toFixed(1) + 'k/sec';
    }
    if (availabilityEl && state.availability) {
        availabilityEl.textContent = (state.availability * 100).toFixed(2) + '%';
    }
}

function updateInventoryFromState(inventory) {
    console.log('📦 Updating inventory from state:', inventory);
    
    const container = document.getElementById('inventoryTable');
    if (!container) return;
    
    let html = `
        <table>
            <thead>
                <tr>
                    <th>Medicine ID</th>
                    <th>Total Quantity</th>
                    <th>Count</th>
                    <th>Avg per Rx</th>
                </tr>
            </thead>
            <tbody>
    `;
    
    // Convert inventory object to array and sort by total
    const inventoryArray = Object.entries(inventory)
        .map(([id, data]) => ({
            medicine_id: id,
            total: data.total || 0,
            count: data.count || 0,
            avg: data.count > 0 ? (data.total / data.count).toFixed(2) : 0
        }))
        .sort((a, b) => b.total - a.total)
        .slice(0, 10);
    
    inventoryArray.forEach(item => {
        html += `
            <tr>
                <td>${item.medicine_id}</td>
                <td>${item.total}</td>
                <td>${item.count}</td>
                <td>${item.avg}</td>
            </tr>
        `;
    });
    
    html += '</tbody></table>';
    container.innerHTML = html;
}

// ============================================================
// CHARTS
// ============================================================

function initializeCharts() {
    console.log('📈 Initializing charts...');
    
    // Stream Chart
    const streamCtx = document.getElementById('streamChart');
    if (streamCtx) {
        streamChart = new Chart(streamCtx.getContext('2d'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Total Prescriptions',
                    data: [],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }

    // Latency Chart
    const latencyCtx = document.getElementById('latencyChart');
    if (latencyCtx) {
        latencyChart = new Chart(latencyCtx.getContext('2d'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Latency (ms)',
                    data: [],
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 400
                    }
                }
            }
        });
    }

    // Throughput Chart
    const throughputCtx = document.getElementById('throughputChart');
    if (throughputCtx) {
        throughputChart = new Chart(throughputCtx.getContext('2d'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Throughput (req/sec)',
                    data: [],
                    borderColor: '#8b5cf6',
                    backgroundColor: 'rgba(139, 92, 246, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }
    
    console.log('✅ Charts initialized');
}

function updateStreamChart(value) {
    if (!streamChart) return;
    
    const now = new Date().toLocaleTimeString();
    
    if (streamChart.data.labels.length > 20) {
        streamChart.data.labels.shift();
        streamChart.data.datasets[0].data.shift();
    }
    
    streamChart.data.labels.push(now);
    streamChart.data.datasets[0].data.push(value);
    streamChart.update('none'); // 'none' for better performance
}

function addDataToChart(chart, value) {
    if (!chart) return;
    
    const now = new Date().toLocaleTimeString();
    
    if (chart.data.labels.length > 30) {
        chart.data.labels.shift();
        chart.data.datasets[0].data.shift();
    }
    
    chart.data.labels.push(now);
    chart.data.datasets.forEach((dataset) => {
        dataset.data.push(value);
    });
    
    chart.update('none');
}

// ============================================================
// LOAD INITIAL DATA
// ============================================================

async function loadInitialData() {
    try {
        console.log('📥 Loading initial data...');
        
        // Load medicines
        const medicinesResponse = await fetch('/api/medicines');
        const medicinesJson = await medicinesResponse.json();
        if (medicinesJson && medicinesJson.success && medicinesJson.data) {
            displayInventory(medicinesJson.data);
        }

        // Load alerts
        const alertsResponse = await fetch('/api/alerts');
        const alertsJson = await alertsResponse.json();
        if (alertsJson && alertsJson.success && alertsJson.data) {
            displayAlerts(alertsJson.data);
        }
        
        console.log('✅ Initial data loaded');
    } catch (error) {
        console.error('❌ Error loading initial data:', error);
    }
}

function displayInventory(medicines) {
    const container = document.getElementById('inventoryTable');
    if (!container) return;
    
    let html = `
        <table>
            <thead>
                <tr>
                    <th>Medicine</th>
                    <th>Category</th>
                    <th>Current Stock</th>
                    <th>Reorder Point</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
    `;
    
    medicines.slice(0, 10).forEach(med => {
        const stockRatio = med.current_inventory / (med.reorder_point || 100);
        let status, statusClass;
        
        if (stockRatio < 0.5) {
            status = 'Low';
            statusClass = 'status-low';
        } else if (stockRatio < 1) {
            status = 'Medium';
            statusClass = 'status-high';
        } else {
            status = 'Good';
            statusClass = 'status-normal';
        }
        
        html += `
            <tr>
                <td>${med.medicine_name || med.medicine_id}</td>
                <td>${med.category || 'N/A'}</td>
                <td>${Math.round(med.current_inventory || 0)}</td>
                <td>${Math.round(med.reorder_point || 0)}</td>
                <td><span class="status-badge ${statusClass}">${status}</span></td>
            </tr>
        `;
    });
    
    html += '</tbody></table>';
    container.innerHTML = html;
}

function displayAlerts(alerts) {
    const container = document.getElementById('alertsList');
    if (!container) return;
    
    if (alerts.length === 0) {
        container.innerHTML = '<p style="color: #6b7280;">No active alerts</p>';
        return;
    }
    
    let html = '';
    alerts.slice(0, 10).forEach(alert => {
        const severityClass = (alert.severity || 'info').toLowerCase();
        html += `
            <div class="alert-item ${severityClass}">
                <div class="alert-header">${alert.alert_type}: ${alert.medicine_id}</div>
                <div>${alert.message}</div>
                <div class="alert-time">${new Date(alert.created_at).toLocaleString()}</div>
            </div>
        `;
    });
    
    container.innerHTML = html;
}

function updateMetrics(stats) {
    updateMetricsFromState(stats);
}

// ============================================================
// EVENT LISTENERS
// ============================================================

function setupEventListeners() {
    console.log('🎛️  Setting up event listeners...');
    
    // Start Streaming
    const startBtn = document.getElementById('startStreamBtn');
    if (startBtn) {
        startBtn.addEventListener('click', async () => {
            try {
                console.log('▶️  Starting streaming...');
                
                const response = await fetch('/api/streaming/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ rate_per_second: 10 })
                });
                
                const result = await response.json();
                
                if (result && result.success) {
                    streamActive = true;
                    prescriptionCount = 0;
                    streamStartTime = Date.now();
                    
                    startBtn.disabled = true;
                    const stopBtn = document.getElementById('stopStreamBtn');
                    if (stopBtn) stopBtn.disabled = false;
                    
                    showNotification('Streaming started! Watch the numbers update live.', 'success');
                    console.log('✅ Streaming started');
                } else {
                    showNotification(result.error || 'Failed to start streaming', 'error');
                }
            } catch (error) {
                console.error('❌ Error starting stream:', error);
                showNotification('Failed to start streaming', 'error');
            }
        });
    }

    // Stop Streaming
    const stopBtn = document.getElementById('stopStreamBtn');
    if (stopBtn) {
        stopBtn.addEventListener('click', async () => {
            try {
                console.log('⏹️  Stopping streaming...');
                
                const response = await fetch('/api/streaming/stop', {
                    method: 'POST'
                });
                
                const result = await response.json();
                
                if (result && result.success) {
                    streamActive = false;
                    
                    const startBtn = document.getElementById('startStreamBtn');
                    if (startBtn) startBtn.disabled = false;
                    stopBtn.disabled = true;
                    
                    showNotification('Streaming stopped', 'info');
                    console.log('✅ Streaming stopped');
                } else {
                    showNotification(result.error || 'Failed to stop streaming', 'error');
                }
            } catch (error) {
                console.error('❌ Error stopping stream:', error);
                showNotification('Failed to stop streaming', 'error');
            }
        });
    }

    // Run Optimization
    const optimizeBtn = document.getElementById('optimizeBtn');
    if (optimizeBtn) {
        optimizeBtn.addEventListener('click', async () => {
            try {
                showNotification('Running optimization...', 'info');
                
                const medicinesResponse = await fetch('/api/medicines');
                const medicinesJson = await medicinesResponse.json();
                
                if (!medicinesJson || !medicinesJson.success) {
                    throw new Error('Failed to fetch medicines');
                }
                
                const medicines = medicinesJson.data || [];
                const medicineIds = medicines.map(m => m.medicine_id).slice(0, 20);
                
                const response = await fetch('/api/optimize', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ medicine_ids: medicineIds })
                });
                
                const result = await response.json();
                
                if (result && result.success) {
                    displayRecommendations(result.recommendations || []);
                    showNotification(
                        `Optimization complete! ${result.optimized_medicines || 0} recommendations generated`,
                        'success'
                    );
                    await loadInitialData();
                } else {
                    showNotification(result.error || 'Optimization failed', 'error');
                }
            } catch (error) {
                console.error('❌ Error running optimization:', error);
                showNotification('Optimization failed', 'error');
            }
        });
    }
    
    console.log('✅ Event listeners configured');
}

function displayRecommendations(recommendations) {
    const container = document.getElementById('recommendationsTable');
    if (!container) return;
    
    if (recommendations.length === 0) {
        container.innerHTML = '<p style="color: #6b7280;">No recommendations</p>';
        return;
    }
    
    let html = `
        <table>
            <thead>
                <tr>
                    <th>Medicine</th>
                    <th>Order Qty</th>
                    <th>Priority</th>
                    <th>Urgency</th>
                    <th>Action</th>
                </tr>
            </thead>
            <tbody>
    `;
    
    recommendations.forEach(rec => {
        html += `
            <tr>
                <td>${rec.medicine_id}</td>
                <td>${Math.round(rec.order_quantity || 0)}</td>
                <td>${((rec.priority || 0) * 100).toFixed(0)}%</td>
                <td><span class="status-badge status-${(rec.urgency || 'normal').toLowerCase()}">${rec.urgency || 'NORMAL'}</span></td>
                <td>
                    <button class="btn btn-primary" onclick="placeOrder('${rec.medicine_id}', ${rec.order_quantity || 0})">
                        Order
                    </button>
                </td>
            </tr>
        `;
    });
    
    html += '</tbody></table>';
    container.innerHTML = html;
}

async function placeOrder(medicineId, quantity) {
    try {
        const response = await fetch('/api/orders/place', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                medicine_id: medicineId,
                quantity: quantity
            })
        });
        
        const result = await response.json();
        
        if (result && result.success) {
            showNotification(`Order placed for ${medicineId}`, 'success');
            await loadInitialData();
        } else {
            showNotification(result.error || 'Failed to place order', 'error');
        }
    } catch (error) {
        console.error('❌ Error placing order:', error);
        showNotification('Failed to place order', 'error');
    }
}

function showNotification(message, type) {
    console.log(`📢 Notification: ${message} (${type})`);
    
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#3b82f6'};
        color: white;
        border-radius: 8px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.2);
        z-index: 1000;
        animation: slideIn 0.3s ease;
    `;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// ============================================================
// INITIALIZATION COMPLETE
// ============================================================

console.log('✅ Dashboard JavaScript loaded and initialized');
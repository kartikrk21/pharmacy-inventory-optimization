// analytics.js - COMPLETE WORKING VERSION
let forecastChart = null;

document.addEventListener('DOMContentLoaded', async function() {
    console.log('📊 Analytics page loaded');
    await loadMedicines();
});

async function loadMedicines() {
    try {
        console.log('Loading medicines...');
        const response = await fetch('/api/medicines');
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const result = await response.json();
        console.log('Response:', result);
        
        if (!result.success || !result.data) {
            throw new Error('Invalid response');
        }
        
        const medicines = result.data;
        const select = document.getElementById('medicineSelect');
        
        select.innerHTML = '<option value="">-- Select Medicine --</option>';
        
        medicines.slice(0, 50).forEach(med => {
            const option = document.createElement('option');
            option.value = med.medicine_id;
            option.textContent = `${med.medicine_name || med.medicine_id} (${med.category || 'N/A'})`;
            select.appendChild(option);
        });
        
        console.log(`✅ Loaded ${medicines.length} medicines`);
    } catch (error) {
        console.error('❌ Error loading medicines:', error);
        showError('Failed to load medicines: ' + error.message);
    }
}

document.getElementById('loadAnalyticsBtn').addEventListener('click', async function() {
    const medicineId = document.getElementById('medicineSelect').value;
    
    if (!medicineId) {
        alert('Please select a medicine');
        return;
    }

    console.log(`Loading analytics for ${medicineId}...`);
    
    const loading = document.getElementById('forecastLoading');
    const canvas = document.getElementById('forecastChart');
    
    loading.style.display = 'block';
    canvas.style.display = 'none';

    try {
        console.log(`Fetching: /api/forecast/${medicineId}`);
        const response = await fetch(`/api/forecast/${medicineId}`);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        console.log('Forecast data:', data);

        if (data.error) {
            throw new Error(data.error);
        }

        if (!data.forecast || !Array.isArray(data.forecast)) {
            throw new Error('Invalid forecast data');
        }

        loading.style.display = 'none';
        canvas.style.display = 'block';

        displayForecastChart(data);
        displayForecastStats(data);
        displayModelInfo(data);
        
        console.log('✅ Analytics loaded successfully');

    } catch (error) {
        console.error('❌ Error:', error);
        loading.style.display = 'none';
        showError('Failed to load analytics: ' + error.message);
    }
});

function displayForecastChart(data) {
    console.log('Displaying chart...');
    
    const ctx = document.getElementById('forecastChart').getContext('2d');

    const dates = [];
    const today = new Date();
    for (let i = 1; i <= data.horizon_days; i++) {
        const date = new Date(today);
        date.setDate(date.getDate() + i);
        dates.push(date.toLocaleDateString());
    }

    if (forecastChart) {
        forecastChart.destroy();
    }

    forecastChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [
                {
                    label: 'Forecast',
                    data: data.forecast,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.4,
                    borderWidth: 3,
                    fill: true
                },
                {
                    label: 'Upper Bound (95%)',
                    data: data.upper_bound,
                    borderColor: '#ef4444',
                    backgroundColor: 'transparent',
                    borderDash: [5, 5],
                    tension: 0.4,
                    fill: false,
                    borderWidth: 2
                },
                {
                    label: 'Lower Bound (95%)',
                    data: data.lower_bound,
                    borderColor: '#10b981',
                    backgroundColor: 'transparent',
                    borderDash: [5, 5],
                    tension: 0.4,
                    fill: false,
                    borderWidth: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: `Demand Forecast - ${data.medicine_id} (${data.model})`,
                    font: { size: 16, weight: 'bold' }
                },
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Quantity'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Date'
                    }
                }
            }
        }
    });
    
    console.log('✅ Chart displayed');
}

function displayForecastStats(data) {
    const avgForecast = data.forecast.reduce((a, b) => a + b, 0) / data.forecast.length;
    const totalForecast = data.forecast.reduce((a, b) => a + b, 0);
    const avgUncertainty = data.uncertainty.reduce((a, b) => a + b, 0) / data.uncertainty.length;

    const html = `
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 20px;">
            <div style="padding: 15px; background: #f3f4f6; border-radius: 8px;">
                <h4 style="margin: 0 0 5px 0; color: #6b7280; font-size: 14px;">Avg Daily Forecast</h4>
                <p style="font-size: 24px; font-weight: bold; color: #1f2937; margin: 0;">${avgForecast.toFixed(1)}</p>
            </div>
            <div style="padding: 15px; background: #f3f4f6; border-radius: 8px;">
                <h4 style="margin: 0 0 5px 0; color: #6b7280; font-size: 14px;">Total 30-Day Demand</h4>
                <p style="font-size: 24px; font-weight: bold; color: #1f2937; margin: 0;">${totalForecast.toFixed(0)}</p>
            </div>
            <div style="padding: 15px; background: #f3f4f6; border-radius: 8px;">
                <h4 style="margin: 0 0 5px 0; color: #6b7280; font-size: 14px;">Avg Uncertainty</h4>
                <p style="font-size: 24px; font-weight: bold; color: #1f2937; margin: 0;">±${avgUncertainty.toFixed(1)}</p>
            </div>
            <div style="padding: 15px; background: #f3f4f6; border-radius: 8px;">
                <h4 style="margin: 0 0 5px 0; color: #6b7280; font-size: 14px;">Confidence</h4>
                <p style="font-size: 24px; font-weight: bold; color: #1f2937; margin: 0;">${(data.confidence * 100).toFixed(0)}%</p>
            </div>
        </div>
    `;

    document.getElementById('forecastStats').innerHTML = html;
}

function displayModelInfo(data) {
    const html = `
        <div style="padding: 20px;">
            <h3 style="margin-top: 0;">Model: ${data.model}</h3>
            <p><strong>Medicine ID:</strong> ${data.medicine_id}</p>
            <p><strong>Forecast Horizon:</strong> ${data.horizon_days} days</p>
            <p><strong>Confidence Level:</strong> ${(data.confidence * 100).toFixed(0)}%</p>
            <p style="color: #6b7280; font-size: 14px; margin-top: 15px;">
                This forecast uses advanced ML models (ARIMA, LSTM, Prophet) to predict future demand with 95% confidence intervals.
            </p>
        </div>
    `;
    document.getElementById('modelInfo').innerHTML = html;
}

function showError(message) {
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        background: #ef4444;
        color: white;
        border-radius: 8px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.2);
        z-index: 1000;
        max-width: 400px;
    `;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    setTimeout(() => notification.remove(), 5000);
}

console.log('✅ Analytics JavaScript loaded');
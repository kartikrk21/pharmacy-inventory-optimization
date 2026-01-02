// ==============================================================
// webapp/static/js/charts.js
// ==============================================================
/**
 * Chart utilities and configurations
 */

// Color schemes
const CHART_COLORS = {
    primary: '#3b82f6',
    success: '#10b981',
    warning: '#f59e0b',
    danger: '#ef4444',
    purple: '#8b5cf6',
    gray: '#6b7280'
};

/**
 * Create a line chart
 */
function createLineChart(canvasId, data, options = {}) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    const defaultOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
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
                beginAtZero: true
            }
        }
    };

    const finalOptions = { ...defaultOptions, ...options };

    return new Chart(ctx, {
        type: 'line',
        data: data,
        options: finalOptions
    });
}

/**
 * Create a bar chart
 */
function createBarChart(canvasId, data, options = {}) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    const defaultOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: false
            }
        },
        scales: {
            y: {
                beginAtZero: true
            }
        }
    };

    const finalOptions = { ...defaultOptions, ...options };

    return new Chart(ctx, {
        type: 'bar',
        data: data,
        options: finalOptions
    });
}

/**
 * Update chart data
 */
function updateChartData(chart, newData) {
    chart.data = newData;
    chart.update();
}

/**
 * Add data point to chart
 */
function addDataPoint(chart, label, dataPoint) {
    // Limit to 30 data points
    if (chart.data.labels.length > 30) {
        chart.data.labels.shift();
        chart.data.datasets.forEach(dataset => {
            dataset.data.shift();
        });
    }

    chart.data.labels.push(label);
    chart.data.datasets.forEach((dataset, index) => {
        if (Array.isArray(dataPoint)) {
            dataset.data.push(dataPoint[index]);
        } else {
            dataset.data.push(dataPoint);
        }
    });

    chart.update();
}

/**
 * Create real-time metric gauge
 */
function createGaugeChart(canvasId, value, max, label) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    return new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: [label, 'Remaining'],
            datasets: [{
                data: [value, max - value],
                backgroundColor: [
                    value / max > 0.8 ? CHART_COLORS.success : 
                    value / max > 0.5 ? CHART_COLORS.warning : 
                    CHART_COLORS.danger,
                    '#e5e7eb'
                ],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            rotation: -90,
            circumference: 180,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    enabled: false
                }
            }
        }
    });
}

/**
 * Export chart as image
 */
function exportChartAsImage(chart, filename = 'chart.png') {
    const url = chart.toBase64Image();
    const link = document.createElement('a');
    link.download = filename;
    link.href = url;
    link.click();
}

/**
 * Animate number counter
 */
function animateValue(element, start, end, duration) {
    const range = end - start;
    const increment = range / (duration / 16);
    let current = start;

    const timer = setInterval(() => {
        current += increment;
        if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
            current = end;
            clearInterval(timer);
        }
        element.textContent = Math.round(current);
    }, 16);
}

/**
 * Format large numbers
 */
function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toFixed(0);
}

/**
 * Generate random data for testing
 */
function generateRandomData(count, min = 0, max = 100) {
    return Array.from({ length: count }, () => 
        Math.floor(Math.random() * (max - min + 1)) + min
    );
}

// Export functions
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        createLineChart,
        createBarChart,
        createGaugeChart,
        updateChartData,
        addDataPoint,
        exportChartAsImage,
        animateValue,
        formatNumber,
        CHART_COLORS
    };
}

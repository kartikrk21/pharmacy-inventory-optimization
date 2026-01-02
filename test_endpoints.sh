#!/bin/bash
# Test all API endpoints

API_URL="http://localhost:5000/api"

echo "========================================"
echo "Testing Pharmacy Inventory API Endpoints"
echo "========================================"
echo ""

# Test health check
echo "1. Testing health check..."
curl -s "${API_URL}/health" | python3 -m json.tool
echo ""

# Test medicines list
echo "2. Testing GET /medicines..."
curl -s "${API_URL}/medicines" | python3 -m json.tool | head -30
echo ""

# Test medicine details
echo "3. Testing GET /medicine/<id>..."
curl -s "${API_URL}/medicine/MED0001" | python3 -m json.tool
echo ""

# Test statistics
echo "4. Testing GET /statistics..."
curl -s "${API_URL}/statistics" | python3 -m json.tool
echo ""

# Test placing order
echo "5. Testing POST /orders/place..."
curl -s -X POST "${API_URL}/orders/place" \
  -H "Content-Type: application/json" \
  -d '{"medicine_id":"MED0001","quantity":100}' | python3 -m json.tool
echo ""

# Test adding prescription
echo "6. Testing POST /prescriptions..."
curl -s -X POST "${API_URL}/prescriptions" \
  -H "Content-Type: application/json" \
  -d '{"medicine_id":"MED0001","quantity":5,"timestamp":"2024-01-15T10:00:00"}' | python3 -m json.tool
echo ""

# Test alerts
echo "7. Testing GET /alerts..."
curl -s "${API_URL}/alerts" | python3 -m json.tool
echo ""

# Test inventory adjustment
echo "8. Testing POST /inventory/adjust..."
curl -s -X POST "${API_URL}/inventory/adjust" \
  -H "Content-Type: application/json" \
  -d '{"medicine_id":"MED0001","adjustment":50,"reason":"Manual restock"}' | python3 -m json.tool
echo ""

# Test optimization
echo "9. Testing POST /optimize..."
curl -s -X POST "${API_URL}/optimize" \
  -H "Content-Type: application/json" \
  -d '{"medicine_ids":["MED0001","MED0002","MED0003"]}' | python3 -m json.tool | head -50
echo ""

# Test forecast
echo "10. Testing GET /forecast/<id>..."
curl -s "${API_URL}/forecast/MED0001" | python3 -m json.tool | head -30
echo ""

echo "========================================"
echo "All tests completed!"
echo "========================================"
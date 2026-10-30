#!/usr/bin/env bash
set -e
URL="${URL:-http://127.0.0.1:5001}"

echo "== 1. Service health =="
curl -sS "$URL/health" | python3 -m json.tool

echo
echo "== 2. Customer likely to leave (new, month-to-month, fiber, high bill) =="
curl -sS -X POST "$URL/predict" -H "Content-Type: application/json" -d @demo/high_risk.json | python3 -m json.tool

echo
echo "== 3. Customer likely to stay (5 years, two-year contract, support, family) =="
curl -sS -X POST "$URL/predict" -H "Content-Type: application/json" -d @demo/low_risk.json | python3 -m json.tool

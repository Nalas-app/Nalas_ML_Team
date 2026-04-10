# ML Costing Service — Backend Integration Guide
**For: Jai (Backend Lead)**  
**Written by: Vasanth (ML Team)**  
**Endpoint base URL:** `http://localhost:8001` (dev) → will update with production URL on deployment

---

## Quick Start

### Predict cost for a menu item

```http
POST /ml/predict-cost
Content-Type: application/json
```

**Request body:**
```json
{
  "menuItemId": "ITEM-001",
  "quantity": 50,
  "eventDate": "2026-12-15",
  "guestCount": 200
}
```

- `menuItemId` — required, string
- `quantity` — required, integer, must be ≥ 1
- `eventDate` — optional, string, format `YYYY-MM-DD`
- `guestCount` — optional, integer, must be ≥ 1

**Response:**
```json
{
  "ingredientCost": 251802.11,
  "laborCost": 37770.32,
  "overheadCost": 25180.21,
  "totalCost": 360077.02,
  "confidence": 0.89,
  "modelVersion": "v1.0.0",
  "method": "ml_model"
}
```

- `method` will be `"ml_model"` or `"rule_based"` — use this to know which engine responded
- `confidence` is between 0.0 and 1.0 — flag anything below 0.6 for review

---

## Node.js / JavaScript Integration

### Using fetch (no extra packages needed)

```javascript
async function getMenuItemCost(menuItemId, quantity, options = {}) {
  const payload = {
    menuItemId,
    quantity,
    ...(options.eventDate && { eventDate: options.eventDate }),
    ...(options.guestCount && { guestCount: options.guestCount }),
  };

  const response = await fetch("http://localhost:8001/ml/predict-cost", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(`ML service error: ${error.error.message}`);
  }

  return await response.json();
}

// Usage
const cost = await getMenuItemCost("ITEM-001", 50, {
  eventDate: "2026-12-15",
  guestCount: 200,
});
console.log(`Total cost: Rs.${cost.totalCost}`);
console.log(`Predicted by: ${cost.method} (confidence: ${cost.confidence})`);
```

### Using axios

```javascript
const axios = require("axios");

async function getMenuItemCost(menuItemId, quantity, options = {}) {
  try {
    const { data } = await axios.post("http://localhost:8001/ml/predict-cost", {
      menuItemId,
      quantity,
      ...options,
    });
    return data;
  } catch (err) {
    if (err.response) {
      throw new Error(`ML service error ${err.response.status}: ${err.response.data.error.message}`);
    }
    throw new Error("ML service unreachable");
  }
}
```

---

## Error Handling

| Status | Meaning | What to do |
|--------|---------|------------|
| 200 | Success | Use the response normally |
| 422 | Invalid request (bad menuItemId, quantity=0, etc.) | Log and reject the order request |
| 400/404 | Menu item not found in ML system | Fall back to manual costing |
| 500 | ML service internal error | Fall back to manual costing, alert team |

**Always wrap the ML call in a try/catch.** If the service is down, do not block the order flow — fall back gracefully.

```javascript
async function getCostWithFallback(menuItemId, quantity) {
  try {
    const result = await getMenuItemCost(menuItemId, quantity);
    return { source: "ml", ...result };
  } catch (err) {
    console.error("ML service unavailable, using fallback:", err.message);
    return { source: "fallback", totalCost: null, confidence: 0 };
  }
}
```

---

## Other Useful Endpoints

### Health check — call this on backend startup to verify ML service is up
```http
GET /ml/health
```
```json
{
  "status": "healthy",
  "menuItemsLoaded": 34,
  "ingredientsLoaded": 129,
  "mlActive": true,
  "uptime": "0:12:34"
}
```

### Get all valid menu item IDs
```http
GET /ml/menu-items
```
```json
{
  "total": 34,
  "predictionReady": 34,
  "items": [
    { "id": "ITEM-001", "name": "Chicken Fry (KCF)", "category": "..." }
  ]
}
```

### Shadow deployment metrics — check how ML vs rule-based is performing
```http
GET /ml/metrics
```

---

## Swagger UI (interactive testing)
Open in browser while the service is running:  
`http://localhost:8001/docs`

You can test all endpoints directly from there without writing any code.

---

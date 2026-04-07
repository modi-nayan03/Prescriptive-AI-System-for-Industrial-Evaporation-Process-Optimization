# 🚀 Project: Steam Economy Optimization System (6 Effect Evaporation)

---

## 🧠 Overview

This system is designed to **optimize steam economy** in an industrial evaporation plant using:

* Historical plant data (6 lakh records)
* Machine Learning (XGBoost)
* Optimization engine
* Rule-based constraints
* Explainable AI (SHAP-based reasoning)
* RAG (plant manual understanding)

---

## 📊 Data Description

* Total dataset size: **~6,00,000 rows**
* Total features: **72 columns**
* Target variable: **Steam_Economy**

### 🔍 Data Filtering

We focus only on **effective operating range**:

```python
Steam_Economy ∈ [3, 5]
```

After filtering:

* Usable dataset: **~3,00,000 rows**

---

## 🧬 Feature Engineering

### ✅ Selected Features (22 → used in ML model)

```python
ML_MODEL_FEATURES_6_EFFECT = [
    "Spent Liquor Feed Pump Battery Temperature",
    "De-Superheater LP Steam Flow",
    "SEL Density",
    "Process Condensate to Tank Farm",
    "Live Steam Condensate Temperature",
    "SPL Flow",
    "LP Steam Temperature Before De-Superheater",
    "SPL A/C",
    "Strong Evaporated Liquor Discharge Pump Temperature",
    "3rd Effect 2nd Drum Condensate",
    "MP Steam Inlet Temperature",
    "SEL Flow",
    "Separator Vessel 3rd Effect",
    "1st Product Flash Drum Liquor O/L Temperature",
    "Split Flow in 4th Effect",
    "Barometric Condensor Vacuum",
    "Barometric Condensor Cooling Water Flow",
    "Chest Pressure",
    "Evaporation Rate"
]
```

---

## ⚙️ ML Model Configuration

```python
EFFECT_MODE_CONFIG = {
    "6_EFFECT": {
        "MLFLOW_TRACKING_URI": "mlflow-artifacts:/812180679983349338/beb3e3d4d3e54fdb8fe1352e86d4c26a/artifacts/model/model.pkl",
        "features": ML_MODEL_FEATURES_6_EFFECT,
        "name": "6/6 Effect Mode"
    }
}
```

* Model: **XGBoost Regressor**
* Purpose: Predict Steam Economy

---

## 🔒 Safe Operating Bounds (Process Constraints)

Defines **valid operating ranges** for each parameter:

```python
SAFE_OPERATING_BOUNDS = {
    "Spent Liquor Feed Pump Battery Temperature": (65, 90),
    "De-Superheater LP Steam Flow": (40, 80),
    "SEL Density": (1.29, 1.31),
    "Process Condensate to Tank Farm": (140, 260),
    "Live Steam Condensate Temperature": (110, 145),
    "SPL Flow": (920, 1200),
    "Barometric Condensor Cooling Water Flow": (2700, 3900),
    "LP Steam Temperature Before De-Superheater": (140, 180),
    "SPL A/C": (0.30, 0.35),
    "Strong Evaporated Liquor Discharge Pump Temperature": (77, 84),
    "3rd Effect 2nd Drum Condensate": (85, 110),
    "MP Steam Inlet Temperature": (185, 200),
    "SEL Flow": (750, 900),
    "Separator Vessel 3rd Effect": (45, 55),
    "1st Product Flash Drum Liquor O/L Temperature": (104, 125),
    "Split Flow in 4th Effect": (0.47, 0.53),
    "Barometric Condensor Vacuum": (-0.91, -0.925),
    "Chest Pressure": (1.8, 2.9),
    "Evaporation Rate": (200, 350)
}
```

---

## 🎯 Parameter Classification

### 🔹 Fixed Parameters (Non-adjustable)

```python
FIXED_PARAMETERS = [
    "MP Steam Inlet Temperature",
    "Strong Evaporated Liquor Discharge Pump Temperature",
    "SEL Density",
    "De-Superheater LP Steam Flow",
    "Process Condensate to Tank Farm",
    "Live Steam Condensate Temperature",
    "LP Steam Temperature Before De-Superheater",
    "SPL A/C",
    "3rd Effect 2nd Drum Condensate",
    "SEL Flow",
    "Separator Vessel 3rd Effect",
    "Barometric Condensor Vacuum"
]
```

---

### 🔹 Controllable Parameters (Optimization Variables)

```python
CONTROLLABLE_PARAMETERS = [
    "Chest Pressure",
    "Split Flow in 4th Effect",
    "1st Product Flash Drum Liquor O/L Temperature",
    "Barometric Condensor Cooling Water Flow",
    "SPL Flow",
    "Spent Liquor Feed Pump Battery Temperature"
]
```

---

### 🔹 Max Change Constraints (Safety Limits)

```python
MAX_CHANGE_CONSTRAINTS = {
    "Chest Pressure": 0.3,
    "1st Product Flash Drum Liquor O/L Temperature": 0.5,
    "Spent Liquor Feed Pump Battery Temperature": 0.1,
    "SPL Flow": 2.0,
    "Split Flow in 4th Effect": 0.05,
    "Barometric Condensor Cooling Water Flow": 50.0
}
```

---

## 🧠 SHAP-Based Intelligence Layer

Used for **explainability and decision reasoning**

### Purpose:

* Identify which parameters:

  * Increase steam economy
  * Decrease steam economy
* Provide **directional insights**

### Example:

```text
Chest Pressure ↑ → Steam Economy ↓
Split Flow ↑ → Steam Economy ↑
```

---

## ⚙️ Optimization Engine

### Objective:

Adjust controllable parameters to:

* Reach target Steam Economy
* Stay within safe bounds
* Respect max change constraints

### Inputs:

* Current plant values
* Target Steam Economy
* ML model predictions

### Output:

* Recommended parameter changes

---

## 📚 RAG System (Plant Knowledge)

* Uses plant manual (PDF)
* Tables converted into structured text
* Stored in vector DB (FAISS)

### Purpose:

* Provide process explanations
* Support engineering reasoning

Example:

```text
- Steam flow affects heat transfer efficiency
- Split flow balances evaporation load
```

---

## 🔄 End-to-End Pipeline

```text
User Query
   ↓
Query Parsing
   ↓
Target Steam Economy Calculation
   ↓
Optimization Engine
   ↓
ML Prediction
   ↓
Data Band Retrieval (historical patterns)
   ↓
RAG Explanation
   ↓
Final Response Generation
```

---

## 📈 Example Output

```text
Target Steam Economy:
4.3 → 4.43

Recommended Changes:
- Steam Flow: 60 → 58
- Split Flow: 0.48 → 0.52

Based on historical data:
- Steam Flow: 57–59
- Split Flow: 0.50–0.54

Process Explanation:
- Lower steam improves energy efficiency
- Balanced split improves evaporation

Confidence:
Based on 140 records
```

---

## 🧠 System Type

This is an:

### 👉 Explainable Industrial AI Optimization System

Combines:

| Component    | Role             |
| ------------ | ---------------- |
| ML Model     | Prediction       |
| Optimization | Recommendation   |
| Data         | Validation       |
| SHAP         | Explainability   |
| RAG          | Domain Knowledge |

---

## 🔥 Key Strengths

* Data-driven decisions
* Safe & constrained optimization
* Explainable recommendations
* Industrial-grade logic
* Scalable to multiple effect modes

---

## 🚀 Usage (For New Chat Context)

Paste this file and say:

```text
This is my project context.
Now help me improve: [your problem]
```

---

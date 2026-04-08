# 🏭 Boiler Optimization System — Complete Code Documentation
### `Boiler.py` | Utkal Aluminium Power Plant | Unit 1 Boiler

> **Prepared by:** Engineering Team  
> **Date:** March 2026  
> **Status:** Active Deployment — Code Structuring & Justification Document  
> **Purpose:** This document explains what `Boiler.py` does, how it works step by step, and what changes were made — written so that anyone (technical or non-technical) can understand it.

---

## 📖 The Story in Simple Words

Imagine you are running a **huge furnace** (Boiler) that burns coal to produce steam. That steam spins a turbine which generates electricity for the entire Utkal Aluminium factory.

Every day, operators ask the same question:

> **"How can we produce the SAME amount of steam while burning LESS coal?"**

This is exactly what `Boiler.py` does — **automatically and continuously, every 30 seconds.**

It reads live sensor data from the plant, uses AI models to find the best settings, and tells the plant operators exactly what adjustments to make to **save coal without reducing steam output.**

---

## 🏗️ What is the System Architecture?

```
┌─────────────────────────────────────────────────────────────────┐
│                   BOILER OPTIMIZATION SYSTEM                    │
│                        (Boiler.py)                              │
│                                                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │  PI System  │───▶│  AI Models   │───▶│   PI System      │   │
│  │  (Sensors)  │    │  (XGBoost)   │    │   (Results)      │   │
│  │             │    │              │    │                  │   │
│  │ 13 readings │    │ Steam Model  │    │ Recommended      │   │
│  │ every 30s   │    │ SCC Model    │    │ settings written  │   │
│  │             │    │ Efficiency   │    │ back to PI       │   │
│  └─────────────┘    └──────────────┘    └──────────────────┘   │
│                                                                 │
│  Connected to: MLflow (model server) + MinIO (config storage)   │
└─────────────────────────────────────────────────────────────────┘
```

### What is PI System?
- **PI** = Plant Information system — a database that stores all live sensor readings from the boiler (temperature, pressure, flow rates, etc.)
- This code reads from PI and writes recommendations back to PI

### What is MLflow?
- **MLflow** = A server that stores trained AI/ML models  
- The code connects to MLflow to download 3 pre-trained XGBoost models

### What is MinIO?
- **MinIO** = A file storage server (like Google Drive for industrial files)
- The code reads tag mapping CSV files from MinIO to know which PI sensor = which URL

---

## 🤖 The 3 AI Models Used

| # | Model Name | What It Predicts | Role in System |
|---|-----------|-----------------|----------------|
| 1 | **Steam Model** | How much steam (t/h) will be produced? | 🔵 GUIDE — runs inside the optimizer loop |
| 2 | **SCC Model** | How much coal per ton of steam? (SCC ratio) | 🟢 VALIDATOR — checks result after optimization |
| 3 | **Efficiency Model** | What is the boiler efficiency %? | 🟡 SAFETY GUARD — prevents physically impossible suggestions |

> **SCC** = Specific Coal Consumption = Coal Flow ÷ Steam Flow  
> Lower SCC = Better performance (using less coal per unit of steam)  
> **Target Zone:** `0.180 < SCC < 0.220`

---

## 🔄 Full System Flow — Step by Step

### ⏱️ This entire cycle runs automatically every 30 seconds

---

### 📡 STEP 1 — Read Live Data from Plant

```
PI System (Plant Sensors)
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│  13 LIVE SENSOR READINGS (model inputs)                        │
│                                                                │
│  01. CALC_FUEL_AIR_RATIO          → How much air per kg coal?  │
│  02. CALC_WIND_BOX_DP             → Air pressure in windbox    │
│  03. PA_FAN_PRI_AIR_FL_TOT        → Primary air fan speed      │
│  04. WIND_BOX_FAN_TOT             → Secondary air fan speed    │
│  05. SUPR_HTR_ECO_FLUE_GAS_TEMP  → Flue gas temp after ECO    │
│  06. SUPR_HTR_MN_STM_HDR_TEMP_02 → Main steam temperature     │
│  07. COAL_FDR_TOT_FLW             → Coal feed rate (TPH)       │
│  08. ESP_DUCT_INL_FLUE_GAS_TEMP  → APH outlet temperature     │
│  09. COMPN_FD_WTR_FLW             → Feed water flow rate       │
│  10. FD_WTR_REG_STN_TEMP          → Feed water temperature     │
│  11. APH_OXG_ANLY_AVG             → Oxygen % at APH outlet     │
│  12. CALC_STM_ENTHLP              → Steam heat content         │
│  13. CALC_FD_WTR_ENTHLP           → Feed water heat content    │
│                                                                │
│  + ACTUAL STEAM FLOW (Master Parameter — read separately)      │
└────────────────────────────────────────────────────────────────┘
```

✅ **If any sensor reading is missing or invalid → skip this cycle (safety)**

---

### ⚙️ PHASE 1 — Physics Gate (Pre-Optimization Safety Check)

> Before optimization even starts, the system asks: *"How efficient is this boiler RIGHT NOW?"*

```
Physics Inputs (Boiler Chemistry):
├── Flue Gas Temperature (Tfg)
├── Ambient Air Temp (FD Fan Inlet)
├── Hydrogen % in coal (H2)
├── Excess Air %
├── Theoretical Air Required
├── CO at APH Outlet
├── Ash content in coal
├── Moisture in coal (TM)
├── Coal Calorific Value (GCV)
├── Carbon, Sulphur, Nitrogen % in coal
├── Radiation Loss
├── Bottom Ash & ESP Fly Ash %
├── Blowdown Loss
└── Mill Reject Quality & GCV

         │
         ▼
  [PHYSICS FORMULA — ASME PTC 4.1 Standard]
  Calculate: L1 (Dry Flue Gas Loss) + L2 + L3 + L4 + L5 + L6 + L7 ...
  Efficiency = 100% − Total Losses

         │
         ▼
  Set MINIMUM COAL LIMIT for optimizer
  (So AI cannot suggest coal lower than physics allows)
```

> 🛡️ This prevents the AI from giving physically impossible recommendations.

---

### 🔵 PHASE 2 — Steam Optimization (Main AI Engine)

> The optimizer searches for the **best combination of 12 settings** that produces the RIGHT amount of steam with MINIMUM coal.

```
SLSQP Optimizer (Mathematical Solver)
   Runs 10 attempts with different starting points
   
   Each Attempt:
   ┌─────────────────────────────────────────────────────┐
   │  1. Try a set of parameters (fuel, air, coal, etc.) │
   │  2. Ask Steam Model: "Will this produce X steam?"   │
   │  3. Calculate: Coal Reduction + Steam Match Error   │
   │  4. Adjust parameters to reduce the error           │
   │  5. Repeat until satisfied or 150 iterations        │
   └─────────────────────────────────────────────────────┘

Constraints (Rules the optimizer MUST follow):
   ✓ SCC must be between 0.180 and 0.220
   ✓ Coal reduction cannot exceed 20% at once
   ✓ Coal cannot go below physics minimum (from Phase 1)
   ✓ Steam temperature must stay above 120°C
   ✓ O2 % fixed at 3.0%
   ✓ APH temperature fixed at 130°C
   ✓ WindBox DP ≥ 0.25 × Total Air

         │
         ▼
Output: 13 Optimized Parameter Values
```

**What the optimizer minimizes:**
```
MINIMIZE = (Steam Error Penalty × 10) + (Coal Flow Penalty)
         = Difference from target steam  +  Coal used
```

---

### 🔢 Why Does Steam Model Have 13 Inputs but Optimizer Uses Only 12?

> This is one of the most important design decisions in the code.

**The Core Problem:**  
The Steam Model was trained on 13 sensor features. But two of those features — `PA_FAN_PRI_AIR_FL_TOT` (Primary Air) and `WIND_BOX_FAN_TOT` (Secondary Air) — are **not independent**. They are both controlled by the same single dial: **Total Air Flow**.

> In a real boiler: Primary Air is always ~38.5% of Total Air, and Secondary Air is always ~57.33% of Total Air. You cannot set them independently.

**The Solution — Merge 2 into 1:**  
Instead of giving the optimizer 2 separate variables (which would allow it to suggest physically impossible combinations), the code collapses them into a single variable `total_air`:

```
13 Steam Model Features          →    12 Optimizer Variables
─────────────────────────────────────────────────────────────
x[0] FUEL_AIR_RATIO              →    r[0] fuel_air_ratio
x[1] WIND_BOX_DP                 →    r[1] wind_box_dp
x[2] PA_FAN_PRI_AIR_FL_TOT  ─┐
                               ├──▶  r[5] total_air  (MERGED)
x[3] WIND_BOX_FAN_TOT       ─┘
x[4] ECO_FLUE_GAS_TEMP           →    r[2] eco_gas_temp
x[5] MAIN_STEAM_TEMP             →    r[3] steam_temp
x[6] COAL_FDR_TOT_FLW            →    r[4] coal
x[7] APH_OUTLET_TEMP             →    r[7] aph_temp  (FIXED=130°C, not optimized)
x[8] FEEDWATER_FLOW              →    r[6] fdw_flow
x[9] FEEDWATER_TEMP              →    r[8] fw_temp   (FIXED=160°C, not optimized)
x[10] O2_%                       →    r[9] o2        (FIXED=3.0%, not optimized)
x[11] STEAM_ENTHALPY             →    r[10] stm_enth (FIXED=810, not optimized)
x[12] FEEDWATER_ENTHALPY         →    r[11] fdw_enth (FIXED=160, not optimized)
─────────────────────────────────────────────────────────────
13 features  →  7 active vars + 1 merged + 4 fixed = 12 total
```

**When sending to Steam Model** (to check steam output):  
The code **expands back** from 12 → 13 using fixed ratios:
```
x[2] = 0.385 × total_air   (Primary Air = 38.5% of total)
x[3] = 0.5733 × total_air  (Wind Box Air = 57.33% of total)
```

**Summary of the 12 variables:**

| # | Variable | Type | Notes |
|---|---------|------|-------|
| r[0] | Fuel-Air Ratio | 🟢 Active | Optimizer adjusts freely |
| r[1] | Wind Box DP | 🟢 Active | Optimizer adjusts freely |
| r[2] | ECO Flue Gas Temp | 🟢 Active | Optimizer adjusts freely |
| r[3] | Main Steam Temp | 🟢 Active | Optimizer adjusts freely |
| r[4] | Coal Flow | 🟢 Active | **Primary target of reduction** |
| r[5] | Total Air | 🟢 Active | Merged from PA+WB fans |
| r[6] | Feed Water Flow | 🟢 Active | Tied to steam target |
| r[7] | APH Outlet Temp | 🔴 Fixed | Always 130°C (design) |
| r[8] | Feed Water Temp | 🔴 Fixed | Always 160°C (design) |
| r[9] | O2 % | 🔴 Fixed | Always 3.0% (design) |
| r[10] | Steam Enthalpy | 🔴 Fixed | Always 810 kJ/kg |
| r[11] | FD Water Enthalpy | 🔴 Fixed | Always 160 kJ/kg |

> ✅ **Result:** 7 truly free variables + 1 physically constrained merge + 4 fixed design points = a faster, more physically meaningful optimization.

---


> Take the 13 optimized parameters + actual steam flow, feed into SCC Model

```
Input: 13 Optimized Parameters + Actual Steam Flow (14 total)
         │
         ▼
   [SCC Model — XGBoost]
   Predicts: "What will the Specific Coal Consumption be?"
         │
         ▼
Safety Check:
   IF predicted SCC > current actual SCC:
      → Use current SCC (do not report worse than actual)
      (This protects against model over-estimation)
```

---

### 🟡 PHASE 4 — Boiler Efficiency Decision (Safety Override)

> The system compares 3 different efficiency estimates and picks the HIGHEST (most conservative):

```
Three Efficiency Sources:
   1. ML Current Prediction    (XGBoost — current state)
   2. ML Optimized Prediction  (XGBoost — with optimized params)
   3. PI Actual Efficiency     (Live from plant DCS)

         │
         ▼
   Pick HIGHEST value → "Adopted Efficiency"

         │
         ▼
   Required Coal = (Steam × Heat Content) ÷ (Efficiency × GCV)

         │
         ▼
Final Check:
   IF ML-suggested coal < Physics-required coal by >0.5 TPH:
      → OVERRIDE: Use physics coal (AI was too optimistic)
   ELSE:
      → Use ML-suggested coal (AI and physics agree)
```

---

### 📐 Efficiency Model — Which Features Come From Where?

> The Efficiency Model (XGBoost) takes **15 features**. They come from two sources:
> - Features **inherited from previous models** (Steam Optimization outputs)
> - Features **newly computed from physics losses** (calculated in Phase 1)

```
┌─────────────────────────────────────────────────────────────────────┐
│  15 FEATURES FOR EFFICIENCY MODEL                                   │
│                                                                     │
│  ── FROM STEAM OPTIMIZATION (previous model outputs) ──────────────│
│                                                                     │
│  🔵  PA_FAN_TOT              ← x[2] from Steam Model               │
│       Primary Air Fan Total Flow                                    │
│                                                                     │
│  🔵  SUPR_HTR_ECO_FLUE_GAS_TEMP  ← x[4] from Steam Model          │
│       ECO Outlet Flue Gas Temperature                              │
│                                                                     │
│  🔵  SUPR_HTR_MN_STM_HDR_TEMP_02  ← x[5] from Steam Model         │
│       Main Steam Header Temperature                                │
│                                                                     │
│  🔵  WIND_BOX_FAN_TOT        ← x[3] from Steam Model               │
│       Wind Box Fan Total Flow                                       │
│                                                                     │
│  🔵  CALC_SPECEFIC_COAL_CONS ← derived from optimized coal/steam   │
│       Specific Coal Consumption (SCC) after optimization            │
│                                                                     │
│  🔵  ESP_OUTL_GAS_TEMP_AVG   ← x[7] (APH temp) from Steam Model    │
│       APH / ESP Outlet Gas Temperature                              │
│                                                                     │
│  ── FROM PHYSICS LOSS CALCULATION (Phase 1 & 4 outputs) ───────────│
│                                                                     │
│  🟡  CALC_TOT_LOSS           ← Total of L1+L2+L3+L4+L5+L6+L7+L10  │
│       Sum of all boiler heat losses (physics formula)               │
│                                                                     │
│  🟡  HEAT_LOSS_IN_DRY_FLUE_GAS  ← L1 (physics)                    │
│       Dry Flue Gas Heat Loss                                        │
│                                                                     │
│  🟡  CALC_LOSS_DUE_TO_H2O_IN_FUEL ← L2 (physics)                  │
│       Heat Loss from Moisture in Coal                               │
│                                                                     │
│  🟡  CALC_LOSS_DUE_TO_H2O_IN_AIR  ← L4 (physics)                  │
│       Heat Loss from Moisture in Combustion Air                     │
│                                                                     │
│  🟡  CALC_MILL_REJCT_LOSS    ← L12 (physics)                       │
│       Mill Reject Loss                                              │
│                                                                     │
│  🟡  CALC_BLW_DOWN_LOSS      ← L11 (from PI sensor)               │
│       Blowdown Loss                                                 │
│                                                                     │
│  🟡  MNUL_ENTRY_ESP_RAD_LOSS ← Radiation_Loss (from PI sensor)     │
│       Radiation + Convection Loss                                   │
│                                                                     │
│  ── FROM PI SENSOR (read directly, not from any model) ────────────│
│                                                                     │
│  🟠  CHMNY_FLUE_GAS_NOX      ← NOx at chimney (proxy = 0)          │
│       Note: Set to 0 as proxy — tag not always available            │
│                                                                     │
│  🟠  CHMNY_FLUE_GAS_CO       ← CO_APH_OUT (from PI sensor)         │
│       Carbon Monoxide at chimney outlet                             │
└─────────────────────────────────────────────────────────────────────┘
```

**When run TWICE — current vs optimized:**

The Efficiency Model is actually called **twice in Phase 4**:

| Run | Input Source | Purpose |
|-----|-------------|--------|
| **Run 1 (Current)** | Current plant sensor values (x[2],[3],[4],[5] from live PI) | Baseline: *"What is efficiency NOW?"* |
| **Run 2 (Optimized)** | Optimized values from Steam Optimizer (x[2],[3],[4],[5] post-optimization) | Target: *"What will efficiency BE after our changes?"* |

The 6 features that **change between the two runs** (the ones that come from Steam Optimizer):

```python
# These 6 are SWAPPED to optimized values in Run 2:
PA_FAN_TOT               → optimized_params[2]   (was model_inputs_13[2])
CALC_SPECEFIC_COAL_CONS  → optimized_scc          (was current_scc)
ESP_OUTL_GAS_TEMP_AVG    → optimized_params[7]   (was current APH temp)
SUPR_HTR_ECO_FLUE_GAS_TEMP → optimized_params[4] (was model_inputs_13[4])
SUPR_HTR_MN_STM_HDR_TEMP_02 → optimized_params[5] (was model_inputs_13[5])
WIND_BOX_FAN_TOT         → optimized_params[3]   (was model_inputs_13[3])
```

> The other 9 features (physics losses) **stay the same** in both runs, because the physics calculations don't change between current and optimized state in Phase 4.

---


### 📤 FINAL STEP — What Gets Written to PI vs Only Printed to Console?

> **Key question:** You read 13 parameters — how many are actually written back to PI with optimized values?

---

#### 🔵 GROUP A — The 13 Inputs: Written Back to PI as Optimized Values

After optimization, each of the 13 input features is mapped to a **prediction write-tag**.  
A value is written to PI **only if that write-tag exists in the write CSV file on MinIO**.

| # | Feature Read from PI | Optimized? | Write Tag | Notes |
|---|---------------------|-----------|-----------|-------|
| x[0] | CALC_FUEL_AIR_RATIO | ✅ Active | `FUEL_AIR_RATIO_PRED` | Optimizer changes this |
| x[1] | CALC_WIND_BOX_DP | ✅ Active | `WIND_BOX_DP_PRED` | Optimizer changes this |
| x[2] | PA_FAN_PRI_AIR_FL_TOT | ✅ Active | `PA_FAN_TOT_PRED` | = 38.5% × total_air |
| x[3] | WIND_BOX_FAN_TOT | ✅ Active | `WIND_BOX_AIR_TOT_PRED` | = 57.33% × total_air |
| x[4] | SUPR_HTR_ECO_FLUE_GAS_TEMP | ✅ Active | `ECO_GAS_TEMP_PRED` | Optimizer changes this |
| x[5] | SUPR_HTR_MN_STM_HDR_TEMP_02 | ✅ Active | `STEAM_TEMP_PRED` | Optimizer changes this |
| x[6] | COAL_FDR_TOT_FLW | ✅ Active | `COAL_FLOW_PRED` | **Main KPI — coal reduction** |
| x[7] | ESP_DUCT_INL_FLUE_GAS_TEMP | ✅ Active | `ESP_TEMP_PRED` | Optimizer changes this |
| x[8] | COMPN_FD_WTR_FLW | ✅ Active | `COMPN_FD_WTR_FLW_PRED` | Feed water flow |
| x[9] | FD_WTR_REG_STN_TEMP | ✅ Active | `FDW_TEMP_PRED` | Optimizer changes this |
| x[10] | APH_OXG_ANLY_AVG | 🔴 Fixed=3.0% | `O2_PRED` | Written but always = 3.0% |
| x[11] | CALC_STM_ENTHLP | 🔴 Fixed=810 | `STM_ENTH_PRED` | Written but always = 810 |
| x[12] | CALC_FD_WTR_ENTHLP | 🔴 Fixed=160 | `FDW_ENTH_PRED` | Written but always = 160 |

> All 13 are attempted to be written — but **only if the tag exists in the write CSV on MinIO**.  
> x[10], x[11], x[12] go to PI with constant (design) values, not truly optimized ones.

---

#### 🟢 GROUP B — Calculated KPIs Written to PI (Not from the 13 Inputs)

These are **extra computed values** added to PI for monitoring and dashboards:

| Write Tag | Value Written | Source |
|-----------|-------------|--------|
| `SCC_PRED` | Final SCC with safety cap | SCC XGBoost Model |
| `SCC_CURRENT` | Current SCC before optimization | Calculated: coal÷steam |
| `SCC_PRED_OPTIMIZED` | SCC from optimizer result | Optimizer |
| `SCC_PRED_MODEL` | SCC predicted by ML model on optimized inputs | SCC XGBoost |
| `SCC_IMPROVEMENT` | Current SCC − Optimized SCC | KPI |
| `STEAM_ACTUAL` | Live steam flow | Direct from PI |
| `STEAM_GENERATION` | Same as actual steam (alias) | Direct from PI |
| `STEAM_FLOW_PRED` | Steam predicted by Steam Model | Steam XGBoost |
| `STEAM_PRED` | Same as above (alias) | Steam XGBoost |
| `COAL_FLOW_PRED` | Final coal recommendation (may be physics-overridden) | Optimizer or Physics |
| `COAL_FLOW_CURRENT` | Current coal before optimization | Raw PI value |
| `COAL_SAVINGS` | Current coal − Optimized coal (t/h) | KPI |
| `COAL_SAVINGS_PCT` | Coal savings as % | KPI |
| `Boiler_eff_prediction` | Adopted efficiency (highest of 3 sources) | Efficiency Model |
| `BOILER_EFF_BASE` | Physics base efficiency (before deductions) | Physics formula |
| `BOILER_EFF_ADOPTED` | Final efficiency used for coal calculation | Efficiency decision |
| `COAL_FLOW_EFF_REQ` | Physics-mandated minimum coal | Physics formula |
| `SCC_EFF_REQ` | SCC implied by physics coal | Physics formula |

---

#### 🔴 GROUP C — Only Printed to Console, NEVER Written to PI

| Console Output | Why Not Written to PI |
|---------------|-----------------------|
| Steam prediction with current inputs (before opt) | Diagnostic comparison only |
| Individual losses L1 to L12 | Intermediate calculation steps |
| Coal analysis data source (live/default status) | Debug info |
| Efficiency comparison (OR vs ML vs PI eff) | Decision audit trail |
| Per-attempt optimizer results (10 attempts) | Iteration debug |
| Δ change for each of 13 parameters | Console display only |
| ✓ Valid / ⚠ Override messages | Status flags |

---

#### 📊 Complete Count Summary

```
INPUT:    13 parameters fetched from PI (steam model inputs)
───────────────────────────────────────────────────────
OUTPUT written to PI per 30-second cycle:
  Group A:  13 optimized parameter values
            → 9 truly optimized  +  3 fixed-value (x[10,11,12])
  Group B:  18 calculated KPIs
───────────────────────────────────────────────────────
TOTAL ~31 values written to PI per cycle

Console only: Many diagnostic prints (Group C) — NEVER go to PI
```

> ⚠️ **Important:** Actual number written depends on tags present in MinIO CSV files  
> (`SCC_boiler1_write_tag_map.csv` + `BE_Boiler_1_write_tags.csv`).  
> Missing tags are **silently skipped** — no error is thrown.



## 🗂️ Code Structure Map (Section by Section)

```
Boiler.py
│
├── [IMPORTS & LIBRARY SETUP]
│   numpy, pandas, scipy, mlflow, requests, minio...
│
├── [TOP-LEVEL HELPER]
│   load_tag_map_csv_from_minio()
│   → Reads sensor tag-to-URL mapping from MinIO storage
│
└── class IntegratedSteamSCCOptimizer
    │
    ├── ─── ENGINEERING CONSTANTS (__init__) ──────────────────
    │   All fixed design values:
    │   MAX_SCC=0.220, MIN_SCC=0.180, DESIGN_O2=3.0%
    │   DESIGN_APH_TEMP=130°C, DESIGN_FW_TEMP=160°C
    │   MAX_COAL_REDUCTION_PCT=20%, MIN_STEAM_FLOW=120 t/h
    │   Connects to: PI, MLflow, MinIO
    │
    ├── ─── MODEL LOADING ──────────────────────────────────────
    │   load_steam_model()  → XGBoost (13 inputs → steam t/h)
    │   load_scc_model()    → XGBoost (14 inputs → SCC ratio)
    │   load_eff_model()    → XGBoost (15 inputs → efficiency %)
    │
    ├── ─── PI COMMUNICATION ───────────────────────────────────
    │   _get_batch_url()       → Build batch API endpoint
    │   _batch_fetch()         → Fetch multiple tags in 1 request
    │   read_tag_value()       → Read single sensor with caching
    │   _extract_numeric_value() → Handle bad/error sensor codes
    │
    ├── ─── STEAM OPTIMIZATION — DESIGN BOUNDS ─────────────────
    │   get_design_bounds()    → Safe operating range for 13 vars
    │   _reduced_to_full()     → 12D optimizer → 13D model input
    │   _full_to_reduced()     → 13D model input → 12D optimizer
    │   get_reduced_bounds()   → Bounds for 12-dim optimizer
    │   build_initial_bounds() → Setup bounds at startup
    │
    ├── ─── STEAM OPTIMIZATION — OBJECTIVE & SOLVER ────────────
    │   objective()                 → What to minimize
    │   get_steam_constraints()     → Rules optimizer must follow
    │   validate_solution()         → Check if result is valid
    │   steam_constraint()          → Steam equality constraint
    │   update_bounds_for_current_steam() → Dynamic bound update
    │   multi_start_optimization()  → Run 10 attempts, pick best
    │
    ├── ─── SCC PREDICTION ─────────────────────────────────────
    │   predict_scc_from_optimized_params()
    │   → Feed optimized 13 params + steam → SCC Model
    │   → Returns predicted SCC with safety cap
    │
    ├── ─── BOILER EFFICIENCY ──────────────────────────────────
    │   calculate_losses_indirect() → ASME physics formula
    │   │   L1=Dry flue gas, L2=Moisture in fuel, L3=H2 loss
    │   │   L4=Moisture in air, L5=Partial combustion
    │   │   L6=Fly ash, L7=Bottom ash, L8/L9=Sensible heat
    │   │   L10=Radiation, L11=Blowdown, L12=Mill reject
    │   │
    │   calculate_efficiency_based_coal()
    │   → Required coal = (Steam × ΔH) ÷ (Efficiency × GCV)
    │   │
    │   decide_efficiency()
    │   → Compare 3 sources → pick highest → adopt it
    │
    └── ─── DATA PIPELINE ──────────────────────────────────────
        _read_be_data()           → [NEW] Read all BE sensor tags
        collect_current_data()    → Read 13 steam inputs from PI
        write_predictions_to_pi() → Write all results back to PI
        run_optimization_cycle()  → Master: Phase 1→2→3→4→write
        run_real_time_loop()      → Runs forever (every 30 sec)
```

---

## 🔧 Changes Made to This Code (Justification Document)

### Context
The original code was written by the senior engineer and handed over. The following changes were made to **improve readability and maintainability** without changing any business logic or execution behavior.

---

### ✅ Change 1 — Added Section Banners

**What was done:**  
Added visual section header comments at the start of each logical group of functions.

**Why:**  
The original code had no clear visual separation between different parts. When a new engineer opens the file, they cannot immediately understand where the Steam logic ends and the SCC logic begins.

**Impact on execution:** ❌ Zero — comments have no effect on Python execution.

```python
# BEFORE (no section marker):
    def load_steam_model(self):
        ...

# AFTER (clear section banner):
    # ==========================================
    # MODEL LOADING
    # Load all 3 XGBoost models from MLflow server
    # ==========================================
    def load_steam_model(self):
        ...
```

---

### ✅ Change 2 — Extracted `_read_be_data()` Helper Method

**What was done:**  
The same 18-line code block that reads Boiler Efficiency sensor tags from PI appeared **twice** in `run_optimization_cycle()` — once in Phase 1 and once in Phase 3. This was consolidated into a single helper method `_read_be_data()`.

**Why:**  
- Violates the **DRY principle** (Don't Repeat Yourself) — a core software engineering standard  
- If a tag name needs to change, it had to be changed in 2 places — easy to miss one  
- The new method makes it clear this is a reusable data-reading step  

**Impact on execution:** ❌ Zero — the helper does the exact same thing as the inline code did.

```python
# BEFORE (repeated twice in the file — once in Phase 1, once in Phase 3):
    be_data = {}
    be_defaults = {"GCV": 3500.0, "C": 45.0, "H2": 3.5, "TM": 12.0, "ASH": 35.0}
    for var_name, tag_name in [
        ("Tfg", "Tfg"), ("FDfanInlet", "FDfanInlet"), ("H2", "H2"),
        ("Excess_Air", "Excess_Air"), ("Theoritical_Air", "Theoritical_Air"),
        ("CO_APH_OUT", "CO_APH_OUT"), ("ASH", "ASH"), ("TM", "TM"),
        ("GCV", "GCV"), ("C", "C"), ("S", "S"), ("N2", "N2"),
        ("Radiation_Loss", "Radiation_Loss"), ("BottomAsh", "BottomAsh"),
        ("EspAsh", "EspAsh"), ("Blow_down_loss", "Blow_down_loss"),
        ("RejectQuality", "RejectQuality"), ("RejectGCV", "RejectGCV")
    ]:
        tag_url = self.be_input_tags.get(tag_name)
        if tag_url:
            val = self.read_tag_value(tag_url)
            be_data[tag_name] = val if val is not None else be_defaults.get(tag_name, 0.0)
        else:
            be_data[tag_name] = be_defaults.get(tag_name, 0.0)

# AFTER (one helper method, called in both places):
    def _read_be_data(self):
        """Helper: Read all BE sensor tags from PI. Used in Phase 1 and Phase 3."""
        ...
        return be_data

    # In Phase 1:
    be_data = self._read_be_data()

    # In Phase 3:
    be_data = self._read_be_data()
```

---

### 📊 Summary of All Changes

| # | Change Type | What Changed | Lines Affected | Execution Impact |
|---|------------|-------------|----------------|-----------------|
| 1 | Cosmetic | Added `MODEL LOADING` section banner | Line 284 | None |
| 2 | Cosmetic | Added `PI COMMUNICATION` section banner | Line 303 | None |
| 3 | Cosmetic | Added `STEAM OPTIMIZATION - DESIGN BOUNDS` section banner | Line 403 | None |
| 4 | Structural | Added `_read_be_data()` helper method | Lines 1006–1031 | None (identical behavior) |
| 5 | Structural | Replaced Phase 1 duplicate loop with `self._read_be_data()` | ~Line 1138 | None (identical behavior) |
| 6 | Structural | Replaced Phase 3 duplicate loop with `self._read_be_data()` | ~Line 1310 | None (identical behavior) |

---

### 🔍 What Was NOT Changed

> The following were intentionally left **untouched**:

- ✅ All original comments (every single one preserved)
- ✅ All physics formulas (`calculate_losses_indirect`)
- ✅ All optimization logic (`multi_start_optimization`, `objective`)
- ✅ All ML model loading and prediction calls
- ✅ All PI read/write logic
- ✅ All constants and configuration values
- ✅ All constraint definitions
- ✅ All unused functions (kept for future use)
- ✅ All MLflow and MinIO connection settings

---

## 🔑 Key Engineering Design Constants (Quick Reference for Client Changes)

> If the client requests changes to operating limits, these are the ONLY values to change:

```python
# In __init__() method — all safe to modify:

self.MAX_SCC = 0.220             # ← Upper SCC limit (increase to relax)
self.MIN_SCC = 0.180             # ← Lower SCC limit
self.MAX_COAL_REDUCTION_PCT = 0.20  # ← Max 20% coal cut per cycle
self.MIN_STEAM_FLOW = 120.0      # ← Don't optimize below this steam flow
self.WINDOW_SEC = 30             # ← How often the cycle runs (seconds)
self.DESIGN_APH_TEMP = 130.0     # ← Fixed APH temperature (°C)
self.DESIGN_FW_TEMP = 160.0      # ← Fixed feed water temperature (°C)
self.DESIGN_O2 = 3.0             # ← Fixed O2 target (%)
self.ML_STEAM_TOLERANCE = 2.0    # ← Acceptable steam prediction error (t/h)
self.REFERENCE_DESIGN_EFF = 85.0 # ← Design boiler efficiency (%)
```

---

## 📁 Related Files in This Project

| File | Purpose |
|------|---------|
| `Boiler.py` | **This file** — Main boiler optimization system |
| `PSO_TR1.py` | Particle Swarm Optimization — Train Route 1 |
| `PSO_TR2.py` | Particle Swarm Optimization — Train Route 2 |
| `Train 1.py` | Training pipeline for ML models |
| `En_PSO_TR1.py` | Enhanced PSO — Train Route 1 |
| `PI_Credential.py` | PI system login credentials (imported by Boiler.py) |

---

## ✅ Verification Checklist

Before deploying any change to `Boiler.py`:

- [ ] Does the optimization cycle still complete without errors?
- [ ] Is the SCC output within the 0.180–0.220 range?
- [ ] Is the steam prediction within ±2 t/h of actual?
- [ ] Is the coal recommendation physically reasonable (not below physics floor)?
- [ ] Are all 13 outputs being written to PI successfully?
- [ ] Is the boiler efficiency between 70% and 95%?

---

*Document generated: March 2026 | Code base: `Boiler.py` v1.1 (Structured)*

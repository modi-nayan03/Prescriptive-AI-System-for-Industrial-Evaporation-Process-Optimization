# 🏭 Steam Economy Optimization System — Complete Code Documentation
### `llm_agent.py` & `query_engine.py` | Industrial Evaporation Plant

> **Prepared by:** Nayan Modi  
> **Date:** April 2026  
> **Status:** Active Deployment — Code Structuring & Execution Document  
> **Purpose:** This document explains what the Steam Economy Optimization system does, how it works step by step, and how the various "brains" (Machine Learning, Physics, and LLM) work together. Written so that anyone (technical or non-technical) can understand it.

---

## 📖 The Story in Simple Words

Imagine you are managing an **industrial 6-Effect Evaporation plant**. The goal of this plant is to evaporate water from "spent liquor" using hot steam. 

Every day, the engineers want to know:
> **"How can we adjust our valves, pressures, and cooling water to evaporate the maximum amount of water using the MINIMUM amount of steam?"**

This efficiency metric is called **Steam Economy**.

This system, driven by `llm_agent.py` and `query_engine.py` does this automatically. It reads current sensor data, runs thousands of mathematical simulations to find the best configuration, consults actual plant operating manuals to ensure the changes make sense, and prints a final, easy-to-read recommendation report for the plant operators.

---

## 🏗️ What is the System Architecture?

```text
┌─────────────────────────────────────────────────────────────────┐
│              STEAM ECONOMY OPTIMIZATION SYSTEM                  │
│                                                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐    │
│  │ Current KPI │───▶│  PSO Engine  │───▶│   RAG / FAISS    │    │
│  │ (Baseline)  │    │ (query_engine)    │ (Manual lookup)  │    │
│  │             │    │              │    │                  │    │
│  │ reads 20    │    │ ML Model     │    │ vector search on │    │
│  │ parameters  │    │ 100 particles│    │ changed metrics  │    │
│  └─────────────┘    └──────┬───────┘    └─────────┬────────┘    │
│                            │                      │             │
│                            ▼                      ▼             │
│                     ┌──────────────────────────────────────┐    │
│                     │       GROQ LLM (Llama 3.3 70B)       │    │
│                     │                                      │    │
│                     │ Generates Final Engineering Report   │    │
│                     └──────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### What is PSO (Particle Swarm Optimization)?
- **PSO** is a mathematical search engine. Imagine 50 "scouts" (particles) exploring a landscape of different valve settings. They talk to each other to find the absolute highest peak (the highest Steam Economy) without violating safety limits.

### What is RAG (Retrieval-Augmented Generation)?
- **RAG** uses a vector database (`FAISS`) containing your actual plant engineering manuals. Before making a recommendation, the AI "reads the manual" to ensure its mathematical suggestions align with real-world physics.

### What is the XGBoost Model?
- A pre-trained Machine Learning model (`Model/model (1).pkl`) that has learned exactly how the evaporation plant behaves based on historical data.

---

## 🤖 The 3 Core AI "Brains" Used

| # | Brain | Technology | What It Does | Role in System |
|---|-------|------------|--------------|----------------|
| 1 | **Predictor** | XGBoost ML Model | Predicts "If I change this, what will Steam Economy be?" | 🔵 SIMULATOR — acts as the plant physics simulator. |
| 2 | **Optimizer** | PSO Algorithm | Tries thousands of combinations to maximize the Predictor's score. | 🟢 SOLVER — finds the best valve settings. |
| 3 | **Explainer** | Groq LLM + FAISS | Translates math into English and justifies it using factory manuals. | 🟡 COMMUNICATOR — talks to the human engineer. |

---

## 🔄 Full System Flow — Step by Step

### 📡 STEP 1 — Read Live Data (The Baseline)
The code starts by taking a "snapshot" of the plant (`baseline_state`). It records 20 different sensor readings, from cooling water flow to chest pressure. It predicts the *Current Baseline Steam Economy*.

### ⚙️ PHASE 1 — PSO Initialization
The optimizer initializes 50 "particles". It divides the 20 variables into two groups:
*   **Fixed Parameters (14 variables)**: Things we cannot control right now (e.g., Live Steam Condensate Temp).
*   **Controllable Levers (6 variables)**: Things the operator *can* control (e.g., Chest Pressure, Cooling Water).

### 🔵 PHASE 2 — Iterative Mathematical Optimization
> The optimizer runs for 30–50 iterations trying to find the best configuration.

```text
Each Attempt (for 50 particles x 30 loops = 1500 combinations):
   ┌─────────────────────────────────────────────────────┐
   │  1. Adjust the 6 Controllable Levers                │
   │  2. Ask XGBoost Model: "What is the Steam Economy?" │
   │  3. Calculate Safety Penalty (Too high? Clamp it)   │
   │  4. Move particles toward the best score            │
   │  5. Repeat until 30 loops finish.                   │
   └─────────────────────────────────────────────────────┘
```

### 🟡 PHASE 3 — Delta Analysis & Context Retrieval
Once the PSO finishes, the orchestrator (`llm_agent.py`) calculates the **Change (Delta)**.
*   *Chest Pressure:* 2.55 -> 1.80 (Decrease)
*   *Cooling Water:* 3054 -> 2555 (Decrease)

It takes these changed parameters (e.g., "cooling water, chest pressure") and queries the FAISS plant manual database: *"How do these affect steam economy?"*

### 📤 FINAL STEP — LLM Report Generation
The numerical data + the text from the plant manuals are sent to the LLM (Groq Llama-3.3-70B). The LLM generates a professional insight report, which is printed to the console.

---

## 📊 Example Output & How to Explain to Clients

*If a client asks "What exactly is this text telling me?", here is how to explain the final AI output in plain English.*

### The Real AI Output (Example):
```text
Chest Pressure (Kg/cm²G): Increased from 2.55 to 3.17 (+0.62)
Spent Liquor Split Flow (m³/h): Increased from 958.12 to 1157.97 (+199.85)
1st Product Flash Drum Liquor O/L (°C): Decreased from 119.85 to 115.37 (-4.48)

Confidence Score:
Based on the alignment with the plant knowledge base (RAG)... I would assign a Confidence Score of 8 out of 10.
```

### The Explanation for the Client:
1. **The Numbers (Optimization):** During the first stage, the AI figured out the exact numbers needed to use less steam. In this example, it realized that increasing the pressure slightly (from 2.55 to 3.17) and pumping more liquor (from 958 to 1157) creates a "sweet spot" where steam is used perfectly.
2. **The "Why?" (Insights from Manual):** The AI didn't just guess these numbers. After calculating them, it read the factory's safety and operating manuals. It found sections (like the "liquor and process vapour system") that literally state that keeping ideal pressure improves heat transfer. The AI links its math to your actual plant rules.
3. **The Confidence Score (8/10):** The AI tells us exactly how sure it is. An 8/10 means the AI's math perfectly aligns with the plant's manuals and everything looks safe. It docked 2 points simply because humans should always do a final safety check before pushing the buttons.

### 🧠 Wait, What is the "LLM" and How Does It Actually Write This?
*(Client: "How is it writing paragraphs on its own?")*

**LLM** stands for **Large Language Model** (in this case, we use one called "Llama 3"). Think of the LLM as the ultimate **Translator**. 

Here is exactly how it works behind the scenes without any magic:
1. **The Math Engine (PSO)** finishes computing and spits out raw numbers: `Chest Pressure: 2.55 -> 3.17`.
2. **The Database (FAISS)** spits out dry text from page 42 of a manual: `Section 1.6: Chest pressure controls heat conductivity...`
3. We take both of these things, put them in a digital envelope, and hand them to the **LLM**. 
4. The LLM's only job is to act like an experienced engineer sitting at a desk. It reads the math, reads the manual paragraph, and types out a beautifully formatted, easy-to-read summary explaining *why* the numbers make sense based on the manual. 

*In short: The system calculates the best numbers to save money (Math), double-checks those numbers against your factory rulebook (Database), and then uses the LLM (The Translator) to turn it all into a plain-English report for your operators.*

---

## 📐 The ML Model — Which Features Come From Where?

The system analyzes **20 Features**. Here is the breakdown:

#### 🟢 The 6 CONTROLLABLE Parametes (The Optimizer changes these)
1. `Chest Pressure (Kg/cm²G)` — Main steam pressure driver.
2. `Split_Flow_4th_Effect`
3. `1st Product Flash Drum Liquor O/L (°C)`
4. `Cooling Water to Barometric Condenser (m³/h)` — Massive impact on vacuum and boiling.
5. `Spent Liquor Split Flow (m³/h)`
6. `Spent Liquor into Battery (°C)`

#### 🔴 The 14 FIXED Parameters (Read directly from PI/Sensors)
These are environmental or incoming properties. The optimizer *knows* them but *does not change them*:
*   `LP Steam to De-Superheater (TPH)`
*   `Lab - SEL Density (g/cc)`
*   `Process Condensate to Tank Farm (m³/h)`
*   `Live Steam Condensate (°C)`
*   `LP Steam Before De-Superheater (°C)`
*   `SPL - Overflow A/C Ratio`
*   `Strong Evaporated Liquor Out from Battery (°C)`
*   `3rd Effect 2nd Drum Condensate (°C)`
*   `MP Steam I/L (°C)`
*   `SEL_Flow`
*   `Separator Vessel 3rd Effect (%)`
*   `SPL_NA2CO3`
*   `Barometric Condenser (Kg/cm²)`
*   `Total_Evaporation_Rate`

---

## 🗂️ Code Structure Map (Section by Section)

### `llm_agent.py` (The Orchestrator)
```text
class IndustryAgent
 │
 ├── ─── INITIALIZATION (__init__) ─────────────────────────
 │   Loads Groq API key
 │   Locates and loads FAISS Index & Parquet chunks
 │   Initializes SentenceTransformer embedder
 │
 ├── ─── RAG RETRIEVAL ─────────────────────────────────────
 │   safe_search()      → Ensures vector dimensions match (384)
 │   get_plant_context()→ Queries manual for parameters changed
 │
 └── ─── PIPELINE EXECUTION ────────────────────────────────
     run_pipeline()     → Main control loop
                        → Calls PSO
                        → Generates ASCII comparison table
                        → Prompts LLM & prints report
```

### `query_engine.py` (The Mathematical Optimizer)
```text
Steam Economy Optimizer
 │
 ├── ─── CONFIGURATION ──────────────────────────────────────
 │   Feature lists (MODEL_FEATURES, CONTROLLABLE_PARAMETERS)
 │   BOUNDS & MAX_CHANGE_CONSTRAINTS (Safety Guardrails)
 │
 ├── ─── ML MODEL ───────────────────────────────────────────
 │   load_ml_model() → Loads XGBoost .pkl model
 │
 └── ─── PSO ALGORITHM ──────────────────────────────────────
     optimize_steam_economy()
       → Initializes swarm boundaries
       → Applies MAX_CHANGE limits based on current state
       → Runs PSO loop (updates velocities & personal bests)
       → Applies penalty if Steam Economy exceeds +4% cap
       → Returns the best safe Dataframe row
```

---

## 🔑 Key Engineering Design Constants (Quick Guardrails)

If operations asks to change limits, adjust these in `query_engine.py`:

```python
# Absolute safe operational boundaries:
BOUNDS = {
    'Chest Pressure (Kg/cm²G)': (1.2, 4.0), 
    'Split_Flow_4th_Effect': (0.0, 1.0),
    '1st Product Flash Drum Liquor O/L (°C)': (95.0, 145.0), 
    'Cooling Water to Barometric Condenser (m³/h)': (1500.0, 4000.0),
    'Spent Liquor Split Flow (m³/h)': (600.0, 1400.0), 
    'Spent Liquor into Battery (°C)': (65.0, 110.0) 
}

# Maximum allowed move perfectly from the current state (prevent huge jumps):
MAX_CHANGE_CONSTRAINTS = {
    'Chest Pressure (Kg/cm²G)': 0.8,
    'Cooling Water to Barometric Condenser (m³/h)': 500.0,
    ...
}

# Maximum AI Hallucination Guardrail (Cannot improve by more than 4% artificially)
MAX_IMPROVEMENT_PERCENTAGE = 0.04
```

---

## 📁 Related Files in This Project

| File / Folder | Purpose |
|---------------|---------|
| `data_intelligence/llm_agent.py` | Main execution script. Connects logic + LLM. |
| `data_intelligence/query_engine.py` | Standalone PSO search algorithm & Constraints. |
| `Model/model (1).pkl` | Pre-trained XGBoost Simulator. |
| `Manual plant doc/faiss_index.bin` | RAG Vector Database for semantic search. |
| `Manual plant doc/chunks.parquet` | The actual text of the plant manuals. |
| `.env` | Holds the `GROQ_API_KEY` required for the LLM. |

---

## ✅ Verification Checklist

Before deploying any changes to this system, verify:
- [ ] Does `python llm_agent.py` run without dimension or FAISS mismatch errors?
- [ ] Is the predicted baseline Steam Economy realistic (between 3.0 and 5.0)?
- [ ] Are parameter changes within `MAX_CHANGE_CONSTRAINTS`?
- [ ] Is the LLM report clearly identifying the numerical changes?
- [ ] Does the `GROQ_API_KEY` have sufficient rate limits for daily use?

---
*Document generated: April 2026 | Code base: Steam Economy Optimizer (Data Intelligence)*

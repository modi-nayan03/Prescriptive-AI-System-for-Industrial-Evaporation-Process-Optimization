# 🚀 Prescriptive AI System for Industrial Evaporation Process Optimization

An end-to-end industrial AI solution for optimizing **Steam Economy** in a 6-Effect Evaporation plant. This system combines Machine Learning, Particle Swarm Optimization (PSO), and Retrieval-Augmented Generation (RAG) to provide realistic, data-driven operational recommendations backed by actual plant manuals.

---

## 🧠 System Architecture & Knowledge Management

The system intelligence is divided into two distinct but collaborative "brains": **Data Knowledge** and **Plant Knowledge**.

### 1. Data Knowledge (Machine Learning & PSO)
Data knowledge is derived from historical operation data (over 300,000+ industrial records). This is managed via:
*   **Predictive Model (XGBoost Regressor)**: An advanced ML model (`Model/model (1).pkl`) trained to predict the Steam Economy based on 20 different plant parameters (sensors, flows, temperatures).
*   **Particle Swarm Optimization (PSO)**: Contains mathematical constraints and logic (`query_engine.py`). It understands what variables are "Fixed" (environmental or immutable process metrics) and what are "Controllable" (valves, cooling water, chest pressure). It runs simulations to find the mathematical optimum.

### 2. Plant Knowledge (RAG & FAISS)
Mathematical optimization alone isn't enough; the AI must understand *why* doing something makes sense in a real plant. This is supervised by:
*   **Document Vector Database**: Plant manuals and standard operating procedures (SOPs) are chunked and embedded into a FAISS Vector Database (`Manual plant doc/faiss_index.bin` and `chunks.parquet`).
*   **Retrieval-Augmented Generation**: Before the AI makes its final report, it queries this vector database to understand the physics and engineering logic behind the PSO's mathematical recommendations.

---

## ⚙️ Total Workflow: How the flow goes

The entire process operates like a highly experienced digital engineer. Here is the step-by-step flow from intake to final report:

### Step 1: Taking the First Reference (Baseline)
The system never guesses where it is starting. It takes the **Current State** (usually pulled directly from plant PI Tags/sensors) as its "First Reference". 
1. It passes this current state through the XGBoost Model to calculate the **Baseline Steam Economy**. 
2. It registers which parameters cannot be changed, isolating the 6 specific "Controllable Parameters".

### Step 2: Prescriptive Optimization (PSO Execution)
The `query_engine.py` script initializes a swarm of "particles" (scenarios).
1. The PSO algorithm runs for multiple iterations, trying thousands of combinations of the controllable parameters within strictly defined **Safety Bounds** (e.g., Chest Pressure can only move by 0.8 kg/cm² max per step).
2. It scores each scenario by minimizing Steam Consumption while maintaining Target Evaporation.
3. The absolute best scenario is selected: The **Target State**.

### Step 3: Delta Analysis
The orchestrator (`llm_agent.py`) compares the First Reference (Baseline) against the Target State (Optimized). It records exactly which parameters need to change and in what direction (e.g., "Cooling Water decreased by 500 m³/h").

### Step 4: Domain Context Retrieval (RAG)
With the list of recommended changes in hand, the agent queries the FAISS vector database.
*   **Query Example**: "How do these parameters affect steam economy: cooling water, chest pressure, split flow..."
*   The FAISS index returns the top 3 most relevant paragraphs from the actual factory engineering manuals.

### Step 5: Insight Generation (LLM)
Finally, all this data is packaged and sent to the LLM (`Llama-3.3-70b-versatile` via Groq) using a specialized prompt. The LLM receives:
*   The numerical optimization results (Before and After).
*   The exact parameters to change.
*   The text from the plant manuals.

The LLM formats a professional **Industrial AI Insights** report, explaining the recommendations, verifying safety, and assigning a confidence score based on the manuals.

---

## 🛠️ How the Code Works (File Breakdown)

### `data_intelligence/llm_agent.py` (The Orchestrator)
This is the main entry point and the brains behind the operation. 
*   **Initialization**: Loads the Groq API key, the FAISS Index, and the Parquet chunks containing the plant manual text.
*   `run_pipeline()`: Passes the dummy/real current state to the PSO engine. After getting the optimized dataframe back from PSO, it generates an ASCII comparison table.
*   `get_plant_context()`: Encodes the changed parameter names into a high-dimensional vector using `all-MiniLM-L6-v2` and searches the FAISS index for relevant documentation.
*   **Prompting**: Fuses the numerical data and manual text into a structured prompt matrix and calls the LLM for final output generation.

### `data_intelligence/query_engine.py` (The Mathematical Optimizer)
This script is pure math and machine learning.
*   **Configurations**: Defines what features the XGBoost model expects (`MODEL_FEATURES`) and exactly what boundaries are safe for the plant (`BOUNDS` and `MAX_CHANGE_CONSTRAINTS`).
*   `optimize_steam_economy()`: 
    *   Generates a swarm of 50 particles.
    *   Iterates 30+ times.
    *   In every loop, each particle updates its configuration based on its "personal best" and the swarm's "global best".
    *   Applies a severe mathematical penalty if a particle tries to boost Steam Economy beyond realistic limits (e.g., > 4% jump).
    *   Returns the absolute best, physically achievable parameter state.

---

## 🚀 Usage

1.  Set up your `.env` with a valid `GROQ_API_KEY`.
2.  Install requirements ensuring `faiss-cpu`, `langchain-groq`, `xgboost`, and `sentence-transformers` are present.
3.  Run the end-to-end agent:
    ```bash
    python "data_intelligence/llm_agent.py"
    ```
4.  The terminal will output the numerical PSO comparisons followed by the fully generated LLM Insight report.

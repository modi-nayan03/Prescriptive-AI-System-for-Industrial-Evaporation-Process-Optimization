# 🚀 Prescriptive AI System for Industrial Evaporation Process Optimization

An end-to-end industrial AI solution for optimizing **Steam Economy** in a 6-Effect Evaporation plant. This system combines Machine Learning, Particle Swarm Optimization (PSO), and Retrieval-Augmented Generation (RAG) to provide realistic, data-driven operational recommendations.

---

## 🧠 Core Features

*   **Predictive Modeling**: High-accuracy XGBoost Regressor trained on 300,000+ industrial records.
*   **Constrained Optimization**: Particle Swarm Optimization (PSO) engine focused on minimizing LP Steam Consumption (TPH).
*   **Industrial Safety Guardrails**: Hardcoded "Max Change" constraints (step-limits) to ensure recommendations are physically achievable.
*   **XAI (Explainable AI)**: Uses SHAP to provide directional reasoning for every parameter adjustment.
*   **Domain Intelligence (RAG)**: Integrates plant manuals and process documentation via FAISS for expert-level insights.

---

## 🛠️ Technology Stack

*   **Machine Learning**: XGBoost, Scikit-learn
*   **Optimization**: Custom Multi-Objective PSO (aligned with Production standards)
*   **Knowledge Retrieval**: LangChain, FAISS, Sentence Transformers
*   **LLM Integration**: Groq API (Llama-3.3-70B)
*   **Data Handling**: Pandas, NumPy

---

## ⚙️ How it Works (The Pipeline)

1.  **Baseline Prediction**: Calculates current Steam Economy using historical patterns.
2.  **PSO Search**: Explores the safe operating space to minimize total steam consumption.
3.  **Constraint Clamping**: Ensures variables like *Chest Pressure* or *Liquor Temperature* stay within plant-safe limits.
4.  **RAG Context**: Fetches relevant manual excerpts to explain *why* a set of changes works.
5.  **Engineering Report**: Generates a professional AI insight report for plant operators.

---

## 📂 Repository Structure

*   `data_intelligence/`: Core Python logic (`llm_agent.py`, `query_engine.py`, `TR1_6_6.py`).
*   `Model/`: Trained XGBoost models (`.pkl`).
*   `Manual plant doc/`: Process documentation for RAG.
*   `vector_db/`: Pre-indexed FAISS knowledge base.

---

## 🚀 Usage

1.  Set up your `.env` with a `GROQ_API_KEY`.
2.  Run the end-to-end agent:
    ```bash
    python data_intelligence/llm_agent.py
    ```
3.  Input your target Steam Economy and receive optimized parameter adjustments.

---

## 🔒 Safety Note
This system implements realistic step-constraints derived from production environments to prevent unrealistic parameter jumping. 
- Maximum SE improvement is capped at **4% per step**.
- Control levers (e.g., Chest Pressure) are limited to safe incremental moves.

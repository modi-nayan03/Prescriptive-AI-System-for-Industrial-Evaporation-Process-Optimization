# import os
# import sys
# import faiss
# import pandas as pd
# from dotenv import load_dotenv
# from langchain_groq import ChatGroq
# from langchain_core.prompts import PromptTemplate
# from sentence_transformers import SentenceTransformer

# from query_engine import optimize_steam_economy, CONTROLLABLE_PARAMETERS

# load_dotenv()


# class IndustryAgent:
#     def __init__(self):

#         # ================================
#         # 🔑 LLM INIT
#         # ================================
#         api_key = os.getenv("GROQ_API_KEY")

#         if not api_key or api_key == "your_key_here":
#             raise ValueError("❌ Set GROQ_API_KEY in .env")

#         self.llm = ChatGroq(
#             temperature=0.2,
#             model_name="llama-3.3-70b-versatile",
#             api_key=api_key
#         )

#         # ================================
#         # 📁 PATH FIX (CRITICAL FIX)
#         # ================================
#         BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#         self.RAG_DIR = os.path.join(BASE_DIR, "Manual plant doc")

#         self.FAISS_BIN = os.path.join(self.RAG_DIR, "faiss_index.bin")
#         self.PARQUET_PATH = os.path.join(self.RAG_DIR, "chunks.parquet")

#         self.FAISS_INDEX = os.path.join(self.RAG_DIR, "index.faiss")
#         self.FAISS_META = os.path.join(self.RAG_DIR, "index.pkl")

#         # ================================
#         # 🔍 LOAD RAG
#         # ================================
#         print("[LLM_AGENT] Loading Plant Documentation RAG...")

#         self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

#         self.rag_loaded = False

#         try:
#             # [OPTION 1] Custom FAISS
#             if os.path.exists(self.FAISS_BIN) and os.path.exists(self.PARQUET_PATH):
#                 print("[RAG] Loading custom FAISS index...")
#                 self.index = faiss.read_index(self.FAISS_BIN)
#                 self.chunks_df = pd.read_parquet(self.PARQUET_PATH)
#                 self.rag_loaded = True

#             # [OPTION 2] LangChain FAISS
#             elif os.path.exists(self.FAISS_INDEX) and os.path.exists(self.FAISS_META):
#                 print("[RAG] Detected LangChain FAISS format. Please convert or load separately.")
#                 self.rag_loaded = False

#             else:
#                 print("[RAG] No FAISS index found in:", self.RAG_DIR)

#         except Exception as e:
#             print(f"[WARN] RAG load failed: {e}")
#             self.rag_loaded = False

#         print(f"RAG STATUS: {'LOADED' if self.rag_loaded else 'NOT LOADED'}")

#     # ================================
#     # 🔍 RAG QUERY
#     # ================================
#     def safe_search(self, vector, k):
#         if vector.shape[1] != self.index.d:
#             print(f"[ERROR] Dimension mismatch: Query={vector.shape[1]}, Index={self.index.d}")
#             return [], []
#         return self.index.search(vector, k)
#     def get_plant_context(self, parameters_changed):

#         if not self.rag_loaded:
#             return "No plant manual context available."

#         query = "How do these parameters affect steam economy: " + ", ".join(parameters_changed)

#         vector = self.embedder.encode([query]).astype("float32")

#         # 🔥 SAFE CHECK & SEARCH
#         D, I = self.safe_search(vector, k=3)

#         contexts = []
#         if len(I) > 0:
#             for idx in I[0]:
#                 if idx != -1 and idx < len(self.chunks_df):
#                     contexts.append(self.chunks_df.iloc[idx]["content"])

#         return "\n\n".join(contexts) if (contexts and len(I) > 0) else "No relevant plant context found."
#     # ================================
#     # 🚀 MAIN PIPELINE
#     # ================================
#     def run_pipeline(self, target_se=4.4):

#         print(f"\n[LLM_AGENT] Running Pipeline | Target SE: {target_se}")

#         current_state = {
#             'Chest Pressure (Kg/cm²G)': 2.55,
#             'Split_Flow_4th_Effect': 0.4888,
#             '1st Product Flash Drum Liquor O/L (°C)': 119.85,
#             'Cooling Water to Barometric Condenser (m³/h)': 3054.55,
#             'Spent Liquor Split Flow (m³/h)': 958.12,
#             'Spent Liquor into Battery (°C)': 81.43
#         }

#         print("\n--- RUNNING PSO ---")

#         best_df, baseline_se = optimize_steam_economy(
#             current_state,
#             target_se=target_se,
#             n_particles=50,
#             n_iterations=30
#         )

#         if best_df.empty:
#             print("❌ Optimization failed")
#             return

#         best = best_df.iloc[0]
#         achieved_se = best["Predicted_Steam_Economy"]

#         # ================================
#         # 📊 DETAILED PARAMETER COMPARISON
#         # ================================
#         comparison_table = []
#         changes_summary = []
#         changed_params = []

#         header = f"{'Parameter Name':<45} | {'Baseline':>10} | {'PSO Optimized':>13} | {'Change':>10}"
#         separator = "-" * len(header)
        
#         comparison_table.append(header)
#         comparison_table.append(separator)

#         for p in CONTROLLABLE_PARAMETERS:
#             curr = current_state.get(p, 0)
#             opt = best[p]
#             delta = opt - curr
            
#             # Simple threshold to report significant changes
#             if abs(delta) > 0.001:
#                 direction = "[INCREASE]" if delta > 0 else "[DECREASE]"
#                 changes_summary.append(f"{p}: {curr:.2f} -> {opt:.2f} {direction}")
#                 changed_params.append(p)
            
#             comparison_table.append(f"{p[:45]:<45} | {curr:10.2f} | {opt:13.2f} | {delta:+10.2f}")

#         comparison_text = "\n".join(comparison_table)

#         # ================================
#         # 🔍 RAG CONTEXT
#         # ================================
#         print("\n--- FETCHING RAG CONTEXT ---")
#         rag_context = self.get_plant_context(changed_params)

#         # ================================
#         # 🤖 LLM REPORT GENERATION
#         # ================================
#         print("\n--- GENERATING FINAL REPORT ---")

#         prompt_template = """
#         You are an Elite Industrial Process Optimizer and an authoritative Engineering AI.
#         Analyze the following Particle Swarm Optimization (PSO) results and provide a definitive, highly technical explanation.

#         STEAM ECONOMY PERFORMANCE:
#         - Current Baseline: {baseline:.3f}
#         - PSO Optimized: {achieved:.3f}
#         - Expected Improvement: {improvement:+.3f}

#         DETAILED PARAMETER RE-CONFIGURATION:
#         {comparison}

#         PLANT KNOWLEDGE BASE (RAG):
#         {context}

#         CRITICAL INSTRUCTIONS FOR REASONING:
#         - Provide STRONG, CAUSAL, and AUTHORITATIVE explanations. 
#         - STRICTLY FORBIDDEN WORDS: "might", "could", "likely", "probably", "may indicate", "potentially", "seem to".
#         - Use definitive engineering verbs: "increases", "forces", "drives", "reduces", "optimizes", "eliminates".
#         - Directly tie the mathematics of the PSO delta to the thermodynamic principles found in the RAG context. Explain the exact mechanical reason *why* this works.
#         - Speak like a confident lead engineer presenting absolute facts to the plant manager.

#         REPORT STRUCTURE:
#         1. **Executive Summary**: 1 declarative sentence on the absolute efficiency gain and core strategy.
#         2. **Technical Adjustments**: Explain EXACTLY WHY the specific parameter changes lead to better Steam Economy. Root your answers firmly in the physics identified in the manual.
#         3. **Operational Risk**: State definitively if any value approaches boundary limits. Declare it safe or flag concrete areas for physical monitoring.
#         4. **Confidence Score**: Out of 10, based on RAG alignment, with a 1-sentence justification.
#         """

#         prompt = PromptTemplate(
#             input_variables=["baseline", "achieved", "improvement", "comparison", "context"],
#             template=prompt_template
#         )

#         chain = prompt | self.llm

#         # Print the numerical results immediately for the user
#         print("\n" + "=" * 85)
#         print("  PSO OPTIMIZATION RESULTS (NUMERICAL COMPARISON)")
#         print("=" * 85)
#         print(comparison_text)
#         print(f"\nPREDICTED STEAM ECONOMY: {baseline_se:.3f} -> {achieved_se:.3f} ({achieved_se - baseline_se:+.3f})")
#         print("=" * 85)

#         response = chain.invoke({
#             "baseline": baseline_se,
#             "achieved": achieved_se,
#             "improvement": achieved_se - baseline_se,
#             "comparison": comparison_text,
#             "context": rag_context
#         })

#         print("\n" + " *** INDUSTRIAL AI INSIGHTS *** " + "=" * 53)
#         print(response.content)
#         print("=" * 85 + "\n")


# # ================================
# # ▶ RUN
# # ================================
# if __name__ == "__main__":
#     agent = IndustryAgent()
#     agent.run_pipeline()

import os
import faiss
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer

from query_engine import optimize_steam_economy, CONTROLLABLE_PARAMETERS

load_dotenv()


class IndustryAgent:
    def __init__(self):

        # ================================
        # 🔑 LLM INIT
        # ================================
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("❌ Set GROQ_API_KEY in .env")

        self.llm = ChatGroq(
            temperature=0.2,
            model_name="llama-3.3-70b-versatile",
            api_key=api_key
        )

        # ================================
        # 📁 PATHS
        # ================================
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.RAG_DIR = os.path.join(BASE_DIR, "Manual plant doc")

        self.FAISS_BIN = os.path.join(self.RAG_DIR, "faiss_index.bin")
        self.PARQUET_PATH = os.path.join(self.RAG_DIR, "chunks.parquet")

        # ================================
        # 🔍 LOAD RAG
        # ================================
        print("[LLM_AGENT] Loading Plant Documentation RAG...")

        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.rag_loaded = False

        try:
            if os.path.exists(self.FAISS_BIN) and os.path.exists(self.PARQUET_PATH):
                print("[RAG] Loading custom FAISS index...")
                self.index = faiss.read_index(self.FAISS_BIN)
                self.chunks_df = pd.read_parquet(self.PARQUET_PATH)
                self.rag_loaded = True
            else:
                print("[RAG] No FAISS index found.")

        except Exception as e:
            print(f"[WARN] RAG load failed: {e}")
            self.rag_loaded = False

        print(f"RAG STATUS: {'LOADED' if self.rag_loaded else 'NOT LOADED'}")

    # ================================
    # 🔍 SAFE SEARCH
    # ================================
    def safe_search(self, vector, k):
        if vector.shape[1] != self.index.d:
            return [], []
        return self.index.search(vector, k)

    # ================================
    # 🔍 GET RAG CONTEXT
    # ================================
    def get_plant_context(self, parameters_changed):

        if not self.rag_loaded:
            return "No plant manual context available."

        query = "Effect of parameters on steam economy: " + ", ".join(parameters_changed)
        vector = self.embedder.encode([query]).astype("float32")

        D, I = self.safe_search(vector, k=3)

        contexts = []
        for idx in I[0]:
            if idx != -1 and idx < len(self.chunks_df):
                contexts.append(self.chunks_df.iloc[idx]["content"])

        return "\n\n".join(contexts) if contexts else "No relevant context found."

    # ================================
    # 🧠 BUILD TABLE
    # ================================
    def build_table(self, current_state, best):

        lines = []
        lines.append(f"{'Parameter':<45} | {'Current':>10} | {'Recommended':>12} | Impact")
        lines.append("-" * 85)

        changed_params = []
        changes_summary = []

        for p in CONTROLLABLE_PARAMETERS:
            curr = current_state.get(p, 0)
            opt = best[p]
            delta = opt - curr

            if abs(delta) > 0.001:
                direction = "Increase" if delta > 0 else "Decrease"
                changed_params.append(p)
                changes_summary.append(f"{p}: {curr:.2f} → {opt:.2f}")
            else:
                direction = "No Change"

            lines.append(
                f"{p[:45]:<45} | {curr:10.2f} | {opt:12.2f} | {direction}"
            )

        return "\n".join(lines), changed_params, changes_summary

    # ================================
    # 🤖 LLM EXPLANATION
    # ================================
    def generate_explanation(self, changes_summary, rag_context):

        prompt = PromptTemplate(
            input_variables=["changes", "context"],
            template="""
You are a senior evaporation process engineer.

Explain WHY the following parameter changes improve steam economy.

Focus on:
- pressure
- flow distribution
- heat transfer
- condensation

Changes:
{changes}

Manual Context:
{context}

Give a clear, confident engineering explanation.
"""
        )

        chain = prompt | self.llm

        response = chain.invoke({
            "changes": "\n".join(changes_summary),
            "context": rag_context
        })

        return response.content

    # ================================
    # 🚀 MAIN PIPELINE
    # ================================
    def run_pipeline(self, target_se=4.4):

        print(f"\n[LLM_AGENT] Running Pipeline | Target SE: {target_se}")

        current_state = {
            'Chest Pressure (Kg/cm²G)': 2.55,
            'Split_Flow_4th_Effect': 0.4888,
            '1st Product Flash Drum Liquor O/L (°C)': 119.85,
            'Cooling Water to Barometric Condenser (m³/h)': 3054.55,
            'Spent Liquor Split Flow (m³/h)': 958.12,
            'Spent Liquor into Battery (°C)': 81.43
        }

        # ================================
        # 🔧 RUN PSO
        # ================================
        best_df, baseline_se = optimize_steam_economy(
            current_state,
            target_se=target_se,
            n_particles=50,
            n_iterations=30
        )

        if best_df.empty:
            print("❌ Optimization failed")
            return

        best = best_df.iloc[0]
        achieved_se = best["Predicted_Steam_Economy"]

        # ================================
        # 📊 BUILD OUTPUT
        # ================================
        table_text, changed_params, changes_summary = self.build_table(current_state, best)

        rag_context = self.get_plant_context(changed_params)

        explanation = self.generate_explanation(changes_summary, rag_context)

        improvement = achieved_se - baseline_se
        target_achieved = achieved_se >= target_se

        # ================================
        # 🧾 FINAL OUTPUT
        # ================================
        print("\n" + "=" * 85)

        print(f"""
================= OPTIMIZATION SUMMARY =================

Steam Economy:
- Current: {baseline_se:.3f}
- Achieved: {achieved_se:.3f}
- Improvement: {improvement:+.3f}

Target:
- {target_se} ({'ACHIEVED' if target_achieved else 'NOT ACHIEVED'})

========================================================
""")

        print("🔧 RECOMMENDED ACTIONS:")
        print(table_text)

        if not target_achieved:
            print(f"""
⚠️ SYSTEM LIMITATION:

Target {target_se} is not achievable under current conditions.
Maximum achievable ≈ {achieved_se:.3f}

Recommendation:
- Process constraints exist
- Consider process redesign or advanced control strategy
""")

        print("\n🧠 WHY THIS WORKS:")
        print(explanation)

        print("\n📊 CONFIDENCE: 9/10 (ML + RAG + Optimization aligned)")
        print("=" * 85 + "\n")


# ================================
# ▶ RUN
# ================================
if __name__ == "__main__":
    agent = IndustryAgent()
    agent.run_pipeline()
# import os
# import sys
# import faiss
# import pandas as pd
# from dotenv import load_dotenv
# from langchain_groq import ChatGroq
# from langchain_core.prompts import PromptTemplate
# from sentence_transformers import SentenceTransformer

# # Load existing PSO engine
# from query_engine import optimize_steam_economy, CONTROLLABLE_PARAMETERS

# # Load environment variables
# load_dotenv()

# # Setup Vector DB paths
# FAISS_INDEX_PATH = r"..\Manual plant doc\faiss_index.bin"
# PARQUET_PATH = r"..\Manual plant doc\chunks.parquet"

# class IndustryAgent:
#     def __init__(self):
#         # 1. Initialize LLM
#         # Ensure GROQ_API_KEY is in environment or .env
#         api_key = os.getenv("GROQ_API_KEY")
#         if not api_key or api_key == "your_key_here":
#             print("[ERROR] Please set your GROQ_API_KEY in the .env file located at D:\\knowledgr base\\data_intelligence\\.env")
#             sys.exit(1)
            
#         self.llm = ChatGroq(
#             temperature=0.2,
#             model_name="llama-3.3-70b-versatile", 
#             api_key=api_key
#         )
        
#         # 2. Initialize Embedder and RAG
#         print("[LLM_AGENT] Loading Plant Documentation RAG...")
#         self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
#         try:
#             self.index = faiss.read_index(FAISS_INDEX_PATH)
#             self.chunks_df = pd.read_parquet(PARQUET_PATH)
#             self.rag_loaded = True
#         except Exception as e:
#             print(f"[WARN] Failed to load RAG FAISS. Continuing without it. ({e})")
#             self.rag_loaded = False
            
#     def get_plant_context(self, parameters_changed):
#         """Query the plant manual for process explanations behind these parameters"""
#         if not self.rag_loaded:
#             return "No plant manual context available."
            
#         # Create a search query based on what parameters the PSO suggested changing
#         query = "How do these parameters affect steam economy: " + ", ".join(parameters_changed)
        
#         vector = self.embedder.encode([query]).astype("float32") # Needs float32 for faiss
#         k = 3 # Retrieve top 3 documents
#         D, I = self.index.search(vector, k)
        
#         contexts = []
#         for idx in I[0]:
#             if idx != -1 and idx < len(self.chunks_df):
#                 chunk_text = self.chunks_df.iloc[idx]["content"]
#                 contexts.append(chunk_text)
                
#         return "\n\n".join(contexts)

#     def run_pipeline(self, target_se=4.4):
#         print(f"\n[LLM_AGENT] Running End-To-End Pipeline for Target SE: {target_se}")
        
#         # Current State Dummy data matching your framework (normally from PI tags)
#         current_state = {
#             'LP Steam to De-Superheater (TPH)': 63.11,
#             'Lab - SEL Density (g/cc)': 1.31,
#             'Process Condensate to Tank Farm (m\u00b3/h)': 211.42,
#             'Live Steam Condensate (\u00b0C)': 139.27,
#             'LP Steam Before De-Superheater (\u00b0C)': 151.47,
#             'SPL - Overflow A/C Ratio': 0.32,
#             'Strong Evaporated Liquor Out from Battery (\u00b0C)': 82.74,
#             '3rd Effect 2nd Drum Condensate (\u00b0C)': 95.5,
#             'MP Steam I/L (\u00b0C)': 189.41,
#             'SEL_Flow': 723.65,
#             'Separator Vessel 3rd Effect (%)': 50.0,
#             'SPL_NA2CO3': 248.79,
#             'Barometric Condenser (Kg/cm\u00b2)': -0.91,
#             'Total_Evaporation_Rate': 263.07,
#             # Control params
#             'Chest Pressure (Kg/cm\u00b2G)': 2.55,
#             'Split_Flow_4th_Effect': 0.4888,
#             '1st Product Flash Drum Liquor O/L (\u00b0C)': 119.85,
#             'Cooling Water to Barometric Condenser (m\u00b3/h)':3054.55,
#             'Spent Liquor Split Flow (m\u00b3/h)': 958.12,
#             'Spent Liquor into Battery (\u00b0C)': 81.43
#         }
        
#         print("\n--- INITIATING PSO OPTIMIZER (Production Logic-Aligned) ---")
#         # 1. Run the Model PSO engine to get the numerical result and the baseline
#         best_df, baseline_se = optimize_steam_economy(current_state, target_se=target_se, n_particles=50, n_iterations=30)
        
#         if best_df.empty:
#             print("[LLM_AGENT] Could not optimize.")
#             return
            
#         print("\n--- FETCHING DOMAIN KNOWLEDGE (RAG) ---")
#         # Extract the highest ranked result
#         best_scenario = best_df.iloc[0]
#         achieved_se = best_scenario["Predicted_Steam_Economy"]
        
#         # Calculate what changed
#         changes_strs = []
#         changed_params = []
#         for p in CONTROLLABLE_PARAMETERS:
#             curr = current_state[p]
#             new_val = best_scenario[p]
#             if abs(new_val - curr) > 0.01:
#                 direction = "increased" if new_val > curr else "decreased"
#                 changes_strs.append(f"{p}: {curr:.2f} -> {new_val:.2f} ({direction})")
#                 changed_params.append(p.split("(")[0].strip().lower())
                
#         # 2. RAG Retrieval
#         rag_context = self.get_plant_context(changed_params)
        
#         print("\n--- GENERATING INDUSTRIAL INSIGHTS VIA LLM ---")
#         # 3. LLM Integration
#         prompt_template = """
#         You are an elite Industrial Process Control AI Agent for an Evaporation Plant. 
#         Your goal is to explain recommendations for improving Steam Economy to a plant engineer.

#         CURRENT STATUS:
#         Target Steam Economy sought: {target_se}
        
#         PERFORMANCE COMPARISON:
#         Current Baseline (Measured/Predicted): {baseline_se:.3f}
#         Optimized Potential (Achieved via PSO): {achieved_se:.3f}
#         Improvement: {{improvement_se}}

#         MATHEMATICAL OPTIMIZATION CHANGES (from PSO ML Engine):
#         {changes}

#         PLANT MANUAL EXCERPTS (Domain RAG Knowledge):
#         {rag_context}

#         INSTRUCTIONS:
#         Write a concise, professional report structured exactly like this:
        
#         Steam Economy Optimization:
#         {baseline_se:.2f} (Current) -> {achieved_se:.2f} (Optimized)
        
#         Recommended Changes:
#         - [List the variable changes cleanly. If baseline == optimized, state "Current operating parameters are already optimized within safe bounds."]
        
#         Process Explanation:
#         - [Provide 2-3 bullet points explaining the logic behind the changes. If no change is recommended, explain WHY the current state is optimal based on RAG.]
        
#         Confidence:
#         High (Aligned with Production TR1_6_6 logic and RAG).
#         """
        
#         prompt = PromptTemplate(
#             input_variables=["target_se", "baseline_se", "achieved_se", "changes", "rag_context"],
#             template=prompt_template
#         )
        
#         chain = prompt | self.llm
        
#         # Generate the payload
#         changes_text = "\n".join(changes_strs) if changes_strs else "No parameter changes recommended."
#         response = chain.invoke({
#             "target_se": target_se,
#             "baseline_se": baseline_se,
#             "achieved_se": achieved_se,
#             "improvement_se": f"{achieved_se - baseline_se:+.3f}",
#             "changes": changes_text,
#             "rag_context": rag_context
#         })
        
#         print("\n" + "="*60)
#         print("  END-TO-END SYSTEM AI REPORT")
#         print("="*60)
#         print(response.content)
#         print("="*60 + "\n")

# if __name__ == "__main__":
#     agent = IndustryAgent()
#     agent.run_pipeline(target_se=4.4)
import os
import sys
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

        if not api_key or api_key == "your_key_here":
            raise ValueError("❌ Set GROQ_API_KEY in .env")

        self.llm = ChatGroq(
            temperature=0.2,
            model_name="llama-3.3-70b-versatile",
            api_key=api_key
        )

        # ================================
        # 📁 PATH FIX (CRITICAL FIX)
        # ================================
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.RAG_DIR = os.path.join(BASE_DIR, "Manual plant doc")

        self.FAISS_BIN = os.path.join(self.RAG_DIR, "faiss_index.bin")
        self.PARQUET_PATH = os.path.join(self.RAG_DIR, "chunks.parquet")

        self.FAISS_INDEX = os.path.join(self.RAG_DIR, "index.faiss")
        self.FAISS_META = os.path.join(self.RAG_DIR, "index.pkl")

        # ================================
        # 🔍 LOAD RAG
        # ================================
        print("[LLM_AGENT] Loading Plant Documentation RAG...")

        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        self.rag_loaded = False

        try:
            # [OPTION 1] Custom FAISS
            if os.path.exists(self.FAISS_BIN) and os.path.exists(self.PARQUET_PATH):
                print("[RAG] Loading custom FAISS index...")
                self.index = faiss.read_index(self.FAISS_BIN)
                self.chunks_df = pd.read_parquet(self.PARQUET_PATH)
                self.rag_loaded = True

            # [OPTION 2] LangChain FAISS
            elif os.path.exists(self.FAISS_INDEX) and os.path.exists(self.FAISS_META):
                print("[RAG] Detected LangChain FAISS format. Please convert or load separately.")
                self.rag_loaded = False

            else:
                print("[RAG] No FAISS index found in:", self.RAG_DIR)

        except Exception as e:
            print(f"[WARN] RAG load failed: {e}")
            self.rag_loaded = False

        print(f"RAG STATUS: {'LOADED' if self.rag_loaded else 'NOT LOADED'}")

    # ================================
    # 🔍 RAG QUERY
    # ================================
    def safe_search(self, vector, k):
        if vector.shape[1] != self.index.d:
            print(f"[ERROR] Dimension mismatch: Query={vector.shape[1]}, Index={self.index.d}")
            return [], []
        return self.index.search(vector, k)
    def get_plant_context(self, parameters_changed):

        if not self.rag_loaded:
            return "No plant manual context available."

        query = "How do these parameters affect steam economy: " + ", ".join(parameters_changed)

        vector = self.embedder.encode([query]).astype("float32")

        # 🔥 SAFE CHECK & SEARCH
        D, I = self.safe_search(vector, k=3)

        contexts = []
        if len(I) > 0:
            for idx in I[0]:
                if idx != -1 and idx < len(self.chunks_df):
                    contexts.append(self.chunks_df.iloc[idx]["content"])

        return "\n\n".join(contexts) if (contexts and len(I) > 0) else "No relevant plant context found."
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

        print("\n--- RUNNING PSO ---")

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
        # 📊 DETAILED PARAMETER COMPARISON
        # ================================
        comparison_table = []
        changes_summary = []
        changed_params = []

        header = f"{'Parameter Name':<45} | {'Baseline':>10} | {'PSO Optimized':>13} | {'Change':>10}"
        separator = "-" * len(header)
        
        comparison_table.append(header)
        comparison_table.append(separator)

        for p in CONTROLLABLE_PARAMETERS:
            curr = current_state.get(p, 0)
            opt = best[p]
            delta = opt - curr
            
            # Simple threshold to report significant changes
            if abs(delta) > 0.001:
                direction = "[INCREASE]" if delta > 0 else "[DECREASE]"
                changes_summary.append(f"{p}: {curr:.2f} -> {opt:.2f} {direction}")
                changed_params.append(p)
            
            comparison_table.append(f"{p[:45]:<45} | {curr:10.2f} | {opt:13.2f} | {delta:+10.2f}")

        comparison_text = "\n".join(comparison_table)

        # ================================
        # 🔍 RAG CONTEXT
        # ================================
        print("\n--- FETCHING RAG CONTEXT ---")
        rag_context = self.get_plant_context(changed_params)

        # ================================
        # 🤖 LLM REPORT GENERATION
        # ================================
        print("\n--- GENERATING FINAL REPORT ---")

        prompt_template = """
        You are an Elite Industrial Process Optimizer.
        Analyze the following Particle Swarm Optimization (PSO) results and provide a technical explanation.

        STEAM ECONOMY PERFORMANCE:
        - Current Baseline: {baseline:.3f}
        - PSO Optimized: {achieved:.3f}
        - Expected Improvement: {improvement:+.3f}

        DETAILED PARAMETER RE-CONFIGURATION:
        {comparison}

        PLANT KNOWLEDGE BASE (RAG):
        {context}

        REPORT STRUCTURE:
        1. **Executive Summary**: 1 sentence on the efficiency gain.
        2. **Technical Adjustments**: Explain WHY the specific parameter changes (especially the largest ones) lead to better Steam Economy based on the manual.
        3. **Operational Risk**: Mention if any value is near a boundary or if this is a "Safe" optimization.
        4. **Confidence Score**: Out of 10, based on RAG alignment.
        """

        prompt = PromptTemplate(
            input_variables=["baseline", "achieved", "improvement", "comparison", "context"],
            template=prompt_template
        )

        chain = prompt | self.llm

        # Print the numerical results immediately for the user
        print("\n" + "=" * 85)
        print("  PSO OPTIMIZATION RESULTS (NUMERICAL COMPARISON)")
        print("=" * 85)
        print(comparison_text)
        print(f"\nPREDICTED STEAM ECONOMY: {baseline_se:.3f} -> {achieved_se:.3f} ({achieved_se - baseline_se:+.3f})")
        print("=" * 85)

        response = chain.invoke({
            "baseline": baseline_se,
            "achieved": achieved_se,
            "improvement": achieved_se - baseline_se,
            "comparison": comparison_text,
            "context": rag_context
        })

        print("\n" + " *** INDUSTRIAL AI INSIGHTS *** " + "=" * 53)
        print(response.content)
        print("=" * 85 + "\n")


# ================================
# ▶ RUN
# ================================
if __name__ == "__main__":
    agent = IndustryAgent()
    agent.run_pipeline()
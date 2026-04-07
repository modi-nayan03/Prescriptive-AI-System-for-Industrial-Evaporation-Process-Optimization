"""
query_engine.py
----------------
Prescriptive Optimizer for Steam Economy

Identifies optimal settings for CONTROLLABLE parameters to maximize
Steam Economy, while keeping FIXED parameters at their current real-time values.
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import argparse

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration & Mappings
# =============================================================================

MODEL_PATH = os.path.join("Model", "model (1).pkl")

# These are the exact column names the XGBoost model expects
MODEL_FEATURES = [
    'Spent Liquor into Battery (\u00b0C)',
    'LP Steam to De-Superheater (TPH)',
    'Lab - SEL Density (g/cc)',
    'Process Condensate to Tank Farm (m\u00b3/h)',
    'Live Steam Condensate (\u00b0C)',
    'Spent Liquor Split Flow (m\u00b3/h)',
    'LP Steam Before De-Superheater (\u00b0C)',
    'SPL - Overflow A/C Ratio',
    'Strong Evaporated Liquor Out from Battery (\u00b0C)',
    '3rd Effect 2nd Drum Condensate (\u00b0C)',
    'MP Steam I/L (\u00b0C)',
    'SEL_Flow',
    'Separator Vessel 3rd Effect (%)',
    '1st Product Flash Drum Liquor O/L (\u00b0C)',
    'Split_Flow_4th_Effect',
    'SPL_NA2CO3',
    'Barometric Condenser (Kg/cm\u00b2)',
    'Total_Evaporation_Rate',
    'Cooling Water to Barometric Condenser (m\u00b3/h)',
    'Chest Pressure (Kg/cm\u00b2G)'
]

CONTROLLABLE_PARAMETERS = [
    'Chest Pressure (Kg/cm\u00b2G)',
    'Split_Flow_4th_Effect',
    '1st Product Flash Drum Liquor O/L (\u00b0C)',
    'Cooling Water to Barometric Condenser (m\u00b3/h)',
    'Spent Liquor Split Flow (m\u00b3/h)',
    'Spent Liquor into Battery (\u00b0C)'
]

# Safe Operating Bounds for Controllable Parameters 
# (estimated from historical typical values to prevent extreme extrapolation)
BOUNDS = {
    'Chest Pressure (Kg/cm\u00b2G)': (1.2, 4.0), # Widened from 1.8
    'Split_Flow_4th_Effect': (0.0, 1.0),
    '1st Product Flash Drum Liquor O/L (\u00b0C)': (95.0, 145.0), # Widened from 130 max
    'Cooling Water to Barometric Condenser (m\u00b3/h)': (1500.0, 4000.0),
    'Spent Liquor Split Flow (m\u00b3/h)': (600.0, 1400.0), # Widened from 800-1200
    'Spent Liquor into Battery (\u00b0C)': (65.0, 110.0) # Widened from 95 max
}

MAX_CHANGE_CONSTRAINTS = {
    'Chest Pressure (Kg/cm\u00b2G)': 0.8,
    '1st Product Flash Drum Liquor O/L (\u00b0C)': 10.0,
    'Spent Liquor into Battery (\u00b0C)': 10.0,
    'Split_Flow_4th_Effect': 0.3,
    'Cooling Water to Barometric Condenser (m\u00b3/h)': 500.0,
    'Spent Liquor Split Flow (m\u00b3/h)': 200.0
}
MAX_IMPROVEMENT_PERCENTAGE = 0.04


# =============================================================================
# Optimizer Core
# =============================================================================

def load_ml_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"[ERROR] Model not found at {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


def optimize_steam_economy(current_state, target_se=4.3, n_particles=100, n_iterations=50):
    """
    Search for controllable parameter combinations that maximize Steam Economy
    using Particle Swarm Optimization (PSO).
    """
    model = load_ml_model()
    
    print(f"\n[STEP 1] Initializing PSO Swarm with {n_particles} particles...")
    
    dim = len(CONTROLLABLE_PARAMETERS)
    
    lower_bounds = []
    upper_bounds = []
    for col in CONTROLLABLE_PARAMETERS:
        safe_low, safe_high = BOUNDS[col]
        current_val = current_state.get(col)
        
        if current_val is not None and col in MAX_CHANGE_CONSTRAINTS:
            max_change = MAX_CHANGE_CONSTRAINTS[col]
            final_low = max(safe_low, current_val - max_change)
            final_high = min(safe_high, current_val + max_change)
            # Ensure we don't go outside the absolute bounds
            final_low = max(safe_low, final_low)
            final_high = min(safe_high, final_high)
        else:
            final_low, final_high = safe_low, safe_high
            
        lower_bounds.append(final_low)
        upper_bounds.append(final_high)
        
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)
    
    # Initialize positions randomly within bounds
    positions = np.random.uniform(lower_bounds, upper_bounds, (n_particles, dim))
    # Initialize velocities
    velocities = np.zeros((n_particles, dim))
    
    # Track personal bests and global best
    pbest_positions = positions.copy()
    pbest_scores = np.full(n_particles, -np.inf)
    
    gbest_position = positions[0].copy()
    gbest_score = -np.inf
    
    # Standard PSO Hyperparameters from TR1_6_6.py
    w = 0.7      # Inertia weight
    c1 = 2.0     # Cognitive constant
    c2 = 2.0     # Social constant
    
    # Pre-allocate DataFrame for fast prediction
    df_search = pd.DataFrame([current_state] * n_particles, columns=MODEL_FEATURES)
    for col in MODEL_FEATURES:
        if col not in current_state:
            df_search[col] = 0.0
            
    # Calculate baseline steam economy and target evaporation for objective
    df_base = df_search.iloc[[0]].copy()
    baseline_se = model.predict(df_base)[0, 0]
    max_allowed_se = baseline_se * (1.0 + MAX_IMPROVEMENT_PERCENTAGE)
    
    # Target Evaporation Rate (from current state if available, else baseline)
    evap_target = current_state.get('Total_Evaporation_Rate', 263.0)
            
    print(f"[STEP 2] Running PSO for {n_iterations} iterations... (Current Baseline SE: {baseline_se:.3f})")
    
    all_evaluated = []
    
    for i in range(n_iterations):
        # Update search dataframe with current swarm positions
        for j, col in enumerate(CONTROLLABLE_PARAMETERS):
            df_search[col] = positions[:, j]
            
        # Predict using XGBoost MultiOutputRegressor (Steam Economy is at index 0)
        predictions = model.predict(df_search)
        raw_se = predictions[:, 0] 
        
        # Primary Objective: Minimize LP Steam (TPH) = Evaporation / SE
        # Guard against zero/negative SE
        safe_se = np.where(raw_se > 1e-6, raw_se, 1e-6)
        lp_steam_tph = evap_target / safe_se
        
        # Apply penalty if steam economy exceeds realistic improvement (4% cap)
        over_cap = np.maximum(0.0, raw_se - max_allowed_se)
        penalty = 1e4 * over_cap
        
        # We are MINIMIZING cost (lp_steam_tph + penalty)
        costs = lp_steam_tph + penalty
        
        # PSO logic expects maximization of score? (My manual PSO code below does scores > pbest_scores)
        # So we convert cost to a score by negating it
        scores = -costs
        
        # Clamp extreme SE to max_allowed_se for reporting
        clamped_se = np.where(raw_se > max_allowed_se, max_allowed_se, raw_se)
        
        df_eval = df_search.copy()
        df_eval["Predicted_Steam_Economy"] = clamped_se
        all_evaluated.append(df_eval)
        
        # Update personal bests
        better_mask = scores > pbest_scores
        pbest_scores[better_mask] = scores[better_mask]
        pbest_positions[better_mask] = positions[better_mask]
        
        # Update global best
        current_gbest_idx = np.argmax(scores)
        if scores[current_gbest_idx] > gbest_score:
            gbest_score = scores[current_gbest_idx]
            gbest_position = positions[current_gbest_idx].copy()
            
        # Update velocities
        r1 = np.random.rand(n_particles, dim)
        r2 = np.random.rand(n_particles, dim)
        
        velocities = (w * velocities + 
                      c1 * r1 * (pbest_positions - positions) + 
                      c2 * r2 * (gbest_position - positions))
                      
        # Update positions & enforce bounds
        positions = positions + velocities
        positions = np.clip(positions, lower_bounds, upper_bounds)
        
        if (i+1) % 10 == 0:
            # For logging, find the actual best SE in the current scores
            current_best_se = np.max(clamped_se)
            print(f"  -> Iteration {i+1}: Best Predicted Steam Economy - {current_best_se:.3f}")

    # Final best SE calculation
    final_best_se = evap_target / (-gbest_score) if gbest_score != 0 else baseline_se
    print(f"[INFO] PSO Optimization Completed. Supreme SE found: {final_best_se:.3f}")
    
    # Aggregate all iterations to find the diverse top scenarios
    df_all = pd.concat(all_evaluated, ignore_index=True)
    
    # Filter out near-identical configurations to give distinct options
    temp_df = df_all.copy()
    for col in CONTROLLABLE_PARAMETERS:
        # round up to significant digits to identify practically identical inputs
        temp_df[col] = temp_df[col].round(1) 
        
    df_unique = temp_df.drop_duplicates(subset=CONTROLLABLE_PARAMETERS)
    # Restore actual unrounded values for the chosen unique combinations
    df_unique = df_all.loc[df_unique.index] 
    
    # Baseline Cost (Minimization objective)
    baseline_cost = evap_target / (baseline_se if baseline_se > 1e-6 else 1e-6)
    
    # Fallback logic: If best result is not an improvement (gbest_score is -cost)
    if gbest_score <= -baseline_cost:
        print("[INFO] PSO could not find an improvement within safe constraints. Returning current baseline.")
        # Create a single-row DF with the baseline state
        best_scenarios = df_base.copy()
        best_scenarios["Predicted_Steam_Economy"] = baseline_se
    else:
        # Filter for actual improvements
        successful = df_unique[df_unique["Predicted_Steam_Economy"] >= target_se]
        
        if successful.empty:
            print(f"\n[INFO] PSO concluded. Could not reach target {target_se}, but found improvement.")
            best_scenarios = df_unique.sort_values("Predicted_Steam_Economy", ascending=False).head(3)
        else:
            print(f"[INFO] Found multiple configurations reaching target SE >= {target_se}")
            best_scenarios = successful.sort_values("Predicted_Steam_Economy", ascending=False).head(3)
        
    return best_scenarios, baseline_se


# =============================================================================
# CLI / Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Steam Economy Optimizer")
    parser.add_argument("--target", type=float, default=4.3, help="Target Steam Economy (default: 4.3)")
    args = parser.parse_args()

    print("================================================================")
    print("  STEAM ECONOMY PRESCRIPTIVE OPTIMIZER")
    print("================================================================")

    # Simulated "Current State" for FIXED parameters
    # (In a real system, you'd pull these from PI tags / sensors)
    current_state = {
        'LP Steam to De-Superheater (TPH)': 63.11,
        'Lab - SEL Density (g/cc)': 1.31,
        'Process Condensate to Tank Farm (m\u00b3/h)': 211.42,
        'Live Steam Condensate (\u00b0C)': 139.27,
        'LP Steam Before De-Superheater (\u00b0C)': 151.47,
        'SPL - Overflow A/C Ratio': 0.32,
        'Strong Evaporated Liquor Out from Battery (\u00b0C)': 82.74,
        '3rd Effect 2nd Drum Condensate (\u00b0C)': 95.5,
        'MP Steam I/L (\u00b0C)': 189.41,
        'SEL_Flow': 723.65,
        'Separator Vessel 3rd Effect (%)': 50.0,
        'SPL_NA2CO3': 248.79,
        'Barometric Condenser (Kg/cm\u00b2)': -0.91,
        'Total_Evaporation_Rate': 263.07
    }
    
    # Estimate current baseline (including current suboptimal controllables)
    baseline_state = current_state.copy()
    baseline_state.update({
        'Chest Pressure (Kg/cm\u00b2G)': 2.55,  # User mentioned they are at 2.1
        'Split_Flow_4th_Effect': 0.4888,
        '1st Product Flash Drum Liquor O/L (\u00b0C)': 119.85,
        'Cooling Water to Barometric Condenser (m\u00b3/h)':3054.55,
        'Spent Liquor Split Flow (m\u00b3/h)': 958.12,
        'Spent Liquor into Battery (\u00b0C)': 81.43
    })
    
    # Get baseline prediction
    model = load_ml_model()
    df_base = pd.DataFrame([baseline_state], columns=MODEL_FEATURES)
    # Fill missing with 0s
    for c in MODEL_FEATURES: 
        if c not in df_base.columns: df_base[c] = 0.0
        
    current_se = model.predict(df_base[MODEL_FEATURES])[0, 0]
    
    print(f"\n[CURRENT STATE] Predicted Steam Economy: {current_se:.3f}")
    print(f"Current Chest Pressure: {baseline_state['Chest Pressure (Kg/cm\u00b2G)']}")
    
    # Run PSO optimization (passing baseline_state which includes controllable values)
    best_df, baseline_se = optimize_steam_economy(baseline_state, target_se=args.target, n_particles=100, n_iterations=50)
    
    print("\n================================================================")
    print(f"  RECOMMENDED ACTIONS (PSO OPTIMIZER) TO REACH STEAM ECONOMY >= {args.target}")
    print("================================================================")
    
    for idx, (_, row) in enumerate(best_df.iterrows(), 1):
        print(f"\n--- PSO OPTION {idx} ---")
        print(f"Achievable Steam Economy : {row['Predicted_Steam_Economy']:.3f} (Maximized via PSO)")
        print(f"\nAdjust your Controllable Levers to these precise values:")
        
        for p in CONTROLLABLE_PARAMETERS:
            curr_val = baseline_state.get(p, 0)
            new_val = row[p]
            diff = new_val - curr_val
            trend = "(INCREASE)" if diff > 0 else "(DECREASE)"
            
            print(f"  * {p[:45]:<45} : {new_val:<8.2f} {trend}")
            
        print("-" * 65)


if __name__ == "__main__":
    main()
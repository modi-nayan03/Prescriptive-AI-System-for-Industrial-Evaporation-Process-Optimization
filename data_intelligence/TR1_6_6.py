import os
import sys
import json
import subprocess
import tempfile
import requests
import urllib3
import logging
import time
import re
import pickle
import atexit
import numpy as np
import pandas as pd
import mlflow
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from logging import StreamHandler
# from pyswarm import pso  # PSO optimizer
import pyswarms as ps
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed  # OPTIMIZATION: Added for parallel I/O

os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# ============================================================================
# JSON / SUBPROCESS HELPERS
# ============================================================================

def _json_default(o):
    """Helper for JSON encoding of numpy and other non-standard types."""
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def run_pso_in_subprocess(config, plant_data, timeout_seconds: int = 1800):
    """
    Run the PSO optimization in a separate Python subprocess.

    This isolates heavy PSO work from the Airflow worker process and ensures
    the child process is always waited on (no OS-level zombies).
    """
    # Prepare temp input/output files
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as in_f:
        input_path = in_f.name
        json.dump({"config": config, "plant_data": plant_data}, in_f, default=_json_default)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as out_f:
        output_path = out_f.name

    cmd = [sys.executable, os.path.abspath(__file__), "--transform-once", input_path, output_path]
    logging.info(f"Spawning PSO subprocess: {cmd}")

    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )

        if completed.returncode != 0:
            logging.error(
                "PSO subprocess failed "
                f"(returncode={completed.returncode})\n"
                f"STDOUT:\n{completed.stdout}\n"
                f"STDERR:\n{completed.stderr}"
            )
            raise RuntimeError(f"PSO subprocess failed with code {completed.returncode}")

        with open(output_path, "r") as f:
            optimization_results = json.load(f)

        return optimization_results

    finally:
        # Best-effort cleanup of temp files
        for path in (input_path, output_path):
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass


# ============================================================================
# RESOURCE MANAGEMENT
# ============================================================================

@contextmanager
def requests_session():
    """Context manager for requests session to ensure cleanup"""
    session = requests.Session()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def safe_execution():
    """Context manager for safe execution with cleanup"""
    try:
        yield
    except Exception as e:
        logging.error(f"Error during execution: {e}")
        raise
    finally:
        # Cleanup any remaining resources
        pass


# ============================================================================
# LOGGING SETUP
# ============================================================================

class ISTFormatter(logging.Formatter):
    """Custom formatter for IST timezone logging"""
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, ZoneInfo("Asia/Kolkata"))
        return dt.strftime(datefmt if datefmt else '%Y-%m-%d %H:%M:%S %Z')


def log_execution_time(func):
    """Decorator to log function execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logging.info(f" {func.__name__} completed in {elapsed:.2f} seconds")
        return result
    return wrapper


def setup_logging():
    """Configure logging with IST timezone"""
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    log_format = '%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
    log_formatter = ISTFormatter(log_format)

    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    stream_handler = StreamHandler()
    stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(stream_handler)
    root_logger.setLevel(logging.INFO)


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

TAG_TO_PARAM = {
    # 6/6 Effect Features
    "HIL_AL_UTKL_REF_WA_EVAP_TRAIN_1_HOT_WELL_PMP_PU550_101A/PU550_101B/PU550_101S_COOL_WTR_INL_TEMP": "Cooling Tower Water Intel Temperature",
    "HIL_AL_UTKL_REF_WA_EVAP_TRAIN_1_SPL_TNK_TK261_101/TK261_201_BTRY_TEMP": "Spent Liquor Feed Pump Battery Temperature",
    "HIL_AL_UTKL_REF_WA_EVAP_TRAIN_1_DE_SUPR_HTR_DS260_110_EVAPTR_LP_STM_FL": "De-Superheater LP Steam Flow",
    "HIL_AL_UTKL_LAB_RA_EVAP_IN_PROC_RA_TRAIN_01_SEL_DEN": "SEL Density",
    "HIL_AL_UTKL_REF_WA_EVAP_TRAIN_1_PROC_CNDS_DRM_05_EFF_VE260_150_PMP_FL": "Process Condensate to Tank Farm",
    "HIL_AL_UTKL_REF_WA_EVAP_TRAIN_1_01_EFF_STM_CNDS_DRM_VE260_112_STM_CNDS_CGPP_TEMP": "Live Steam Condensate Temperature",
    "HIL_AL_UTKL_REF_WA_EVAP_TRAIN_1_TOTL_FD_FL_VE260_161/EX260_140_SPL_FL": "SPL Flow",
    "HIL_AL_UTKL_REF_WA_EVAP_TRAIN_1_BARO_CNDSR_VE260_170_COOL_WTR_FL": "Barometric Condensor Cooling Water Flow",
    "HIL_AL_UTKL_REF_WA_EVAP_TRAIN_1_DE_SUPR_HTR_DS260_110_LP_STM_TEMP_BFR_DE_SUPR_HTR": "LP Steam Temperature Before De-Superheater",
    "HIL_AL_UTKL_LAB_RA_EVAP_IN_PROC_RA_TRAIN_01_SEL_A_C": "SPL A/C",
    "HIL_AL_UTKL_REF_WA_EVAP_TRAIN_1_SEL_DISCH_PMP_PU260_142A/PU260_142B_TEMP": "Strong Evaporated Liquor Discharge Pump Temperature",
    "HIL_AL_UTKL_REF_WA_EVAP_TRAIN_1_PROC_CNDS_DRM_03_EFF_VE260_134_DRM_CNDS_TEMP": "3rd Effect 2nd Drum Condensate",
    "HIL_AL_UTKL_REF_WA_EVAP_TRAIN_1_EJECT_EJ260_171_MP_STM_INL_TEMP": "MP Steam Inlet Temperature",
    "HIL_AL_UTKL_REF_WA_EVAP_TR01_SEL_FL_CALC": "SEL Flow",
    "HIL_AL_UTKL_REF_WA_EVAP_TRAIN_1_SEPR_VESL_SE260_130_VESL_LVL": "Separator Vessel 3rd Effect",
    "HIL_AL_UTKL_REF_WA_EVAP_TRAIN_1_01_PRDCT_FLSH_DRM_VE260_122_LIQR_OUTL_TEMP": "1st Product Flash Drum Liquor O/L Temperature",
    "HIL_AL_UTKL_REF_WA_EVAP_TRAIN_1_SPITE_FLOW_6_EFFECT": "Split Flow in 4th Effect",
    "HIL_AL_UTKL_LAB_WA_SEED_PREP_IN_PROC_WA_SPL_OVRFLW_NA2CO3": "SPL Na2co3",
    "HIL_AL_UTKL_REF_WA_EVAP_TRAIN_1_HOT_WELL_PMP_PU550_101A/PU550_101B/PU550_101S_COOL_WTR_OUTL_TEMP": "Cooling Tower Water Outlet Temperature",
    "HIL_AL_UTKL_REF_WA_EVAP_TRAIN_1_FALLNG_FILM_EVAPTR_06_EFF_EX260_160_LIQR_VPOR_TEMP": "6th Effect vapor",
    "HIL_AL_UTKL_LAB_RA_EVAP_IN_PROC_RA_TRAIN_01_SEL_NA2CO3": "SEL Na2CO3",
    "HIL_AL_UTKL_REF_WA_EVAP_TRAIN_1_BARO_CNDSR_VE260_170_PRESS": "Barometric Condensor Vacuum",
    "HIL_AL_UTKL_REF_WA_EVAP_TRAIN_1_COOL_WTR_HEAT_RATE_CALC": "Cooling Water Heat",
    # 5/5 Effect Extra
    "HIL_AL_UTKL_REF_WA_EVAP_TRAIN_1_DE_SUPR_HTR_DS260_110_LP_STM_FL_CALC_PRESS_CTRL": "Chest Pressure",
    "HIL_AL_UTKL_REF_WA_EVAP_TRAIN_1_03_PRDCT_FLSH_DRM_VE260_142_LIQR_OUTL_TEMP": "3rd Product Flash Drum Liquor Outlet Temperature",
    "HIL_AL_UTKL_REF_WA_EVAP_TRAIN_1_SEPR_VESL_SE260_150_VESL_LVL": "Separator Vessel 5th Effect Level",
    "HIL_AL_UTKL_REF_WA_EVAP_TRAIN_1_EFFECT_OUT": "Train 1 Effect Out"
}

PARAM_TO_TAG = {v: k for k, v in TAG_TO_PARAM.items()}

LAB_DATA_PARAMS = ["SEL Density", "SPL A/C", "SPL Na2co3", "SEL Na2CO3"]

SAFE_OPERATING_BOUNDS = {
    "Spent Liquor Feed Pump Battery Temperature": (65, 90),
    "De-Superheater LP Steam Flow": (40, 80),
    "SEL Density": (1.29, 1.31),
    "Process Condensate to Tank Farm": (140, 260),
    "Live Steam Condensate Temperature": (110, 145),
    "SPL Flow": (920, 1100),
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
    "SPL Na2co3": (240, 255),
    "Cooling Tower Water Outlet Temperature": (40, 45),
    "6th Effect vapor": (40, 52),
    "Barometric Condensor Vacuum": (-0.91, -0.925),
    "Chest Pressure": (1.2, 2.9),
    "3rd Product Flash Drum Liquor Outlet Temperature": (80, 89),
    "Separator Vessel 5th Effect Level": (45, 55),
    "Cooling Water Heat": (70, 120),
    "Evaporation Rate": (200, 350),
    "Cooling Tower Water Intel Temperature": (20, 35)
}

ML_MODEL_FEATURES_6_EFFECT = [
    "Spent Liquor Feed Pump Battery Temperature", "De-Superheater LP Steam Flow", "SEL Density",
    "Process Condensate to Tank Farm", "Live Steam Condensate Temperature", "SPL Flow",
    "LP Steam Temperature Before De-Superheater", "SPL A/C",
    "Strong Evaporated Liquor Discharge Pump Temperature", "3rd Effect 2nd Drum Condensate",
    "MP Steam Inlet Temperature", "SEL Flow", "Separator Vessel 3rd Effect",
    "1st Product Flash Drum Liquor O/L Temperature", "Split Flow in 4th Effect", "SPL Na2co3",
    "Barometric Condensor Vacuum", "Barometric Condensor Cooling Water Flow", "Chest Pressure",
    "Evaporation Rate"
]

ML_MODEL_FEATURES_5_EFFECT = [
    "Cooling Tower Water Intel Temperature", "Spent Liquor Feed Pump Battery Temperature",
    "De-Superheater LP Steam Flow", "SEL Density", "Process Condensate to Tank Farm",
    "Live Steam Condensate Temperature", "SPL Flow", "Barometric Condensor Cooling Water Flow",
    "LP Steam Temperature Before De-Superheater", "SPL A/C",
    "Strong Evaporated Liquor Discharge Pump Temperature", "Chest Pressure",
    "3rd Product Flash Drum Liquor Outlet Temperature", "Separator Vessel 5th Effect Level",
]

EFFECT_MODE_CONFIG = {
    "6_EFFECT": {
        "MLFLOW_TRACKING_URI": "mlflow-artifacts:/812180679983349338/cfb765864ec64e20ac89c4dde83ef6a4/artifacts/model/model.pkl",
        "features": ML_MODEL_FEATURES_6_EFFECT,
        "name": "6/6 Effect Mode"
    },
    "5_EFFECT": {
        "MLFLOW_TRACKING_URI": "mlflow-artifacts:/812180679983349338/2b5d2b34b5484f319e25920fd92a6655/artifacts/models/final_model.pkl",
        "features": ML_MODEL_FEATURES_5_EFFECT,
        "name": "5/5 Effect Mode"
    }
}

ACTUAL_VALUE_TAGS = {
    "Actual Steam Economy": "HIL_AL_UTKL_REF_WA_EVAP_CALC_TR01_PC_STM_ECO",
    "Actual Evaporation Rate": "HIL_AL_UTKL_REF_WA_EVAP_CALC_TR01_PC_EVAP_RTE",
    "Actual SEL Na2CO3": "HIL_AL_UTKL_LAB_RA_EVAP_IN_PROC_RA_TRAIN_01_SEL_NA2CO3"
}

KPI_PREDICTION_TAGS = {
    "ML Predicted Steam Economy": "HIL_AL_UTKL_REF_WA_EVAP_CALC_TR01_PC_STM_ECO_PRED",
    "ML Predicted SEL Na2CO3": "HIL_AL_UTKL_LAB_RA_IN_PROC_RA_TRAIN_01_SEL_NA2CO3_PRED",
    "ML Predicted Evaporation Rate": "HIL_AL_UTKL_REF_WA_EVAP_CALC_TR01_PC_EVAP_RTE_PRED"
}

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
    "SPL Na2co3",
    "Barometric Condensor Vacuum"   # Fixed at -0.91; cannot be raised on this plant
]

MAX_CHANGE_CONSTRAINTS = {
    "Chest Pressure": 0.3,
    "1st Product Flash Drum Liquor O/L Temperature": 0.5,
    "Spent Liquor Feed Pump Battery Temperature": 0.1,
    "SPL Flow": 2.0,
    # "Barometric Condensor Vacuum": 0.01,
    "Split Flow in 4th Effect": 0.05,
    "Barometric Condensor Cooling Water Flow": 50.0
}

CONTROLLABLE_PARAMETERS = [
    "Chest Pressure",
    "Split Flow in 4th Effect",
    "1st Product Flash Drum Liquor O/L Temperature",
    "Barometric Condensor Cooling Water Flow",
    "SPL Flow",
    "Spent Liquor Feed Pump Battery Temperature"
]

EXCLUDE_FROM_PRED_WRITE = {
    "De-Superheater LP Steam Flow"
}

# PSO-specific parameters
PSO_CONFIG = {
    "swarmsize": 30,           # Number of particles in swarm
    "maxiter": 50,             # Maximum iterations
    "omega": 0.7,              # Inertia weight (particle velocity)
    "phip": 2.0,               # Cognitive parameter (personal best)
    "phig": 2.0,               # Social parameter (global best)
    "minstep": 1e-6,           # Minimum step size for convergence
    "minfunc": 1e-6,           # Minimum objective function change
    "debug": True              # Enable debug output
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PIDataPoint:
    """Data structure for PI data point"""
    value: float
    timestamp: str
    quality: str = "Good"
    age_hours: float = 0.0


# ============================================================================
# PLANT DATA HANDLER
# ============================================================================

class PlantDataHandler:
    """Handles all PI Web API interactions with robust fallback mechanisms"""

    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.pi_auth = (config['PI_USERNAME'], config['PI_PASSWORD'])
        self.tag_api_map: Dict[str, str] = {}
        self.data_cache: Dict[str, PIDataPoint] = {}
        self.session = None

        logging.info("Initializing PI Web API connection...")
        self._initialize_session()
        self._initialize_tag_mapping()
        logging.info(f" Tag mapping initialized with {len(self.tag_api_map)} tags")

        # Register cleanup on exit
        atexit.register(self.cleanup)

    def _initialize_session(self):
        """Initialize requests session with proper configuration"""
        self.session = requests.Session()
        self.session.auth = self.pi_auth
        self.session.verify = False
        # Set reasonable timeout defaults
        adapter = requests.adapters.HTTPAdapter(
            max_retries=3,
            pool_connections=10,
            pool_maxsize=20
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

    def cleanup(self):
        """Cleanup method to close session and free resources"""
        try:
            if self.session:
                self.session.close()
                logging.info("Session closed successfully")
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")

    def _initialize_tag_mapping(self):
        """Fetch and map all PI tags from Element"""
        url = self.config['PI_ELEMENT_URL']
        max_retries = 3

        for attempt in range(max_retries):
            try:
                logging.info(f"Fetching tags from PI Element (Attempt {attempt + 1}/{max_retries})...")
                response = self.session.get(url, timeout=30)
                response.raise_for_status()

                attributes = response.json().get("Items", [])

                for item in attributes:
                    config_string = item.get("ConfigString", "")
                    match = re.search(r'\\\\[^\\]*\\([^\?]+)', config_string)
                    if match:
                        tag_name = match.group(1)
                        self.tag_api_map[tag_name] = item.get("Links", {}).get("Value")

                if not self.tag_api_map:
                    logging.warning("No tags mapped from PI Element. Check permissions.")

                return

            except requests.exceptions.RequestException as e:
                logging.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    raise ConnectionError(f"Failed to initialize PI connection after {max_retries} attempts") from e

    def _get_tag_url(self, tag_name: str) -> Optional[str]:
        """Get URL for a tag from dynamic mapping"""
        return self.tag_api_map.get(tag_name)

    def _extract_numeric_values(self, items: List[Dict]) -> List[float]:
        """Extract numeric values from PI API response items"""
        values = []
        for item in items:
            val = item.get("Value")
            if isinstance(val, dict):
                val = val.get("Value")
            if isinstance(val, (int, float)) and np.isfinite(val):
                values.append(float(val))
        return values

    def _calculate_data_age(self, timestamp_str: str) -> float:
        """Calculate data age in hours from ISO timestamp"""
        try:
            ts_obj = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            now = datetime.now(ts_obj.tzinfo)
            return (now - ts_obj).total_seconds() / 3600
        except Exception as e:
            logging.debug(f"Could not parse timestamp: {e}")
            return 999.0

    def get_current_value_with_fallback(self, param_name: str, read_url: str) -> Optional[float]:
        """
        Industrial-grade data fetching with safe fallback strategy.

        IMPORTANT:
        - Never returns fake numeric values (like 0.0)
        - Returns None if no trustworthy data is available
        """

        is_lab = param_name in LAB_DATA_PARAMS
        start_time = "*-16h" if is_lab else "*-10min"
        duration = "16h" if is_lab else "10min"

        # OPTIMIZATION: Pre-compute URLs once instead of multiple .replace() calls
        summary_url = read_url.replace("/value", "/summary")
        recorded_url = read_url.replace("/value", "/recorded")

        # -------------------------
        # Strategy 1: Recent Summary
        # -------------------------
        try:
            window = "16h" if is_lab else "10m"
        
            params = {
                "summaryType": "Average",
                "startTime": f"*-{window}",
                "endTime": "*",
                "calculationBasis": "TimeWeighted"
            }
        
            resp = self.session.get(summary_url, params=params, timeout=15)
            resp.raise_for_status()
        
            items = resp.json().get("Items", [])
            values = self._extract_numeric_values(items)
        
            if values:
                avg = float(values[0])  # summary returns single aggregated value
                logging.info(f"[Summary-{window}] {param_name} = {avg:.3f}")
                return avg
        
        except Exception as e:
            logging.warning(f"[Summary Failed] {param_name}: {str(e)[:120]}")

        # -------------------------
        # Strategy 2: Recorded 6H
        # -------------------------
        try:
            params = {"startTime": "*-6h", "endTime": "*"}
            resp = self.session.get(recorded_url, params=params, timeout=15)
            resp.raise_for_status()

            values = self._extract_numeric_values(resp.json().get("Items", []))
            if values:
                avg = float(np.mean(values))
                logging.info(f"[Recorded-6H] {param_name} = {avg:.3f}")
                return avg

        except Exception as e:
            logging.warning(f"[6H Failed] {param_name}: {str(e)[:100]}")

        # -------------------------
        # Strategy 3: Recorded 24H
        # -------------------------
        try:
            params = {"startTime": "*-24h", "endTime": "*"}
            resp = self.session.get(recorded_url, params=params, timeout=15)
            resp.raise_for_status()

            values = self._extract_numeric_values(resp.json().get("Items", []))
            if values:
                avg = float(np.mean(values))
                logging.warning(f"[Recorded-24H] {param_name} = {avg:.3f} (STALE)")
                return avg

        except Exception as e:
            logging.warning(f"[24H Failed] {param_name}: {str(e)[:100]}")

        # -------------------------
        # Strategy 4: Recorded 7 Days (last valid value)
        # -------------------------
        try:
            params = {
                "startTime": "*-7d",
                "endTime": "*",
                "maxCount": 100,
                "sortField": "Timestamp",
                "sortOrder": "Descending"
            }
            resp = self.session.get(recorded_url, params=params, timeout=20)
            resp.raise_for_status()

            for item in resp.json().get("Items", []):
                val = item.get("Value")
                ts = item.get("Timestamp")

                if isinstance(val, (int, float)) and np.isfinite(val):
                    age_hours = self._calculate_data_age(ts)
                    logging.warning(
                        f"[7-Day Fallback] {param_name} = {val:.3f} "
                        f"({age_hours:.1f}h old)"
                    )
                    return float(val)

        except Exception as e:
            logging.error(f"[7-Day Failed] {param_name}: {str(e)[:100]}")

        # -------------------------
        # Strategy 5: NO DATA → RETURN None (CRITICAL)
        # -------------------------
        logging.critical(
            f"NO VALID DATA for '{param_name}'. "
            f"Returning None (optimizer will handle safely)."
        )
        return None


    def detect_effect_mode(self) -> str:
        """Detect current plant operating mode (5/5 or 6/6 effect)"""
        mode_tag = "HIL_AL_UTKL_REF_WA_EVAP_TRAIN_1_EFFECT_OUT"
        read_url = self._get_tag_url(mode_tag)

        if not read_url:
            logging.warning(f"Mode tag '{mode_tag}' not found. Defaulting to 6/6 Effect.")
            return "6_EFFECT"

        try:
            resp = self.session.get(read_url, timeout=10)
            resp.raise_for_status()
            mode_value = resp.json().get("Value")

            if mode_value is None or not isinstance(mode_value, (int, float)):
                logging.warning("Could not read mode value. Defaulting to 6/6 Effect.")
                return "6_EFFECT"

            detected_mode = "5_EFFECT" if int(mode_value) == 1 else "6_EFFECT"
            logging.info(f" Detected Mode: {EFFECT_MODE_CONFIG[detected_mode]['name']}")
            return detected_mode

        except Exception as e:
            logging.error(f"Mode detection failed: {e}. Defaulting to 6/6 Effect.")
            return "6_EFFECT"

    @log_execution_time
    def get_current_plant_state_for_mode(self, effect_mode: str) -> Tuple[Dict, Dict]:
        """Fetch all required plant data for the specified operating mode"""
        mode_config = EFFECT_MODE_CONFIG[effect_mode]
        mode_features = mode_config['features']

        logging.info(f"Fetching data for {mode_config['name']} ({len(mode_features)} parameters)...")

        # Fetch actual KPI values
        actual_outputs = {}
        for name, tag in ACTUAL_VALUE_TAGS.items():
            read_url = self._get_tag_url(tag)
            if read_url:
                actual_outputs[name] = self.get_current_value_with_fallback(name, read_url)
            else:
                logging.error(f"KPI tag '{tag}' not found in mapping.")
                actual_outputs[name] = 0.0

        # Fetch input parameters
        current_inputs = {}
        for param in mode_features:
            if param == "Evaporation Rate":
                continue

            tag = PARAM_TO_TAG.get(param)
            if tag:
                read_url = self._get_tag_url(tag)
                if read_url:
                    try:
                        value = self.get_current_value_with_fallback(param, read_url)
                        current_inputs[param] = value if value is not None else 0.0
                    except Exception as e:
                        logging.error(f"Failed to fetch {param}: {e}")
                        current_inputs[param] = 0.0
                else:
                    logging.warning(f"No read URL found for parameter: {param}")
                    current_inputs[param] = 0.0
            else:
                logging.warning(f"Parameter '{param}' not found in TAG mapping")
                current_inputs[param] = 0.0

        return actual_outputs, current_inputs

    def write_prediction_to_pi(self, write_url: str, value: float,
                              param_name: str, max_retries: int = 3) -> bool:
        """Write prediction value to PI with retry logic"""
        if value is None or not np.isfinite(value):
            logging.error(f"Invalid value for '{param_name}': {value}")
            return False

        value = float(value)

        if not write_url or not write_url.startswith('http'):
            logging.error(f"Invalid write URL for '{param_name}'")
            return False

        payload = {"Timestamp": "*", "Value": value}

        for attempt in range(max_retries):
            try:
                resp = self.session.post(write_url, json=payload, timeout=15)

                if 200 <= resp.status_code < 300:
                    logging.info(f" Wrote {param_name} = {value:.4f}")
                    return True
                else:
                    logging.error(
                        f"Write failed for '{param_name}' (Attempt {attempt + 1}/{max_retries}). "
                        f"Status: {resp.status_code}"
                    )

            except requests.exceptions.RequestException as e:
                logging.error(f"Write error '{param_name}' (Attempt {attempt + 1}): {e}")

            if attempt < max_retries - 1:
                time.sleep(2)

        logging.critical(f" FAILED to write '{param_name}' after {max_retries} attempts")
        return False

    def get_live_value_only(self, param_name: str, read_url: str) -> Optional[float]:
        """Fetch ONLY live value (no fallback). Used for plant ON/OFF checks."""
        try:
            resp = self.session.get(read_url, timeout=10)
            resp.raise_for_status()

            val = resp.json().get("Value")

            if isinstance(val, dict):
                val = val.get("Value")

            if isinstance(val, (int, float)) and np.isfinite(val):
                logging.info(f"LIVE CHECK  {param_name} = {val:.2f}")
                return float(val)

        except Exception as e:
            logging.error(f"LIVE CHECK FAILED for {param_name}: {e}")

        return None


# ============================================================================
# OPTIMIZATION ENGINE (PSO-based)
# ============================================================================

class EvaporationPlantOptimizer:
    """Main optimization engine for evaporation plant using PSO"""

    def __init__(self, config: Dict[str, str], force_mode: Optional[str] = None):
        self.config = config
        self.target_evaporation_rate: Optional[float] = None
        self.data_handler = None
        self.ml_model = None
        self.cleanup_registered = False

        logging.info("="*80)

        os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = 'true'
        os.environ["MLFLOW_TRACKING_USERNAME"] = self.config['MLFLOW_USERNAME']
        os.environ["MLFLOW_TRACKING_PASSWORD"] = self.config['MLFLOW_PASSWORD']
        mlflow.set_tracking_uri(self.config['MLFLOW_TRACKING_URI'])

        self.data_handler = PlantDataHandler(config)

        if force_mode:
            self.effect_mode = force_mode
        else:
            self.effect_mode = self.data_handler.detect_effect_mode()

        self.mode_config = EFFECT_MODE_CONFIG[self.effect_mode]
        self.last_detected_mode = self.effect_mode
        self.ml_model = self._load_model_from_mlflow(self.mode_config['MLFLOW_TRACKING_URI'])

        self.model_input_buffer = np.zeros((1, len(self.mode_config['features'])))

        # Register cleanup
        if not self.cleanup_registered:
            atexit.register(self.cleanup)
            self.cleanup_registered = True

    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.data_handler:
                self.data_handler.cleanup()
        except Exception as e:
            logging.error(f"Error during optimizer cleanup: {e}")

    def _load_model_from_mlflow(self, MLFLOW_TRACKING_URI: str):
        """Load ML model from MLflow registry"""
        logging.info(f"Loading model from: {MLFLOW_TRACKING_URI}")

        try:
            model = mlflow.pyfunc.load_model(MLFLOW_TRACKING_URI)
            logging.info(" Model loaded via MLflow PyFunc")
            return model
        except Exception as e:
            logging.warning(f"PyFunc load failed: {e}. Trying pickle...")

            try:
                local_path = mlflow.artifacts.download_artifacts(artifact_uri=MLFLOW_TRACKING_URI)

                possible_paths = [
                    os.path.join(local_path, "model.pkl"),
                    os.path.join(local_path, "final_model.pkl"),
                    local_path
                ]

                for model_path in possible_paths:
                    if os.path.exists(model_path):
                        with open(model_path, "rb") as f:
                            model = pickle.load(f)
                        logging.info(f" Model loaded from pickle: {model_path}")
                        return model

                raise FileNotFoundError(f"No model file found in: {local_path}")

            except Exception as e2:
                logging.critical(f" Model loading failed: {e2}")
                raise

    @log_execution_time
    def extract_data(self) -> Dict[str, Any]:
        """Extract phase: Fetch current plant state and sync target"""
        logging.info("=" * 80)
        logging.info("EXTRACT PHASE: Dynamic Target Sync")
        logging.info("=" * 80)

        # Check for mode change
        current_mode = self.data_handler.detect_effect_mode()
        if self.last_detected_mode != current_mode:
            logging.info(f"Mode changed: {self.last_detected_mode} → {current_mode}")
            self.effect_mode = current_mode
            self.mode_config = EFFECT_MODE_CONFIG[current_mode]
            self.ml_model = self._load_model_from_mlflow(self.mode_config['MLFLOW_TRACKING_URI'])
            self.last_detected_mode = current_mode
            self.model_input_buffer = np.zeros((1, len(self.mode_config['features'])))

        # Fetch plant data
        actual_outputs, current_inputs = self.data_handler.get_current_plant_state_for_mode(
            self.effect_mode
        )

        # Sync evaporation rate target
        raw_actual_evap = actual_outputs.get('Actual Evaporation Rate', 0.0)

        if raw_actual_evap == 0.0:
            logging.warning(" Actual evaporation rate is 0.0 - using safe default")
            raw_actual_evap = 275.0

        lower_bound, upper_bound = SAFE_OPERATING_BOUNDS["Evaporation Rate"]
        self.target_evaporation_rate = float(np.clip(raw_actual_evap, lower_bound, upper_bound))

        logging.info(f"Actual Evap: {raw_actual_evap:.2f} | Target: {self.target_evaporation_rate:.2f}")

        current_inputs["Evaporation Rate"] = self.target_evaporation_rate

        # Plant running check
        steam_flow_param = "De-Superheater LP Steam Flow"
        steam_flow_tag = PARAM_TO_TAG.get(steam_flow_param)

        if steam_flow_tag:
            steam_flow_url = self.data_handler._get_tag_url(steam_flow_tag)

            if steam_flow_url:
                live_steam_flow = self.data_handler.get_live_value_only(
                    steam_flow_param,
                    steam_flow_url
                )

                if live_steam_flow is None or live_steam_flow < 20:
                    logging.warning(
                        f" PLANT OFF CONDITION DETECTED | "
                        f"{steam_flow_param} = {live_steam_flow}"
                    )
                    logging.warning(" Prediction & Optimization DISABLED")
                    return {
                        "plant_off": True,
                        "reason": f"{steam_flow_param} < 20"
                    }

        return {
            "actuals": actual_outputs,
            "inputs": current_inputs,
            "effect_mode": self.effect_mode,
            "target_evaporation_rate": self.target_evaporation_rate
        }

    @log_execution_time
    def transform_and_optimize(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform phase: Optimize parameters using PSO to minimise LP Steam to DeSuperheater (TPH)"""
        logging.info("=" * 80)
        logging.info("TRANSFORM PHASE: PSO – Minimise LP Steam to DeSuperheater (TPH)")
        logging.info("=" * 80)

        current_inputs = extracted_data['inputs']
        mode_features = self.mode_config['features']
        actuals = extracted_data['actuals']
        actual_steam_economy = actuals.get('Actual Steam Economy', 0.0)

        MAX_IMPROVEMENT_PERCENTAGE = 0.04
        max_allowed_steam_economy = actual_steam_economy * (1 + MAX_IMPROVEMENT_PERCENTAGE)

        optimizable_params = []
        lower_bounds, upper_bounds = [], []
        baseline_inputs = current_inputs.copy()

        for param in mode_features:
            if param == "Evaporation Rate":
                continue

            if param in CONTROLLABLE_PARAMETERS:
                optimizable_params.append(param)
                current_val = current_inputs.get(param)

                safe_low, safe_high = SAFE_OPERATING_BOUNDS[param]

                # Handle missing / invalid PI values
                if current_val is None or not np.isfinite(current_val):
                    current_val = (safe_low + safe_high) / 2
                    logging.warning(
                        f"{param}: invalid PI value → using SAFE midpoint {current_val:.3f}"
                    )

                # CRITICAL LINE — clamp current value into safe operating window
                current_val = np.clip(current_val, safe_low, safe_high)

                # Apply max-change constraint safely
                if param in MAX_CHANGE_CONSTRAINTS:
                    max_change = MAX_CHANGE_CONSTRAINTS[param]
                    final_low = max(safe_low, current_val - max_change)
                    final_high = min(safe_high, current_val + max_change)
                else:
                    final_low, final_high = safe_low, safe_high

                lower_bounds.append(final_low)
                upper_bounds.append(final_high)

        lb = np.array(lower_bounds)
        ub = np.array(upper_bounds)

        bounds = (lb, ub)

        # OPTIMIZATION: Pre-compute feature index mapping ONCE (moved outside objective)
        feature_to_index = {feat: idx for idx, feat in enumerate(mode_features)}

        # OPTIMIZATION: Pre-compute baseline array for vectorized operations
        n_features = len(mode_features)
        baseline_array = np.array([baseline_inputs[feat] for feat in mode_features], dtype=np.float64)

        # OPTIMIZATION: Pre-compute parameter indices for fast array updates
        param_to_opt_idx = {param: i for i, param in enumerate(optimizable_params)}
        opt_param_feature_indices = np.array([feature_to_index[param] for param in optimizable_params], dtype=np.int32)

        # OPTIMIZATION: Pre-allocate PSO model input buffer ONCE (reused across all iterations)
        pso_model_buffer = np.zeros((PSO_CONFIG['swarmsize'], n_features), dtype=np.float64)

        # ── Capture values needed inside the closure ────────────────────────
        evaporation_target = float(extracted_data['target_evaporation_rate'])

        # PSO Objective Function — PRIMARY KPI: minimise LP Steam to DeSuperheater (TPH)
        #
        #   LP_Steam (TPH) = Evaporation / Steam_Economy
        #
        # Constraints encoded as penalties:
        #   • Steam Economy improvement must not exceed 4% of actual
        #   • Evaporation is fixed (locked target) — protected by formula itself
        def objective_function(X):
            """
            X: shape (n_particles, n_dimensions)
            Returns per-particle cost = LP_Steam_TPH + penalty.
            PSO minimises this, which directly minimises steam consumption.
            """
            n_particles = X.shape[0]

            # Broadcast baseline and update controllable parameters
            pso_model_buffer[:n_particles, :] = baseline_array
            pso_model_buffer[:n_particles][:, opt_param_feature_indices] = X

            try:
                preds = self.ml_model.predict(pso_model_buffer[:n_particles])
                predicted_se = preds[:, 0].astype(float)

                # ── Primary objective: minimise LP Steam ─────────────────
                # Guard against zero / negative SE predictions
                safe_se = np.where(predicted_se > 1e-6, predicted_se, 1e-6)
                predicted_lp_steam = evaporation_target / safe_se

                # ── Penalty: SE improvement must not exceed 4% cap ───────
                se_cap = actual_steam_economy * (1.0 + MAX_IMPROVEMENT_PERCENTAGE)
                over_cap = np.maximum(0.0, predicted_se - se_cap)
                penalty_cap = 1e4 * over_cap

                costs = predicted_lp_steam + penalty_cap

            except Exception:
                # On any model failure assign worst-case cost
                costs = np.full(n_particles, 1e15, dtype=float)

            return costs

        logging.info(f"Starting PSO optimization with {PSO_CONFIG['swarmsize']} particles...")
        logging.info(f"PSO Parameters: maxiter={PSO_CONFIG['maxiter']}, omega={PSO_CONFIG['omega']}, "
                    f"phip={PSO_CONFIG['phip']}, phig={PSO_CONFIG['phig']}")

        # Run PSO Optimization with proper cleanup
        optimizer = None
        try:
            options = {
                'c1': PSO_CONFIG['phip'],
                'c2': PSO_CONFIG['phig'],
                'w': PSO_CONFIG['omega']
            }

            optimizer = ps.single.GlobalBestPSO(
                n_particles=PSO_CONFIG['swarmsize'],
                dimensions=len(optimizable_params),
                options=options,
                bounds=(lb, ub)
            )

            best_cost, best_position = optimizer.optimize(
                objective_function,
                iters=PSO_CONFIG['maxiter']
            )

            logging.info(
                f"PSO Optimization completed. "
                f"Best LP Steam (TPH): {best_cost:.4f} | "
                f"SE Improvement: {((actual_steam_economy / (evaporation_target / best_cost)) - 1) * 100:+.2f}%"
            )

        except Exception as e:
            logging.error(f"PSO optimization failed: {e}")
            raise
        finally:
            # Ensure optimizer resources are cleaned up
            if optimizer is not None:
                del optimizer

        # Build best solution
        best_solution = current_inputs.copy()
        for i, param in enumerate(optimizable_params):
            best_solution[param] = best_position[i]

        # Generate final predictions
        for feat in mode_features:
            self.model_input_buffer[0, feature_to_index[feat]] = best_solution[feat]

        final_preds = self.ml_model.predict(self.model_input_buffer)
        best_solution['ML Predicted Steam Economy'] = final_preds[0, 0]
        best_solution['ML Predicted SEL Na2CO3'] = final_preds[0, 1]
        best_solution['ML Predicted Evaporation Rate'] = extracted_data['target_evaporation_rate']

        # Store fixed parameters with _PRED suffix
        for param in FIXED_PARAMETERS:
            if param in current_inputs:
                best_solution[f"{param}_PRED"] = current_inputs[param]

        # Clamp Steam Economy if exceeded (secondary guard)
        predicted_se = best_solution['ML Predicted Steam Economy']
        if predicted_se > max_allowed_steam_economy:
            logging.warning(f" Clamping Steam Economy: {predicted_se:.4f} → {max_allowed_steam_economy:.4f}")
            best_solution['ML Predicted Steam Economy'] = max_allowed_steam_economy

        # ── Primary KPI: compute LP Steam (TPH) ──────────────────────────────
        evap_target = float(extracted_data['target_evaporation_rate'])
        safe_pred_se = max(best_solution['ML Predicted Steam Economy'], 1e-6)
        safe_actual_se = max(actual_steam_economy, 1e-6)

        predicted_lp_steam = evap_target / safe_pred_se
        current_lp_steam   = evap_target / safe_actual_se

        best_solution['ML Predicted LP Steam']  = predicted_lp_steam
        best_solution['Current LP Steam']        = current_lp_steam

        logging.info(
            f"LP Steam to DeSuperheater → "
            f"Current: {current_lp_steam:.3f} TPH | "
            f"Predicted: {predicted_lp_steam:.3f} TPH | "
            f"Reduction: {current_lp_steam - predicted_lp_steam:+.3f} TPH"
        )

        return {"recommendation": best_solution, "optimizable_params": optimizable_params}

    @log_execution_time
    def load_results(self, best_solution: Dict, extracted_data: Dict, optimizable_params: List[str]):
        """Load phase: Write results to PI and log summary"""
        logging.info("=" * 80)
        logging.info(f"LOAD PHASE - {self.mode_config['name']}")
        logging.info("=" * 80)

        self._log_results(best_solution, extracted_data, optimizable_params)
        self._write_results_to_pi(best_solution, optimizable_params)

    def _log_results(self, best: Dict, extracted_data: Dict, optimizable_params: List[str]):
        """Log optimization results"""
        actuals = extracted_data['actuals']
        currents = extracted_data['inputs']
        target_evap = extracted_data['target_evaporation_rate']

        logging.info("\n" + "=" * 80)
        logging.info("PSO OPTIMIZATION SUMMARY")
        logging.info("=" * 80)

        actual_se   = actuals.get('Actual Steam Economy', 0.0)
        pred_se     = best['ML Predicted Steam Economy']
        se_delta    = pred_se - actual_se
        se_improvement_pct = (se_delta / actual_se * 100) if actual_se != 0 else 0.0

        actual_evap = actuals.get('Actual Evaporation Rate', 0.0)
        actual_sel  = actuals.get('Actual SEL Na2CO3', 0.0)

        # ── Primary KPI: LP Steam reduction ──────────────────────────────────
        current_lp_steam   = best.get('Current LP Steam',  0.0)
        predicted_lp_steam = best.get('ML Predicted LP Steam', 0.0)
        lp_steam_reduction = current_lp_steam - predicted_lp_steam
        lp_steam_pct       = (lp_steam_reduction / current_lp_steam * 100) if current_lp_steam != 0 else 0.0

        logging.info("\nKPI PREDICTIONS:")
        logging.info("-" * 60)
        logging.info("★ PRIMARY KPI — LP Steam to DeSuperheater:")
        logging.info(f"  Current (baseline):  {current_lp_steam:8.3f} TPH")
        logging.info(f"  Predicted optimized: {predicted_lp_steam:8.3f} TPH")
        logging.info(f"  Reduction:           {lp_steam_reduction:+8.3f} TPH ({lp_steam_pct:+.2f}%)")
        logging.info(f"\nSteam Economy (context):")
        logging.info(f"  Actual:      {actual_se:8.4f}")
        logging.info(f"  Predicted:   {pred_se:8.4f}")
        logging.info(f"  Improvement: {se_delta:+8.4f} ({se_improvement_pct:+.2f}%)")
        logging.info(f"\nEvaporation Rate:")
        logging.info(f"  Actual:      {actual_evap:8.2f} TPH")
        logging.info(f"  Target:      {target_evap:8.2f} TPH")
        logging.info(f"\nSEL Na2CO3:")
        logging.info(f"  Actual:      {actual_sel:8.4f} g/L")
        logging.info(f"  Predicted:   {best['ML Predicted SEL Na2CO3']:8.4f} g/L")

        logging.info("\n" + "=" * 85)
        logging.info("PARAMETER RECOMMENDATIONS (PSO)")
        logging.info("=" * 85)
        logging.info(f"{'Parameter':<50} | {'Current':>10} | {'Optimized':>10} | {'Change':>10}")
        logging.info("-" * 85)

        for param in optimizable_params:
            if param in currents:
                current_val = currents.get(param, 0.0)
                optimized_val = best[param]
                change = optimized_val - current_val
                change_pct = (change / current_val * 100) if current_val != 0 else 0.0

                marker = "→" if abs(change) > 0.01 else " "
                logging.info(
                    f"{marker}{param:<49} | {current_val:10.3f} | {optimized_val:10.3f} | "
                    f"{change:+6.3f} ({change_pct:+.1f}%)"
                )

        logging.info("=" * 85)

    @log_execution_time
    def _write_results_to_pi(self, best_solution: Dict, optimizable_params: List[str]):
        """Write all predictions to PI tags - OPTIMIZED with parallel execution"""
        logging.info("\nWriting predictions to PI System...")

        write_tasks = []

        # Parameters → write to _PRED
        all_params_to_write = set(optimizable_params) | set(FIXED_PARAMETERS)

        for param in all_params_to_write:
            if param in EXCLUDE_FROM_PRED_WRITE:
                logging.info(f"Skipping _PRED write for: {param}")
                continue

            input_tag = PARAM_TO_TAG.get(param)
            if not input_tag:
                continue

            pred_tag_name = f"{input_tag}_PRED"
            write_url = self.data_handler._get_tag_url(pred_tag_name)

            if not write_url:
                logging.warning(f"PRED tag missing in PI: {pred_tag_name}")
                continue

            if param in optimizable_params:
                value = best_solution.get(param)
            else:
                value = best_solution.get(f"{param}_PRED")

            if value is not None:
                write_tasks.append((write_url, value, f"{param}_PRED"))

        # KPI PREDICTIONS → dedicated KPI tags
        for kpi_name, pred_tag_name in KPI_PREDICTION_TAGS.items():
            value = best_solution.get(kpi_name)
            if value is None:
                logging.warning(f"KPI value missing: {kpi_name}")
                continue

            write_url = self.data_handler._get_tag_url(pred_tag_name)
            if not write_url:
                logging.warning(f"KPI PRED tag missing in PI: {pred_tag_name}")
                continue

            write_tasks.append((write_url, value, kpi_name))

        # =========================================================================
        # OPTIMIZATION: Parallel write execution using ThreadPoolExecutor
        # =========================================================================
        write_success = 0
        write_failure = 0

        # Use ThreadPoolExecutor for parallel I/O (HTTP writes are I/O-bound)
        max_workers = min(10, len(write_tasks))  # Limit concurrent connections

        def write_single_task(task):
            """Helper function for parallel execution"""
            write_url, value, param_name = task
            try:
                success = self.data_handler.write_prediction_to_pi(write_url, value, param_name)
                return (param_name, success, None)
            except Exception as e:
                return (param_name, False, str(e))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all write tasks concurrently
            future_to_task = {executor.submit(write_single_task, task): task for task in write_tasks}

            # Collect results as they complete
            for future in as_completed(future_to_task):
                param_name, success, error = future.result()
                if success:
                    write_success += 1
                else:
                    write_failure += 1
                    if error:
                        logging.error(f"Write error for {param_name}: {error}")

        logging.info(f"\n{'=' * 80}")
        logging.info(f"WRITE SUMMARY:  {write_success} Successful |  {write_failure} Failed")
        logging.info(f"{'=' * 80}")

        if write_success == 0 and write_failure > 0:
            raise Exception(" CRITICAL: All PI write operations failed!")


# ============================================================================
# TASK FUNCTIONS (for Airflow Integration)
# ============================================================================

def extract_task_func(config):
    """Extract task with proper resource management"""
    logging.info("--- Starting Extract Task ---")
    pipeline = None
    try:
        pipeline = EvaporationPlantOptimizer(config)
        plant_data = pipeline.extract_data()

        effect_mode = plant_data.get('effect_mode', 'UNKNOWN')
        logging.info(f"Extract Task Complete - Mode: {effect_mode}")

        return plant_data
    finally:
        if pipeline:
            pipeline.cleanup()


def transform_task_func(config, **context):
    """Transform task with proper resource management"""
    logging.info("--- Starting Transform Task (PSO) ---")
    ti = context['ti']
    plant_data = ti.xcom_pull(task_ids='extract_task')

    if not plant_data:
        raise ValueError("Could not retrieve plant data from extract_task. Halting.")

    effect_mode = plant_data.get('effect_mode', '6_EFFECT')
    logging.info(f"Transform Task running for: {EFFECT_MODE_CONFIG[effect_mode]['name']}")

    # Run heavy PSO optimization in a separate subprocess to avoid
    # interfering with Airflow worker processes and prevent zombie jobs.
    optimization_results = run_pso_in_subprocess(config, plant_data)

    return {
        "optimization_results": optimization_results,
        "plant_data": plant_data
    }


def load_task_func(config, **context):
    """Load task with proper resource management"""
    logging.info("--- Starting Load Task ---")
    ti = context['ti']
    data_from_transform = ti.xcom_pull(task_ids='transform_task')

    if not data_from_transform:
        raise ValueError("No data received from transform_task")

    optimization_results = data_from_transform.get('optimization_results')
    if not optimization_results:
        logging.warning("No optimization results found in transform output. Skipping load task.")
        return

    plant_data = data_from_transform['plant_data']
    effect_mode = plant_data.get('effect_mode', '6_EFFECT')

    logging.info(f"Load Task executing for: {EFFECT_MODE_CONFIG[effect_mode]['name']}")

    recommendation_dict = optimization_results.get('recommendation')
    if not recommendation_dict:
        logging.warning("No recommendation found in optimization results. Skipping load task.")
        return

    best_solution = pd.Series(recommendation_dict)
    optimizable_params = optimization_results['optimizable_params']

    logging.info(f"Best solution contains {len(best_solution)} parameters")
    logging.info(f"KPI predictions in solution: {[k for k in KPI_PREDICTION_TAGS.keys() if k in best_solution]}")

    pipeline = None
    try:
        pipeline = EvaporationPlantOptimizer(config, force_mode=effect_mode)
        pipeline.load_results(best_solution, plant_data, optimizable_params)

        logging.info("Load task completed successfully")
    finally:
        if pipeline:
            pipeline.cleanup()


if __name__ == "__main__":
    """
    CLI entry point used by the Airflow task to run PSO once in a clean subprocess.

    Usage:
        python PSO.py --transform-once <input_json_path> <output_json_path>
    """
    setup_logging()

    if len(sys.argv) >= 4 and sys.argv[1] == "--transform-once":
        input_path = sys.argv[2]
        output_path = sys.argv[3]

        with open(input_path, "r") as f:
            payload = json.load(f)

        config = payload["config"]
        plant_data = payload["plant_data"]
        effect_mode = plant_data.get("effect_mode", "6_EFFECT")

        pipeline = None
        try:
            pipeline = EvaporationPlantOptimizer(config, force_mode=effect_mode)
            optimization_results = pipeline.transform_and_optimize(plant_data)

            # Make recommendation JSON-serializable (same logic as before)
            if optimization_results and optimization_results.get("recommendation") is not None:
                recommendation_series = optimization_results["recommendation"]

                recommendation_dict = {}
                for key, value in recommendation_series.items():
                    if isinstance(value, (np.floating, np.integer)):
                        recommendation_dict[key] = float(value)
                    elif isinstance(value, np.ndarray):
                        recommendation_dict[key] = value.tolist()
                    else:
                        recommendation_dict[key] = value

                optimization_results["recommendation"] = recommendation_dict

            with open(output_path, "w") as f:
                json.dump(optimization_results, f, default=_json_default)

        finally:
            if pipeline:
                pipeline.cleanup()
    else:
        logging.info("PSO.py executed as a script with no or unknown arguments; nothing to run.")
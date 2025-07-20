from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from vllm import LLM, SamplingParams
import torch

# =================================================================================================
# --- 1. API and LLM Setup ---
# =================================================================================================

app = FastAPI(
    title="LLM Behavioral Risk API",
    description="Analyzes session context to provide a behavioral risk score for user authentication.",
    version="1.0.0"
)

# --- LLM Configuration ---
# We use a quantized version of Llama-3-8B for high efficiency.
# 'auto' will use the GPU if available, otherwise CPU.
# On first run, this will download the model (approx. 4GB).
try:
    print("[INFO] Loading Llama-3-8B-Instruct model with vLLM...")
    llm = LLM(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        quantization="awq", # AWQ is a state-of-the-art quantization method
        dtype="auto",
        trust_remote_code=True,
        max_model_len=2048 # Max context length
    )
    print("[INFO] LLM loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load LLM. Ensure you have a GPU with sufficient VRAM. Error: {e}")


# --- Sampling Parameters for the LLM ---
# We want a deterministic, structured output.
sampling_params = SamplingParams(
    temperature=0.0, # No randomness
    max_tokens=150,  # Max length of the generated response
    stop=["}"]       # Stop generating once it forms a complete JSON object
)

# =================================================================================================
# --- 2. Pydantic Models for Request/Response ---
# =================================================================================================

class SessionContext(BaseModel):
    # Historical data about the user's typical behavior
    common_login_times: List[str] = Field(..., example=["09:00-11:00", "19:00-22:00"])
    common_wifi_ssids: List[str] = Field(..., example=["HomeWiFi", "Office-Guest"])
    
    # Data from the current authentication attempt
    current_time: str = Field(..., example="02:45")
    current_wifi_ssid: str = Field(..., example="xfinitywifi")
    
    # The score from your primary biometric model
    biometric_similarity_score: float = Field(..., example=0.85, ge=0.0, le=1.0)


class RiskResponse(BaseModel):
    risk_score: float = Field(..., description="A score from 0.0 (low risk) to 1.0 (high risk).")
    reason: str = Field(..., description="A natural language explanation for the assigned risk score.")
    
# =================================================================================================
# --- 3. Prompt Engineering ---
# =================================================================================================

def create_llm_prompt(context: SessionContext) -> str:
    """
    Creates a detailed, structured prompt for the LLM to ensure it returns
    a reliable and parsable JSON output.
    """
    prompt = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a security risk analysis expert. Your task is to analyze a user's session data against their historical patterns and provide a risk score. The output must be a single, valid JSON object containing 'risk_score' (a float between 0.0 and 1.0) and 'reason' (a brief explanation). Do not add any text before or after the JSON object.

Analyze the following factors:
1.  **Time of Day:** Is the current login time within the user's common login windows?
2.  **Location:** Is the current WiFi network one the user commonly uses?
3.  **Biometric Score:** Is the biometric similarity score typical for this user, or is it suspiciously low? A score below 0.8 is low, below 0.9 is slightly unusual.

Combine these factors to determine the overall risk. A login that is unusual in multiple ways (e.g., late at night, from a strange location, with a low biometric score) is very high risk.<|eot_id|><|start_header_id|>user<|end_header_id|>

Analyze the following session and provide your risk assessment as a JSON object.

**User's Historical Patterns:**
- Common Login Times: {context.common_login_times}
- Common WiFi SSIDs: {context.common_wifi_ssids}

**Current Session Data:**
- Current Time: "{context.current_time}"
- Current WiFi SSID: "{context.current_wifi_ssid}"
- Biometric Similarity Score: {context.biometric_similarity_score:.2f}

JSON Response:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{{
"""
    return prompt

# =================================================================================================
# --- 4. API Endpoint ---
# =================================================================================================

@app.post("/calculate_risk", response_model=RiskResponse)
async def calculate_risk(context: SessionContext):
    """
    Calculates a behavioral risk score based on the provided session context.
    """
    prompt = create_llm_prompt(context)
    
    try:
        # Generate the response from the LLM
        outputs = llm.generate(prompt, sampling_params)
        
        # The output includes the prompt, so we extract only the generated part.
        # We also add back the closing brace that we used as a stop token.
        json_output_str = outputs[0].outputs[0].text + "}"
        
        # Parse the JSON string into our Pydantic model for validation and response
        response_data = RiskResponse.parse_raw(json_output_str)
        
        return response_data
        
    except Exception as e:
        print(f"Error during LLM inference or parsing: {e}")
        raise HTTPException(status_code=500, detail="Failed to process the request with the LLM.")

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "LLM Risk API is running. See /docs for details."}
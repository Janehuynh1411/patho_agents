from pathrag.generate import generate_answer
from confidence_utils import compute_confidence_score_from_logits
import torch

def llava_med_agent(input_text, image_path=None):
    # Step 1: generate answer using Path-RAG
    response, logits, input_ids = generate_answer(input_text, image_path, return_logits=True)
    
    # Step 2: compute confidence score
    confidence = compute_confidence_score_from_logits(logits, input_ids)
    
    return response, confidence

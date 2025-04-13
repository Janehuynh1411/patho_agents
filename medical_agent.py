import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from confidence_utils import compute_confidence_score_from_logits

bio_tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
bio_model = AutoModelForCausalLM.from_pretrained("microsoft/biogpt")
bio_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bio_model.to(device)

def medical_ai_agent(input_text):
    inputs = bio_tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = bio_model.generate(**inputs, max_length=50, output_scores=True, return_dict_in_generate=True)
        response_tokens = outputs.sequences[0]
        response_text = bio_tokenizer.decode(response_tokens, skip_special_tokens=True)

        model_outputs = bio_model(**inputs, labels=inputs["input_ids"])
        confidence = compute_confidence_score_from_logits(model_outputs.logits, inputs["input_ids"])

    return response_text, confidence

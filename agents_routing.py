def route_based_on_confidence_llava_to_doctor(input_text):
    print(f"\n[INPUT] {input_text}")

    # Step 1: LLaVaMed
    llava_response, llava_conf = llava_med_agent(input_text)
    print(f"[LLaVaMed] Confidence: {llava_conf:.2f}")
    if llava_conf >= 0.9:
        return llava_response, "LLaVaMed"

    # Step 2: General AI
    gen_response, gen_conf = general_ai_agent(input_text)
    print(f"[General AI Agent] Confidence: {gen_conf:.2f}")
    if gen_conf >= 0.9:
        return gen_response, "General AI Agent"

    # Step 3: Medical AI (BioGPT)
    med_response, med_conf = medical_ai_agent(input_text)
    print(f"[Medical AI Agent - BioGPT] Confidence: {med_conf:.2f}")
    if med_conf >= 0.9:
        return med_response, "Medical AI Agent"

    # Step 4: Human fallback
    final_response = doctors_agent(input_text)
    return final_response, "Doctors Agent"

# ====== Example Usage ======
if __name__ == "__main__":
    user_input = "Describe the findings in this histopathology image."
    response, source = route_based_on_confidence_llava_to_doctor(user_input)

    print(f"\n[FINAL RESPONSE - From: {source}]\n{response}")

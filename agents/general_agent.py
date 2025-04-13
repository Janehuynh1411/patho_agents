import openai
import os

def general_ai_agent(input_text):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    messages = [
        {"role": "system", "content": "You are a helpful general-purpose AI assistant."},
        {"role": "user", "content": input_text}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7,
        max_tokens=300,
        logprobs=True
    )

    output_text = response["choices"][0]["message"]["content"]
    confidence = 0.85  # Placeholder: OpenAI API doesn't expose full token confidence
    return output_text, confidence

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

exes = {
    1: {
        "name": "Stella",
        "model": 'dandelion4/stella-mistral-7b'
    }, 
    2: {
        "name": "Lily",
        "model": "dandelion4/lily-mistral-7b"
    },
    3: {
        "name": "Anastasia",
        "model": "dandelion4/annie-mixtral"
    }
}
# args 
ex = int(sys.argv[1])
# Initialize the tokenizer and model
adapter_model_id = exes[ex]["model"]
model = AutoModelForCausalLM.from_pretrained(adapter_model_id)
tokenizer = AutoTokenizer.from_pretrained(adapter_model_id)

# Ensure the model is in evaluation mode
model.eval()

# Move model to GPU if available
if torch.cuda.is_available():
    model = model.to("cuda")

# Initialize conversation history
conversation_history = ""

# Chat loop
while True:
    # Get user input
    user_input = input("Andrew: ")
    if user_input.lower() == "quit":
        break

    # Update conversation history with the user input
    name = exes[ex]["name"]
    conversation_history += "Andrew: " + user_input + f"\n{name}: "

    # Encode the conversation history for the model
    inputs = tokenizer.encode(conversation_history, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")

    # Generate a response
    response_ids = model.generate(inputs,
                                repetition_penalty=1.1,
                                max_new_tokens=1024,
                                temperature=0.9,
                                top_p=0.95,
                                top_k=40,
                                bos_token_id=tokenizer.bos_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.pad_token_id,
                                do_sample=True,
                                use_cache=True, 
                                num_return_sequences=1)
    raw_response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    response = raw_response.split(conversation_history)[-1]
    response = response.split(f"\n{name}:")[0]
    # Print the model's response
    print(f"{name}: {response}")

    # Update conversation history with the model's response
    conversation_history += response + "\n"

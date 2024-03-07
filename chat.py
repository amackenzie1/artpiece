import sys

import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

girls = {
    1: {
        "name": "Stella",
        "model": "dandelion4/stella-mistral-7b"
    },
    2: {
        "name": "Lily",
        "model": "dandelion4/lily-mistral-7b"
    },
    3: {
        "name": "Anastasia",
        "model": "dandelion4/annie-mixtral"
        # "model": "mistralai/Mixtral-8x7B-v0.1"
    }
}

# Initialize accelerator
accelerator = Accelerator()

# args
girl = int(sys.argv[1])

# Initialize the tokenizer and model
adapter_model_id = girls[girl]["model"]
if (adapter_model_id == "dandelion4/annie-mixtral"):
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1", 
                                                low_cpu_mem_usage=True, load_in_8bit=True, 
                                             torch_dtype=torch.bfloat16,
                                             device_map="auto", trust_remote_code=True)
    model.load_adapter("dandelion4/annie-mixtral")
else:
    model = AutoModelForCausalLM.from_pretrained(adapter_model_id)
                                               

tokenizer = AutoTokenizer.from_pretrained(adapter_model_id)

# Ensure the model is in evaluation mode
model.eval()

# Prepare the model and move it to the accelerator device
model = accelerator.prepare(model)

# Initialize conversation history
conversation_history = """
    Discord chat logs: 2024-03-07
"""

# Chat loop
while True:
    # Get user input
    user_input = input("Andrew: ")
    if user_input.lower() == "quit":
        break
    
    # Update conversation history with the user input
    name = girls[girl]["name"]
    conversation_history += "Andrew: " + user_input + f"\n{name}: "
    
    # Encode the conversation history for the model
    inputs = tokenizer.encode(conversation_history, return_tensors="pt")
    inputs = inputs.to(accelerator.device)
    
    # Generate a response
    response_ids = accelerator.unwrap_model(model).generate(
        inputs,
        repetition_penalty=1.1,
        max_new_tokens=128,
        temperature=0.8,
        top_p=0.8,
        top_k=0,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        use_cache=True,
        num_return_sequences=1
    )
    
    raw_response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    response = raw_response.split(conversation_history)[-1]
    response = response.split(f"\nAndrew:")[0]
    
    # Print the model's response
    print(f"{name}: {response}")
    
    # Update conversation history with the model's response
    conversation_history += response + "\n"
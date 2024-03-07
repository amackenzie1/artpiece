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
        "model": "dandelion4/annie-mistral-7b"
        # "model": "dandelion4/annie-mixtral"
    }
}

# Initialize accelerator
accelerator = Accelerator()

# args
girl = int(sys.argv[1])

# Initialize the tokenizer and model
adapter_model_id = girls[girl]["model"]
if ("mixtral" in adapter_model_id):
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1", 
                                                low_cpu_mem_usage=True, load_in_8bit=True, 
                                             torch_dtype=torch.bfloat16,
                                             device_map="auto", trust_remote_code=True)
    model.load_adapter(adapter_model_id)
else:
    model = AutoModelForCausalLM.from_pretrained(adapter_model_id)
                                               

tokenizer = AutoTokenizer.from_pretrained(adapter_model_id)

# Ensure the model is in evaluation mode
model.eval()

# Prepare the model and move it to the accelerator device
model = accelerator.prepare(model)

# Initialize conversation history
conversation_history = open(f"Data/{girls[girl]['name']}_start.txt").read().strip() + "\n"
# Encode the initial conversation history
conversation_history = tokenizer.encode(conversation_history, return_tensors="pt")
conversation_history = conversation_history.to(accelerator.device)

while True:
    # Get user input
    user_input = input("Andrew: ")
    if user_input.lower() == "quit":
        break

    # Update conversation history with the user input
    name = girls[girl]["name"]
    user_input_ids = tokenizer.encode(f"Andrew: {user_input}\n{name}:", return_tensors="pt")
    user_input_ids = user_input_ids.to(accelerator.device)
    conversation_history = torch.cat([conversation_history, user_input_ids], dim=-1)

    # Generate a response
    response_ids = accelerator.unwrap_model(model).generate(
        conversation_history,
        repetition_penalty=1.1,
        max_new_tokens=128,
        temperature=0.8,
        top_p=0.8,
        top_k=0,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        use_cache=True,
        num_return_sequences=1
    )

    # Decode the generated response
    response = tokenizer.decode(response_ids[:, conversation_history.shape[-1]:][0], skip_special_tokens=True)

    if "Andrew:" in response:
        response = response.split("Andrew:")[0].strip()

    # Print the model's response
    print(f"{name}: {response}")

    # Update conversation history with the model's response
    response_ids = tokenizer.encode(f"{response}\n", return_tensors="pt")
    response_ids = response_ids.to(accelerator.device)
    conversation_history = torch.cat([conversation_history, response_ids], dim=-1)
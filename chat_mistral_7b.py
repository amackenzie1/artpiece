import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize the tokenizer and model
adapter_model_id = "dandelion4/annie-llama-mistral-7b"
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
    conversation_history += "Andrew:####" + user_input + "####Anastasia:####"

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
    response = response.split("####Andrew:")[0]
    # Print the model's response
    print(f"Annie: {response}")

    # Update conversation history with the model's response
    conversation_history += response + "####"

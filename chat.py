import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Initialize the tokenizer and model
adapter_model_id = "dandelion4/annie-llama-mistral-7b"
model = AutoModelForCausalLM.from_pretrained(adapter_model_id)
tokenizer = AutoModelForCausalLM.from_pretrained(adapter_model_id)

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
    response_ids = model.generate(inputs, max_length=100, pad_token_id=tokenizer.eos_token_id,
                                  top_p=0.95, top_k=50, temperature=0.7,
                                  num_return_sequences=1)
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)

    # Print the model's response
    print(f"Annie: {response}")

    # Update conversation history with the model's response
    conversation_history += response + "####"

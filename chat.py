from threading import Thread

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          GenerationConfig, TextIteratorStreamer, TextStreamer,
                          pipeline)

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


def generate(instruction):
        if not instruction:
            return
        prompt = instruction.strip()
        batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

        model.eval()
        with torch.no_grad():
            generation_config = GenerationConfig(
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
                return_dict_in_generate=True,
                output_attentions=False,
                output_hidden_states=False,
                output_scores=False,
            )
            streamer = TextIteratorStreamer(tokenizer)
            generation_kwargs = {
                "inputs": batch["input_ids"].to(cfg.device),
                "generation_config": generation_config,
                "streamer": streamer,
            }

            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            all_text = ""

            for new_text in streamer:
                all_text += new_text
                yield all_text


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

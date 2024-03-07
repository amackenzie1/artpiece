from transformers import pipeline

# Load the text generation pipeline with the specified model
generator = pipeline('text-generation', model='dandelion4/annie-llama-mistral-7b')

# Generate text based on a prompt
prompt = "Andrew:####Hi love!####Anastasia:"
generated_text = generator(prompt, max_length=100, num_return_sequences=1)

# Print the generated text
print(generated_text[0]['generated_text'])
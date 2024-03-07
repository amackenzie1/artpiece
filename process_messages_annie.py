import json
import re

messages = open("Data/discord.txt").read() 

# split at either Anastasia: or Andrew: (but keep those two as part of the message)
messages = re.split(r"(Anastasia:|Andrew:.*)", messages)[1:]
messages = [message.strip() for message in messages if message.strip() != ""]
def get_chunks():
    chunks = [] 
    chunk = "" 
    for message in messages: 
        if len(chunk) > 2**11:
            chunks.append(chunk)
            chunk = message + "\n"
        else:
            chunk += message + "\n"
    chunks.append(chunk)
    chunks = [re.sub(r"Anastasia:\n", "Anastasia: ", chunk) for chunk in chunks]
    return chunks 

chunks = get_chunks()
print(chunks[0])
# jsonl format {"text": ...}
with open(f"Data/discord_annie.jsonl", "w") as f:
    for chunk in chunks:
        mydict = {
            "text": chunk
        }
        f.write(json.dumps(mydict) + "\n")
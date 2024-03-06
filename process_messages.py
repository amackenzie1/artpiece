import json
import re

messages = open("discord.txt").read() 

# split at either Anastasia: or Andrew: (but keep those two as part of the message)
messages = re.split(r"(Anastasia:|Andrew:.*)", messages)[1:]
messages = [message.strip() for message in messages if message.strip() != ""]
def get_chunks():
    chunks = [] 
    chunk = "" 
    for message in messages: 
        if len(chunk) > 2**11:
            chunks.append(chunk)
            chunk = message + "####"
        else:
            chunk += message + "####"
    chunks.append(chunk)
    return chunks 

chunks = get_chunks()
# jsonl format {"text": ...}
with open(f"discord.jsonl", "w") as f:
    for chunk in chunks:
        mydict = {
            "text": chunk
        }
        f.write(json.dumps(mydict) + "\n")
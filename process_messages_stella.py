import json
import re

messages = open("Data/hangouts-conversation-21.txt").read() 

# split at either Anastasia: or Andrew: (but keep those two as part of the message)
messages = re.split(r"(<Andrew Mackenzie>|<Unknown>)", messages)
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
print(chunks[0])
exit()
# jsonl format {"text": ...}
with open(f"discord.jsonl", "w") as f:
    for chunk in chunks:
        mydict = {
            "text": chunk
        }
        f.write(json.dumps(mydict) + "\n")
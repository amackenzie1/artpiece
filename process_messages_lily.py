import json
import re

messages = open("Data/discord_lily.txt").read() 

messages = re.split(r"(Andrew:|Lily:)", messages)
# if it matches this date format: 2020-04-02T18:47:46.159 - just get rid of that part 
messages = [re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d* - ", "", message) for message in messages]
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
    chunks = [re.sub(r"Andrew:\n", "Andrew: ", chunk) for chunk in chunks]
    chunks = [re.sub(r"Lily:\n", "Lily: ", chunk) for chunk in chunks]
    return chunks 


chunks = get_chunks()
print(chunks[0])
# jsonl format {"text": ...}
with open(f"Data/discord_lily.jsonl", "w") as f:
    for chunk in chunks:
        mydict = {
            "text": chunk
        }
        f.write(json.dumps(mydict) + "\n")
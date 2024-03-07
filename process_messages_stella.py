import json
import re

messages = open("Data/hangouts-conversation-21.txt").read() 

messages = re.split(r"(<Andrew Mackenzie>|<Unknown>)", messages)
# if it matches this date format: 2018-01-15 13:17:07 just get rid of that part 
messages = [re.sub(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", "", message) for message in messages]
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
    # replace <Andrew Mackenze>\n with Andrew: 
    # replace <Unknown>\n with Stella:
    chunks = [re.sub(r"<Andrew Mackenzie>\n", "Andrew: ", chunk) for chunk in chunks]
    chunks = [re.sub(r"<Unknown>\n", "Stella: ", chunk) for chunk in chunks]
    return chunks 


chunks = get_chunks()
print(chunks[0])
# jsonl format {"text": ...}
with open(f"Data/discord_stella.jsonl", "w") as f:
    for chunk in chunks:
        mydict = {
            "text": chunk
        }
        f.write(json.dumps(mydict) + "\n")
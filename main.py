from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

model_name = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

class Message(BaseModel):
    model: str
    messages: list
    stream: bool = False

@app.post("/v1/chat/completions")
async def chat_endpoint(request: Message):
    prompt = "\n".join([msg["content"] for msg in request.messages])
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"choices": [{"message": {"role": "assistant", "content": response}}]}

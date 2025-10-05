import os
import torch
import huggingface_hub
import transformers

class CharacterChatBot:
    def __init__(self, model_path, huggingface_token=None):
        self.model_path = model_path
        self.huggingface_token = huggingface_token

        if huggingface_token:
            huggingface_hub.login(self.huggingface_token)

        print(f"Loading model: {self.model_path} ...")
        self.model = self.load_model(self.model_path)

    def load_model(self, model_path):
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            device_map="auto",        # auto assign GPU
            torch_dtype=torch.float16 # use fp16 for large models
        )
        return pipeline

    def chat(self, message, history):
        # Combine history and current message
        conversation = []
        conversation.append("You are Naruto from the anime 'Naruto'. Respond like him.")

            
        # Add only the conversation content, no "User:" / "Naruto:" labels
        for msg, resp in history:
            prompt += f"{msg}\n{resp}\n"

        # Add the new user message
        prompt += f"{message}\n"



        output = self.model(
            prompt,
            max_length=512,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )
            
        # The model output is full text — strip the prompt to keep only Naruto’s reply
        generated_text = output[0]["generated_text"]
        reply = generated_text[len(prompt):].strip()

        return {"content": reply} 

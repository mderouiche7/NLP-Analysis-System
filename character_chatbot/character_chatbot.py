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
        # Build the conversation prompt
        prompt = "You are Naruto from the anime 'Naruto'. Respond like him.\n"
        
        for user_msg, assistant_msg in history:
            prompt += f"User: {user_msg}\n"
            prompt += f"Naruto: {assistant_msg}\n"
        
        prompt += f"User: {message}\nNaruto:"

        # Tokenize prompt
        inputs = self.model.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate
        output_ids = self.model.generate(
            **inputs,
            max_length=inputs.input_ids.shape[1] + 256,
            eos_token_id=self.model.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )

        # Decode generated text
        generated_text = self.model.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Remove prompt and return only generated part
        response = generated_text[len(prompt):].strip()
        return {"content": response}

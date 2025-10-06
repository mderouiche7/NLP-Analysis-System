<p align="center">

  <img src="NLP-Analysis-System/assets/naruto_banner.jpg" alt="Naruto NLP Chatbot Banner" width="800"/>
</p>

<h1 align="center"> NLP-Analysis-System: Classification, Character Networks, and Chatbots</h1>

<p align="center">
  <b>A full end-to-end NLP fine-tuning & deployment framework that brings anime characters to life.</b>  
  <br>
  Fine-tune open-weight <code>LLaMA</code> models on dialogue datasets and deploy interactive chatbots seamlessly with <code>Gradio</code>.
</p>

---

##  Overview

The **AI–NLP Analysis System** is a modular project designed for **academic research**, **AI experimentation**, or **portfolio demonstration** in applied Natural Language Processing.

### Core Capabilities
1.  Preprocess and analyze **dialogue-based datasets** (e.g., anime subtitles).
2.  Fine-tune open-source **Meta LLaMA models** for personality emulation.
3.  Serve **interactive chatbots** via a Gradio web interface.
4.  Optionally deploy models to **Hugging Face Hub** for public access.

> **Example Use Case:**  
> Fine-tune `meta-llama/Meta-Llama-3-8B` to reproduce *Naruto Uzumaki*’s dialogue style and interact with users dynamically.

---

##  Features

###  Dataset Pipeline
- Cleans and structures raw subtitle or script data  
- Extracts **question–response pairs** automatically  
- Converts processed data into **Hugging Face Dataset** format  

###  Fine-Tuning Pipeline
- Based on **Transformers** + **PEFT (LoRA)** for parameter-efficient adaptation  
- Supports **FP16/BF16 mixed precision** training  
- Includes Hugging Face **upload and model card automation**

###  Character Chatbot Engine
- Loads fine-tuned **LLaMA-3** checkpoints  
- Maintains **multi-turn conversation history**  
- Dynamically applies **character personality system prompts**

###  Gradio Web App
- Clean, responsive, and **GPU-ready interface**  
- Plug-and-play **Hugging Face Space** deployment  
- Lightweight and compatible with CPU demos for academic use

---

##  Demo

<p align="center">
  <img src="assets/gradio_demo.png" alt="Gradio Chatbot Demo" width="700"/>
</p>

##  Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Transformers](https://img.shields.io/badge/🤗Transformers-PEFT-yellow)
![Gradio](https://img.shields.io/badge/Gradio-App-green)
![LLaMA](https://img.shields.io/badge/LLaMA-3-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-lightgrey?logo=huggingface)

</div>

---

##  Installation & Setup

### 1️ Clone the Repository
```bash
git clone https://github.com/<your-username>/NLP-Analysis-System.git
cd NLP-Analysis-System


---

## ⚙️ Required Setup for NER Character Network

The NER module uses SpaCy’s transformer-based English model `en_core_web_trf`, which must be installed manually if you are working in a cloud environment:

```bash
pip install spacy
python -m spacy download en_core_web_trf
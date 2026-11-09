# 🧠 DumLLM: Dual-Model Studio

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/shubhamisok/dumllm-studio)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-orange)](https://huggingface.co/shubhamisok/dumllm-models)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**DumLLM** is an end-to-end Large Language Model project demonstrating from-scratch training, fine-tuning, and cloud deployment of custom GPT-2 architectures built with PyTorch.

The repository includes a modern, dual-tab **Gradio** web application that allows real-time interactive testing of two custom fine-tuned checkpoints:
1. **🛡️ SMS Spam Classifier**: Fine-tuned on GPT-2 Small (124M) with a binary classification head, providing real-time probability confidence scores.
2. **💬 Instruction Assistant**: Fine-tuned on GPT-2 Medium (355M) using Supervised Fine-Tuning (SFT) formatted with Alpaca-style instruction prompts.

---

## 🔗 Quick Links & Live Demos

* **🚀 Live Demo on Hugging Face Spaces**: [shubhamisok/dumllm-studio](https://huggingface.co/spaces/shubhamisok/dumllm-studio)
* **🤗 Hosted Model Checkpoints**: [shubhamisok/dumllm-models](https://huggingface.co/shubhamisok/dumllm-models)
* **💻 GitHub Repository**: [ShubhamKumarSahu-svg/DumLLM](https://github.com/ShubhamKumarSahu-svg/DumLLM)

---

## 📦 Model Checkpoints & Specifications

The fine-tuned weights are hosted on the Hugging Face Hub under [`shubhamisok/dumllm-models`](https://huggingface.co/shubhamisok/dumllm-models) and automatically downloaded on first launch:

| Model Name | Base Architecture | Parameters | Embedding Dim | Layers / Heads | Checkpoint File | Task |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SMS Spam Classifier** | GPT-2 Small | 124M | 768 | 12 / 12 | `spam_classifier.pth` (~548 MB) | Binary Classification (Ham vs. Spam) |
| **Instruction Assistant** | GPT-2 Medium | 355M | 1024 | 24 / 16 | `gpt2-medium355M-sft.pth` (~1.72 GB) | Text Generation & Instruction Following |

---

## 🛠️ Project Structure

```text
DumLLM/
├── app.py                  # Gradio multi-tab web application (ZeroGPU & local compatible)
├── gpt_model.py            # Clean, self-contained GPT-2 PyTorch model implementation
├── requirements.txt        # Production runtime dependencies
├── README.md               # Project documentation
├── notebooks/              # Complete from-scratch training & fine-tuning curriculum
│   ├── ch02_Working_With_Text_Data.ipynb
│   ├── ch03_Coding_Attention_Mechanisms.ipynb
│   ├── ch04_Implementing_a_GPT_Model.ipynb
│   ├── ch05_Pretraining_On_Unlabeled_Data.ipynb
│   ├── ch06_Fine_Tuning_for_Classification.ipynb
│   └── ch07_Fine_Tuning_To_Follow_Instructions.ipynb
└── public/                 # Instruction datasets and resources
```

---

## 🚀 Getting Started

### Option 1: Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ShubhamKumarSahu-svg/DumLLM.git
   cd DumLLM
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the studio:**
   ```bash
   python app.py
   ```
   * The models will automatically download from Hugging Face Hub on first startup and be cached locally.
   * Open `http://localhost:7860` in your browser.
   * A public Gradio link (`https://xxxx.gradio.live`) will also be displayed in your console.

---

### Option 2: Run in Google Colab / Kaggle (Free GPU)

In a new notebook with GPU accelerator enabled (e.g. Kaggle T4 or Colab T4):

```python
# 1. Clone repository
!git clone https://github.com/ShubhamKumarSahu-svg/DumLLM.git
%cd DumLLM

# 2. Install dependencies
!pip install -r requirements.txt

# 3. Start the application
!python app.py
```

---

## 🧪 Sample Prompts for Testing

### 🛡️ Tab 1: SMS Spam Classifier
* **Spam Sample:**
  ```text
  URGENT! You have won a £2000 cash prize or a holiday to Hawaii. Call 09061701461 to claim your reward immediately!
  ```
* **Legitimate (Ham) Sample:**
  ```text
  Hey, are we still meeting for dinner tonight around 7 PM? Let me know when you leave home.
  ```

### 💬 Tab 2: Instruction Assistant
* **Fact / Reasoning:**
  * **Instruction:** `What is photosynthesis and why is it essential for life on Earth?`
* **Style Transfer:**
  * **Instruction:** `Rewrite the following message into a polite, professional corporate email.`
  * **Context / Input:** `Send me the reports right now, I need them before 5 PM.`
* **Creative Writing:**
  * **Instruction:** `Write a four-line rhyming poem about deep space and distant galaxies.`

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).

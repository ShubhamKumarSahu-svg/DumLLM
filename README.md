# 🧠 DumLLM: Dual-Model Studio

DumLLM is a project demonstrating end-to-end training, fine-tuning, and deployment of custom GPT-2 architectures built from scratch in PyTorch. 

The studio features an interactive dual-window Gradio web UI that allows testing two distinct fine-tuned checkpoints:
1. **🛡️ SMS Spam Classifier**: Fine-tuned on GPT-2 Small (124M) with a binary classification head for detecting spam vs. ham messages with real-time confidence scores.
2. **💬 Instruction Assistant**: Fine-tuned on GPT-2 Medium (355M) using instruction-following supervised fine-tuning (SFT) formatted with Alpaca-style task prompts.

---

## 🔗 Quick Links

* **🤗 Hugging Face Models Repository**: [shubhamisok/dumllm-models](https://huggingface.co/shubhamisok/dumllm-models)
* **🚀 Hugging Face Space (Live Demo / Future Deployment)**: [shubhamisok/dumllm-studio](https://huggingface.co/spaces/shubhamisok/dumllm-studio)
* **💻 GitHub Repository**: [ShubhamKumarSahu-svg/DumLLM](https://github.com/ShubhamKumarSahu-svg/DumLLM)

---

## 📦 Model Checkpoints

Both models are hosted on the Hugging Face Hub under [`shubhamisok/dumllm-models`](https://huggingface.co/shubhamisok/dumllm-models) and are downloaded automatically on first launch via `huggingface_hub`:

| Model Name | Base Architecture | Parameters | Checkpoint File | Task |
| :--- | :--- | :--- | :--- | :--- |
| **SMS Spam Classifier** | GPT-2 Small | 124M | `spam_classifier.pth` (~548 MB) | Binary Classification (Ham / Spam) |
| **Instruction Assistant** | GPT-2 Medium | 355M | `gpt2-medium355M-sft.pth` (~1.72 GB) | Text Generation & Instruction Following |

---

## 🛠️ Architecture & Project Structure

```text
DumLLM/
├── app.py                  # Gradio multi-tab web application
├── gpt_model.py            # Self-contained GPT-2 model definition & text generator
├── requirements.txt        # Minimal runtime dependencies
├── notebooks/              # Jupyter notebooks detailing the from-scratch training
│   ├── ch02_Working_With_Text_Data.ipynb
│   ├── ch03_Coding_Attention_Mechanisms.ipynb
│   ├── ch04_Implementing_a_GPT_Model.ipynb
│   ├── ch05_Pretraining_On_Unlabeled_Data.ipynb
│   ├── ch06_Fine_Tuning_for_Classification.ipynb
│   └── ch07_Fine_Tuning_To_Follow_Instructions.ipynb
└── README.md
```

---

## 🚀 Running Locally

### 1. Clone the repository
```bash
git clone https://github.com/ShubhamKumarSahu-svg/DumLLM.git
cd DumLLM
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the studio
```bash
python app.py
```

* The app will download model weights directly from Hugging Face on the first run (and cache them locally for subsequent runs).
* Access the interface in your browser at: `http://localhost:7860`.
* A free public Gradio share link (`https://xxxx.gradio.live`) will also be generated in the terminal.

---

## 🌐 Running on Cloud Platforms (Free GPU)

### Run in Google Colab / Kaggle
```python
# Clone the repository
!git clone https://github.com/ShubhamKumarSahu-svg/DumLLM.git
%cd DumLLM

# Install minimal requirements
!pip install -r requirements.txt

# Run the app
!python app.py
```

---

## 📜 License
This project is licensed under the MIT License.

import sys
import torch
import tiktoken
import gradio as gr
from huggingface_hub import hf_hub_download

from gpt_model import GPTModel, generate, text_to_token_ids, token_ids_to_text

# select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = tiktoken.encoding_for_model("gpt2")

HF_REPO_ID = "shubhamisok/dumllm-models"

print("Downloading model files from Hugging Face Hub...")

spam_weights_path = hf_hub_download(repo_id=HF_REPO_ID, filename="spam_classifier.pth")
instruct_weights_path = hf_hub_download(repo_id=HF_REPO_ID, filename="gpt2-medium355M-sft.pth")

# load sms classifier
SPAM_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
    "emb_dim": 768,
    "n_layers": 12,
    "n_heads": 12,
}
spam_model = GPTModel(SPAM_CONFIG)

spam_model.out_head = torch.nn.Linear(SPAM_CONFIG["emb_dim"], 2)
spam_model.load_state_dict(torch.load(spam_weights_path, map_location=device))
spam_model.to(device)
spam_model.eval()

# load instruction model
INSTRUCT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
    "emb_dim": 1024,
    "n_layers": 24,
    "n_heads": 16,
}
instruct_model = GPTModel(INSTRUCT_CONFIG)
instruct_model.load_state_dict(torch.load(instruct_weights_path, map_location=device))
instruct_model.to(device)
instruct_model.eval()

print("Model files downloaded and loaded successfully.")


def classify_sms(sms_text, max_length=120):
    if not sms_text.strip():
        return {"Not Spam (Ham)": 0.0, "Spam": 0.0}

    token_ids = tokenizer.encode(sms_text)
    token_ids = token_ids[:min(max_length, 1024)]
    token_ids += [50256] * (max_length - len(token_ids))

    input_tensor = torch.tensor(token_ids, device=device).unsqueeze(0)  

    with torch.inference_mode():
        logits = spam_model(input_tensor)[:, -1, :]
        probs = torch.softmax(logits, dim=-1)[0]

    return {"Not Spam (Ham)": float(probs[0]), "Spam": float(probs[1])}

def answer_instruction(instruction_text, optional_context="", max_token=120, temperature=0.7):
    if not instruction_text.strip():
        return "Please provide an instruction."

    prompt = (
        f"Below is an instruction that describes a task.\nWrite a response that appropriately completes the request.\n\n### Instruction:\n{instruction_text}"
    )

    if optional_context.strip():
        prompt += f"\n\n### Input:\n{optional_context}"
    prompt += "\n\n### Response:\n"

    idx = text_to_token_ids(prompt, tokenizer).to(device)
    token_ids = generate(
        model=instruct_model,
        idx=idx,
        max_new_tokens=int(max_token),
        context_size=1024,
        temperature=float(temperature),
        top_k=50,
        eos_id=50256,
    )

    full_output = token_ids_to_text(token_ids, tokenizer)

    if "### Response:" in full_output:
        return full_output.split("### Response:")[-1].strip()
    return full_output[len(prompt):].strip()

# Gradio Interface

with gr.Blocks(title="DumLLM Models Studio") as demo:
    gr.Markdown(
        "# DumLLM Dual Models Studio"
    )

    with gr.Tabs():

        # SMS Spam Classifier Tab
        with gr.Tab("SMS Spam Classifier"):
            gr.Markdown(
                "### Detect spam messages using a fine-tuned GPT-2 model."
            )

            with gr.Row():
                with gr.Column():
                    sms_input = gr.Textbox(
                        label="Enter SMS Text",
                        placeholder="Type your SMS message here...",
                        lines=4,
                    )
                    classify_button = gr.Button("Classify SMS")

                with gr.Column():
                    sms_output = gr.Label(label="Spam Classification Result")

            gr.Examples(
                examples=[
                    ["Congratulations! You've won a $1000 gift card. Click here to claim now!"],
                    ["Hey, are we still on for lunch tomorrow?"],
                    ["URGENT! Your account has been compromised. Reset your password immediately."],
                    ["Don't forget to submit the report by EOD."],
                ],
                inputs=sms_input,
            )
            classify_button.click(classify_sms, inputs=sms_input, outputs=sms_output)

        # Instruction Answering Tab
        with gr.Tab("Instruction Answering"):
            gr.Markdown(
                "### Ask questions to the model and get responses using a fine-tuned GPT-2 Medium model."
            )

            with gr.Row():
                with gr.Column():
                    instruction_box = gr.Textbox(label="Enter Question", placeholder="Type your question here...", lines=3)
                    context_box = gr.Textbox(label="Optional Context", placeholder="Provide additional context (optional)...", lines=2)

                    with gr.Accordion("Parameters", open=False):
                        max_token_slider = gr.Slider(
                            label="Max Tokens",
                            minimum=20,
                            maximum=256,
                            value=256,
                            step=10
                        )
                        temperature_slider = gr.Slider(
                            label="Temperature",
                            minimum=0.0,
                            maximum=1.5,
                            value=0.7,
                            step=0.05
                        )

                    generate_button = gr.Button("Generate Answer", variant="primary")

                with gr.Column():
                    response_box = gr.Textbox(label="Model Response", lines=9)

                    gr.Examples(
                        examples=[
                            ["What is the capital of France?", ""],
                            ["Explain the theory of relativity in simple terms.", ""],
                            ["Write a short poem about the ocean.", ""],
                            ["Rewrite this in a polite tone.", "Send me the report now."],
                        ],
                        inputs=[instruction_box, context_box]
                    )

            generate_button.click(
                answer_instruction,
                inputs=[instruction_box, context_box, max_token_slider, temperature_slider],
                outputs=response_box
            )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), share=True)
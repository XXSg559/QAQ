import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import argparse

# Usage:
#   To score both a strong and a weak model (for Diff-based selection), run
#   this script twice, chaining the output of the first run into the second
#   so both models' RMI end up as columns in the same file:
#
#     python rmi_math.py --data_path raw.json --model_name <strong_model> --output_path with_strong.json
#     python rmi_math.py --data_path with_strong.json --model_name <weak_model> --output_path with_both.json
#
# If you need an HF mirror (e.g. in mainland China), set it before running:
#   export HF_ENDPOINT=https://hf-mirror.com

DEFAULT_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
DEFAULT_RMI_INSTRUCTION_TEMPLATE = (
    "TASK: Given an answer, generate the most likely math problem "
    "that this answer is responding to. If the inferred question is outside "
    "mathematics, respond with \"INVALID\".\nAnswer:\n{answer}"
)

# ================= Config =================
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True, help="Path to the input dataset")
parser.add_argument("--output_path", type=str, default="data/math/OpenR1-Math-me.json", help="Path to write the output dataset")
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct", help="Model used to compute PPL")
parser.add_argument("--tokenizer", type=str, default=None,
                     help="Tokenizer to borrow a chat_template from when model_name is a base "
                          "model without one. Must be the instruct sibling of the same model "
                          "family as model_name (e.g. if the base model is "
                          "deepseek-coder-6.7b-base, pass deepseek-coder-6.7b-instruct here) "
                          "-- do not borrow a template from an unrelated model")
parser.add_argument("--question_field", type=str, default="problem", help="Dataset field name for the question")
parser.add_argument("--answer_field", type=str, default="generations", help="Dataset field name for the answer")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
parser.add_argument("--max_length", type=int, default=20480, help="Max token length for a rendered example; truncated beyond this")
# The system prompt should match the downstream reasoning-SFT template, not
# be customized just for this reverse-generation task.
parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT,
                     help="System prompt for the chat template; should match the downstream SFT setup")
parser.add_argument("--rmi_instruction_template", type=str, default=DEFAULT_RMI_INSTRUCTION_TEMPLATE,
                     help="Instruction template for the reverse-generation task; must contain an {answer} placeholder")
args = parser.parse_args()

MODEL_NAME = args.model_name
BATCH_SIZE = args.batch_size
MAX_LENGTH = args.max_length
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RMI_INSTRUCTION_TEMPLATE = args.rmi_instruction_template
SYSTEM_PROMPT = args.system_prompt
# ============================================

print(f"Loading model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.chat_template is None:
    if args.tokenizer is None:
        raise ValueError(
            f"{MODEL_NAME} has no chat_template (likely a base model). "
            f"Pass --tokenizer with its corresponding "
            f"instruct sibling (same model family) to borrow a template from."
        )
    print(f"[warn] {MODEL_NAME} has no chat_template; borrowing one from {args.tokenizer}")
    template_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    tokenizer.chat_template = template_tokenizer.chat_template

tokenizer.padding_side = "right"

class PPLModelWrapper(nn.Module):
    """Computes log PPL (cross entropy) over a masked span."""
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, input_ids, attention_mask, target_mask):
        """
        Args:
            input_ids: [B, L]
            attention_mask: [B, L]
            target_mask: [B, L] - 1 for positions to include in the loss, 0 to ignore
        Returns:
            log_ppls: [B]
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        logits = outputs.logits  # [B, L, V]

        # Shift for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_target_mask = target_mask[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction='none')
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ).view(shift_labels.size())

        masked_loss = per_token_loss * shift_target_mask
        sum_loss = masked_loss.sum(dim=1)
        count = shift_target_mask.sum(dim=1).clamp(min=1)
        log_ppls = sum_loss / count

        return log_ppls


print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    trust_remote_code=True
)

model = PPLModelWrapper(base_model)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)

model.to(DEVICE)
model.eval()


def tokenize_with_target_mask(texts, target_starts, max_length=MAX_LENGTH):
    """
    Build a target_mask from character-index target_starts.
    """
    encodings = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True
    )

    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    offset_mapping = encodings['offset_mapping']

    batch_size, seq_len = input_ids.shape
    target_mask = torch.zeros_like(input_ids, dtype=torch.float)

    for i in range(batch_size):
        if target_starts[i] is None:
            # None means compute loss over the full text
            target_mask[i] = attention_mask[i].float()
        else:
            char_start = target_starts[i]
            # Find the token index corresponding to char_start
            for j, (start, end) in enumerate(offset_mapping[i].tolist()):
                # Include a token in the loss once its start position is
                # past char_start (and it isn't padding). This is normally
                # the first token of the assistant's reply.
                if start >= char_start and attention_mask[i, j] == 1:
                    target_mask[i, j] = 1.0

    return input_ids, attention_mask, target_mask


def compute_ppl_batch(texts, target_starts):
    input_ids, attention_mask, target_mask = tokenize_with_target_mask(texts, target_starts)

    input_ids = input_ids.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)
    target_mask = target_mask.to(DEVICE)

    with torch.no_grad():
        log_ppls = model(input_ids, attention_mask, target_mask)

    return log_ppls.cpu().numpy().tolist()

def get_model_short_name(model_id):
    """Extract a short name from an HF model id, e.g. Qwen/Qwen2.5-Coder-7B -> Qwen2.5-Coder-7B"""
    return model_id.split("/")[-1]

def compute_raw_rmi_batch(examples):
    """
    Compute RMI using the chat template for PPL(Q|A).
    """
    batch_size = len(examples[args.question_field])

    # -------------------------------------------------------
    # 1. PPL(Q): PPL of the raw question Q (prior)
    # -------------------------------------------------------
    # For PPL(Q), it's enough to compute the loss over the raw text.
    texts_q = []
    target_starts_q = []

    for i in range(batch_size):
        q = examples[args.question_field][i]

        messages_q = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q}
        ]

        full_str_q = tokenizer.apply_chat_template(
            messages_q,
            tokenize=False,
            add_generation_prompt=False
        )

        # Find where Q starts: render a prefix with an empty user turn
        messages_prefix = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ""}
        ]
        prefix_str = tokenizer.apply_chat_template(
            messages_prefix,
            tokenize=False,
            add_generation_prompt=False
        )

        texts_q.append(full_str_q)
        target_starts_q.append(len(prefix_str))


    # -------------------------------------------------------
    # 2. PPL(Q|A): likelihood of predicting Q given A, using the chat template
    # -------------------------------------------------------
    texts_a_q = []
    target_starts_a_q = []

    for i in range(batch_size):
        q = examples[args.question_field][i]
        a = examples[args.answer_field][i]

        # Build the conversation
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": RMI_INSTRUCTION_TEMPLATE.format(answer=a)},
            {"role": "assistant", "content": q}
        ]

        # Step A: render the full conversation string (user + assistant)
        full_str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            # for qwen3
            enable_thinking=False
        )

        # Step B: render just the user-prompt string, to locate the mask boundary
        # (add_generation_prompt=True typically appends an <|assistant|>-style marker)
        prompt_str = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
            # for qwen3
            enable_thinking=False
        )

        texts_a_q.append(full_str)

        # The target (Q) starts right where the prompt string ends, so we
        # can mask out the loss over the prompt portion.
        target_starts_a_q.append(len(prompt_str))

    # Compute PPL
    log_ppls_q = compute_ppl_batch(texts_q, target_starts_q)
    log_ppls_q_a = compute_ppl_batch(texts_a_q, target_starts_a_q)

    # RMI = log P(Q) - log P(Q|A)
    # Note: the loss functions above return cross entropy (i.e. -log P),
    # which is positive. So:
    # RMI (information gain) = H(Q) - H(Q|A) ~= Loss(Q) - Loss(Q|A)
    # If Loss(Q|A) is small (the model can easily guess Q), RMI is large.
    raw_rmis = [log_ppls_q[i] - log_ppls_q_a[i] for i in range(batch_size)]

    short_name = get_model_short_name(MODEL_NAME)

    return {
        f"{short_name}_log_PPL_Q": log_ppls_q,
        f"{short_name}_log_PPL_Q_A": log_ppls_q_a,
        f"{short_name}_RMI": raw_rmis
    }


def main():
    print(f"Loading dataset: {args.data_path}")
    ds = load_dataset("json", data_files=args.data_path, split="train")

    print(f"Computing RMI with Chat Template (batch_size={BATCH_SIZE})...")
    ds_with_rmi = ds.map(
        compute_raw_rmi_batch,
        batched=True,
        batch_size=BATCH_SIZE,
        desc="Computing RMI"
    )

    print(f"Saving to {args.output_path}...")
    ds_with_rmi.to_json(args.output_path)
    print("Done!")


if __name__ == "__main__":
    main()

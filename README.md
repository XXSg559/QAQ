# QAQ: Reverse Mutual Information (RMI) for Data Selection

Code accompanying:

> Jiayin Lei, Ming Ma, Yunxi Duan, Chenxi Li, Tianming Yang.
> **QAQ: Bidirectional Semantic Coherence for Selecting High-Quality Synthetic Code Instructions.**
> EMNLP 2026 (Main track).

## Method

RMI scores a synthetic `(Q, A)` pair by how well the answer explains the
question, rather than how hard the answer is to generate:

```
RMI(Q, A) = log PPL(Q) - log PPL(Q | A)
```

`PPL(Q | A)` is estimated via a reverse-generation task: the model is asked
to infer the most likely question given only the answer, and we measure the
perplexity of the true question under that conditioning. This is the
opposite direction from IFD-style metrics (`PPL(A|Q)`), which measure how
hard the answer is to produce rather than how well it is supported by the
question.

In the paper, RMI is combined with model disagreement: the same data is
scored with a strong and a weak model, and `Diff = rank_strong - rank_weak`
is used to select samples that are valid (strong model finds them coherent)
and learnable (weak model does not already find them easy).

## Results

Fine-tuning DeepSeek-Coder-6.7B-Base on subsets of WarriorCoder (310K
samples), evaluated pass@1 with greedy decoding on HumanEval(+) / MBPP(+):

| Method | % Data | HumanEval | HumanEval+ | MBPP | MBPP+ |
|---|---|---|---|---|---|
| Full Data | 100 | 78.05 | 72.56 | 71.69 | 59.52 |
| Random | 25 | 73.78 | 69.51 | 68.52 | 57.67 |
| IFD | 25 | 71.95 | 66.46 | 64.81 | 54.76 |
| RDS+ | 25 | 76.83 | 71.34 | 71.69 | 58.99 |
| SCAR | 25 | 75.00 | 70.73 | 70.63 | 57.67 |
| **QAQ (Ours)** | 25 | **77.44** | 71.95 | 71.43 | 58.73 |

With only 25% of the data, QAQ matches or beats every other 25%-data
baseline and comes within 1-2 points of full-data training.

On math reasoning (RMI transferred to a different domain, evaluated on
MATH-500 / GPQA-Diamond):

| Method | Data Size | MATH-500 | GPQA-Diamond |
|---|---|---|---|
| Full Data | 100% | 90.6 | 42.4 |
| Random | 25% | 87.8 | 33.3 |
| IFD | 25% | 87.2 | 37.4 |
| SCAR | 25% | 86.6 | 30.8 |
| RDS+ | 25% | 85.6 | 37.9 |
| **QAQ (Ours)** | 25% | **91.6** | 39.9 |

See the paper for the full set of results, ablations (stratification,
disagreement vs. consensus), and analysis.

### Reproducing the WarriorCoder fine-tuning

- Base model: DeepSeek-Coder-6.7B-Base
- Data: a reproduction of WarriorCoder (329K instruction-response pairs),
  filtered to samples under 2048 tokens (~310K remaining)
- Framework: LlamaFactory, 3 epochs
- LR scheduler: cosine decay, warmup ratio 0.2
- Batch size / learning rate scale with the selected data size:

  | Data size | Batch size | Learning rate |
  |---|---|---|
  | 100% | 512 | 1.2e-4 |
  | 50% | 256 | 0.8e-4 |
  | 25% | 256 | 0.4e-4 |

- RMI scoring models: strong = DeepSeek-Coder-6.7B-Base, weak = Qwen3-0.6B
- Evaluation: HumanEval, HumanEval+, MBPP, MBPP+, greedy decoding (pass@1)

### Reproducing the math domain setup

- Base model: Qwen2.5-Math-7B-Instruct
- Data: OpenR1-Math-220k, filtered to a 16,384-token cutoff (~91K samples remaining)
- Training configuration: follows [open-r1/OpenR1-Qwen-7B](https://huggingface.co/open-r1/OpenR1-Qwen-7B); see [huggingface/open-r1#545](https://github.com/huggingface/open-r1/issues/545)
  for background. The original recipe file
  (`recipes/OpenR1-Qwen-7B/sft/config.yaml` in the open-r1 repo) is not
  available at that path as of writing; we recall referencing it from an
  open-r1 issue/PR but could not relocate the exact link
- "Full Data" in the results table above is the official OpenR1-Qwen-7B checkpoint, not a run we trained ourselves
- RMI scoring models: strong = Qwen2.5-Math-7B-Instruct, weak = Qwen2.5-Math-1.5B-Instruct, same 10-bin stratification as the code domain
- Evaluation: MATH-500, GPQA-Diamond, pass@1, via LightEval
- Checkpoint selection: math domain results use the **last training step**
  for every method, not the best checkpoint. This differs from the code
  domain, where checkpoints are selected by best validation performance
  (see the ablation table in the paper). Keep this in mind when comparing
  numbers across domains or reproducing either one.

## Requirements

```
torch
transformers
datasets
```

## Usage

`rmi_math.py` computes `PPL(Q)` and `PPL(Q|A)` for one model at a time and
appends them as columns (`{model}_RMI`, etc.) to the dataset.

To get both a strong and a weak model's RMI in one file (needed for
Diff-based selection), run the script twice, chaining the output of the
first run into the second:

```bash
python rmi_math.py \
  --data_path raw.json \
  --model_name <strong_model> \
  --output_path with_strong.json

python rmi_math.py \
  --data_path with_strong.json \
  --model_name <weak_model> \
  --output_path with_both.json
```

The input dataset is a JSON/JSONL file with a question field and an answer
field (configurable via `--question_field` / `--answer_field`, default
`problem` / `generations`).

### Base models

If `--model_name` points to a base model (no `chat_template`), pass
`--tokenizer` with its instruct sibling from the **same model family** to
borrow a template from (e.g. base = `deepseek-coder-6.7b-base` ->
`--tokenizer deepseek-coder-6.7b-instruct`). Borrowing a template from an
unrelated model family is not valid and will silently give incorrect PPL
values.

### Adapting to a different domain

The script is not math-specific. To reuse it for another domain (e.g.
code), just override the prompt-related arguments:

- `--system_prompt`: should match whatever system prompt the downstream
  SFT training will use, not a prompt customized for this scoring task.
- `--rmi_instruction_template`: the reverse-generation instruction; must
  contain an `{answer}` placeholder.
- `--max_length`: max token length per rendered example.

### Selecting by model disagreement (Diff)

Once a file has RMI columns for both a strong and a weak model, the Diff
selection itself is a few lines:

```python
import pandas as pd

df = pd.read_json("with_both.json", lines=True)
rank_strong = df["<strong_model>_RMI"].rank(pct=True)
rank_weak = df["<weak_model>_RMI"].rank(pct=True)
df["diff_<strong_model>_<weak_model>"] = rank_strong - rank_weak

selected = df.nlargest(int(len(df) * 0.25), "diff_<strong_model>_<weak_model>")
```

## Citation

Coming soon.

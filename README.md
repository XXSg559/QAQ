# QAQ: Reverse Mutual Information (RMI) for Data Selection

Code accompanying:

> Jiayin Lei, Ming Ma, Yunxi Duan, Chenxi Li, Tianming Yang.
> **QAQ: Bidirectional Semantic Coherence for Selecting High-Quality Synthetic Code Instructions.**
> ACL 2025.

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
df["diff"] = rank_strong - rank_weak

selected = df.nlargest(int(len(df) * 0.25), "diff")
```

## Citation

```bibtex
@inproceedings{lei2025qaq,
  title     = {QAQ: Bidirectional Semantic Coherence for Selecting High-Quality Synthetic Code Instructions},
  author    = {Lei, Jiayin and Ma, Ming and Duan, Yunxi and Li, Chenxi and Yang, Tianming},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics},
  year      = {2025}
}
```

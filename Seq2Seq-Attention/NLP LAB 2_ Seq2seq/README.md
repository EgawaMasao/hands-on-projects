
# NLP Lab 2 — Seq2Seq Neural Machine Translation

## Files

- `lab2_handout.tex`: LaTeX handout for students
- `lab2_seq2seq.ipynb`: student notebook skeleton
- `seq2seq_model.py`: reference starter code
- `instructor/solutions_lab2.ipynb`: instructor solution notebook
- `data/`: toy English–Vietnamese parallel corpus

## Setup

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Open `lab2_seq2seq.ipynb` in Jupyter Notebook or VS Code.

## Student Tasks

1. Load parallel data.
2. Build vocabularies.
3. Encode and pad sentences.
4. Implement the encoder.
5. Implement the decoder.
6. Implement Seq2Seq training with teacher forcing.
7. Implement greedy decoding.
8. Evaluate on the dev set.

## Expected Behavior

A correct implementation should show decreasing training loss and produce sensible
memorized translations on this toy corpus.

## Submission

Submit:

- `lab2_seq2seq.ipynb`
- `report.pdf`

## Suggested Extension

Compare training with and without teacher forcing.

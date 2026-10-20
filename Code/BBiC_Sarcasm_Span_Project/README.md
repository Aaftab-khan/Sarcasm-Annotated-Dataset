# BBiC: BERT–BiLSTM–CRF Sarcasm Span Extraction

This repository implements a span-extraction pipeline using fast BERT WordPiece tokenization, saved BIO supervision for annotated sarcastic tweets, all-`O` supervision for non-sarcastic tweets, a fine-tuned BERT encoder, a bidirectional LSTM, token emission scores, and a constrained linear-chain CRF trained by negative log-likelihood and decoded with Viterbi.


### Use of the newly supplied workbooks and annotation preservation

The workbooks in `data/source/TrainAnnotatedSpansBIO.xlsx` and `data/source/TestAnnotatedSpansBIO.xlsx`.

The supplied training workbook contains 3,468 data rows, including one blank-text non-sarcastic row. That unusable row remains in the source workbook but is excluded from the model-ready training CSV, leaving 3,467 rows (867 sarcastic and 2,600 non-sarcastic).




## Project layout

```text
BBiC_Sarcasm_Span_Project/
├── README.md
├── requirements.txt
├── config.yaml
├── train.py
├── evaluate.py
├── predict.py
├── run_ablation.py
├── audit_data.py
├── src/
│   ├── configuration.py
│   ├── crf.py
│   ├── data.py
│   ├── encoding_repair.py
│   ├── engine.py
│   ├── metrics.py
│   ├── model.py
│   ├── preprocessing.py
│   ├── reproducibility.py
│   └── spans.py
├── tests/
│   ├── test_crf.py
│   ├── test_data.py
│   ├── test_encoding_repair.py
│   ├── test_evaluation_pipeline.py
│   ├── test_metrics.py
│   ├── test_model.py
│   ├── test_real_data.py
│   └── test_spans.py
├── data/
│   ├── processed/
│   │   ├── TrainAnnotatedSpansBIO.csv
│   │   └── TestAnnotatedSpansBIO.csv
│   └── source/
│       ├── TrainAnnotatedSpansBIO.xlsx
│       ├── TestAnnotatedSpansBIO.xlsx
│       ├── train_original.csv
│       └── test_original.csv
└── outputs/
```

## Model architecture

```text
Tweet text
   │
   ├── Existing Preprocessed Tweet for annotated sarcastic examples
   └── Conservative preprocessing for raw non-sarcastic examples
   │
BERT WordPiece tokenizer (maximum sequence length 128; [CLS]/[SEP] retained)
   │
BERT-base contextual encoder (768-dimensional hidden states)
   │
Dropout (0.30)
   │
Bidirectional LSTM (256 hidden units per direction)
   │
Dropout (0.30)
   │
Linear token-emission layer (O, B-SARC, I-SARC)
   │
Constrained linear-chain CRF (negative log-likelihood training)
   │
Viterbi decoding
   │
BIO sequence → character-level predicted spans
```

The CRF disallows `I-SARC` at sequence start and disallows `I-SARC` after `O`. Token metrics ignore special tokens and padding. Predicted character spans use the fast tokenizer's offsets, not whitespace-based token guessing.

## Important parameter choices



| Parameter | Project setting | Status |
|---|---:|---|
| Base encoder | `bert-base-uncased` | Explicit implementation choice. |
| Maximum length | 128 | Reported in paper |
| BERT hidden size | 768 | BERT-base representation size reported in paper |
| BiLSTM hidden units | 256 per direction | Reported hidden-unit setting; output size is 512 after concatenation |
| Dropout | 0.30 | Reported in paper |
| Optimizer | AdamW | Reported in paper |
| Learning rate | `2e-5` | Reported in paper |
| Weight decay | 0.01 | Reported in paper |
| Batch size | **32** | Confirmed configuration for this project. |
| Maximum epochs | 50 | Reported in paper |
| Scheduler | Linear warmup then linear decay | Schedule type is specified; `warmup_ratio=0.10` is an operational default |
| Validation fraction | 0.10 from the official training split | Operational default for holding out validation data from training only |
| Early-stop patience | 5 epochs | Operational default for validation-loss early stopping |
| Seed | 42 | Explicit reproducibility setting; not reported in the paper |
| Gradient clipping | 1.0 | Explicit stability setting; not reported in the paper |

The warmup ratio (`0.10`), validation fraction (`0.10`), early-stopping patience (`5`), seed (`42`), and gradient-clipping norm (`1.0`) are explicit operational defaults chosen to make the runs repeatable. 



## Environment setup

Use Python 3.10 or later. The first run needs access to the Hugging Face model/tokenizer or a local copy of `bert-base-uncased`.

### Windows PowerShell

```powershell
cd BBiC_Sarcasm_Span_Project
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

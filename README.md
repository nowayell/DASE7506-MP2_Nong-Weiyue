# DASE7506 Mini Project 2

Nong Weiyue · 3036749717

This repository selects a correct final answer from four supplied mathematical solutions. The frozen method is **E3**, an answer-group linear ranker using supplied embeddings, votes, text/numeric statistics and relative features.

- Report: [MP2 report](MP2_Selecting_Reliable_LLM_Reasoning_Report.docx)
- Frozen inference bundle: [code/runs/E3](code/runs/E3)
- Official test accuracy: **81.79871520342613% (382/467)**
- Official validation accuracy: **92.4812030075188% (738/798)**
- Local complete CPU prediction plus scoring: 1.05835 seconds; inference assets: 25,897 bytes.

## Reproduce without retraining

From `code/`, in Windows PowerShell with Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-method.txt
python check_data.py
python test_selector.py
$env:OMP_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
python predict.py --method selector.py --checkpoint runs/E3 --split test --output runs/reproduced-test-predictions.json
python score_predictions.py --split test --predictions runs/reproduced-test-predictions.json --output runs/reproduced-test-score.json
```

For other platforms, activate the virtual environment and set the same thread environment variables using the shell's syntax. Prediction uses `selector.py`; the original `submission.py` remains the unused official interface stub. Predictions saved in runs/ are evaluation outputs, never inference inputs.

## Training and evidence

Run `python train_experiments.py` from code/ to reproduce the train-only five-fold regularization search and validation comparisons (seed 20261029). Validation is not included in fitting. Test scores did not select the model. Retraining regenerates experimental assets; use the supplied E3 bundle for frozen-score reproduction.

See [experiment instructions and AI disclosure](code/EXPERIMENTS.md). `code/runs/summary.json` records all trials, convergence diagnostics and paired comparisons; `folds.npy` records folds. The saved official scoring JSON, frozen choice, timings and representative arithmetic disagreements are in `code/runs/`. `code/training.log` records the original run. Other configurations are retained as experiment artifacts; only E3 is required for inference.

`SUBMISSION_MANIFEST.json` identifies the report, custom source and assets by SHA-256. The upstream `code/PACKAGE_MANIFEST.json` remains an original-package manifest, not a manifest of these additions. The project guide and original tools are retained for attribution and verification. Preserve `code/data/LICENSE.txt` and `code/data/NOTICE.txt`.

Local timing uses Windows 11, Intel Core i7-12700H and four threads. Eligibility is determined on the official reference CPU. Substantive AI assistance and the student's contributions are disclosed in the report and experiment instructions.

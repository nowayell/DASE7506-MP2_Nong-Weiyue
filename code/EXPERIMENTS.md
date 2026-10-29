# Experiment reproduction

Run from code/ using Python 3.12:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-method.txt
.\.venv\Scripts\python.exe test_selector.py
.\.venv\Scripts\python.exe train_experiments.py
```

The trainer reads train and validation only. Five folds use seed 20261029 and are stratified by training correctness count and answer disagreement. Each configuration searches four L2 values using train OOF hard accuracy; ties prefer stronger regularization. Validation comparisons do not refit preprocessing. Every fitted scaler uses its current training fold only. L-BFGS has a 200 iteration limit; convergence diagnostics are saved.

E1: candidate embeddings; G0: answer-group mean embeddings; E2: G0 plus votes; E3: votes, text/numeric statistics, similarity and group-size interactions; E4: E3 plus conservative whole-line arithmetic checks; A1: E4 without the two explicit vote columns; A2: E4 without basic text/numeric statistics. Note A1 retains group-size interactions and group aggregation, so it is only an ablation of explicit vote columns, not all consensus information.

The arithmetic extractor only examines complete lines containing one numeric equality; it does not verify natural-language reasoning. No approximate numeric grouping is used. Unparseable candidates remain separate groups. All methods preserve their eligibility, without hand-coded test-specific rules.

Artifacts are written to runs/<configuration>/. runs/summary.json contains validation comparisons and paired bootstrap intervals on the hard subset. Accuracy there is a fraction; official submission accuracy is a percentage. The test split is not accessed by training.

Official validation evaluation (replace E3 with the selected configuration):

```powershell
$env:OMP_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
.\.venv\Scripts\python.exe predict.py --method selector.py --checkpoint runs/E3 --split validation --output runs/validation-predictions.json
.\.venv\Scripts\python.exe score_predictions.py --split validation --predictions runs/validation-predictions.json --output runs/validation-score.json
```

AI disclosure: I developed the initial experimental plan and set up the Windows environment, including dependency installation and the official data check. I reviewed the experimental results, accepted the final model choice and conclusions, and personally proofread and revised the report. OpenAI Codex substantially assisted with refining the method and features, implementing the selector and training scripts, debugging, executing experiments, checking results, analysing representative errors, and drafting and editing the report. These contributions are disclosed as substantive AI assistance. No additional language model was called during prediction. The supplied parser, scorer, runner and data were reused unchanged. The original data attribution and licenses are retained.

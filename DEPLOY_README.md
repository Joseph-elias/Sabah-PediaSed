# PediaSed — Streamlit Community Cloud deployment

This folder contains the minimum files to deploy the existing `app.py`.
The source app and trained model have not been changed.

## Deploy

1. Extract the ZIP on your computer.
2. On github.com, create a repository (for example, `pediased`) and set its visibility appropriately.
3. Use **Add file → Upload files** on the repository's `main` branch and upload the **contents** of this folder, keeping the `model_results/` subfolder intact. Do not upload only the ZIP.
4. Open https://share.streamlit.io and connect your GitHub account.
5. Choose **Create app → Yup, I have an app**. Select the repository and branch `main`, and enter `app.py` as the main file path.
6. Under **Advanced settings**, choose Python **3.11** (recommended to match the source project documentation). No secrets are needed by this version of the app.
7. Select **Deploy**. Open the app and try the assessment form and Model Information page.

## Required repository structure

```text
app.py
requirements.txt
model_results/
  best_multiclass_model.joblib
  best_multiclass_model_metadata.json
```

The included README and .gitignore are optional. A Dockerfile, `packages.txt`, training scripts, Jupyter notebooks, and the Excel datasets are **not** needed to run this deployment.

## Important healthcare data and safety notes

- The original project ZIP includes patient-level data files (`data.xlsx`, `final_dataset.xlsx`), which have **not** been included here. Do not upload them to a public repository.
- Use synthetic or de-identified examples while testing. Community Cloud is not a substitute for an institution-approved environment for personal health information.
- This predictive tool is for research / decision support. Do not deploy for clinical decisions without the necessary external validation, clinical review, privacy and regulatory approvals.
- Saved joblib models use Python pickle under the hood: deploy only model files you trust.

## Troubleshooting

- `FileNotFoundError`: verify the exact case and path `model_results/best_multiclass_model.joblib` and metadata JSON are committed to GitHub.
- `ModuleNotFoundError` or an installation failure: open Streamlit **Manage app → logs**, confirm `requirements.txt` is at the repo root, and verify Python is 3.11.
- `InconsistentVersionWarning`: the model was trained with `scikit-learn==1.7.2`; keep that package pinned.

See the official deployment guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

# UNETI14 AI Project — NYC Taxi Trip Duration

## Files
- `model.pkl`: LightGBM model trained on all 1,200,000 labeled rows.
- `preprocessing.py`: same feature engineering used for training and inference.
- `train_model.py`: reproducible training/validation and CSV export script.
- `app.py`: Streamlit demo application.
- `outputs/UNETI14_submission.csv`: predictions for the 3,000-row UNETI14 test.
- `outputs/validation_results.json`: validation metrics.

## Setup (Windows PowerShell)
1. Create a virtual environment:
   `py -m venv .venv`
2. Activate it:
   `.\.venv\Scripts\Activate.ps1`
3. Install packages:
   `python -m pip install -r requirements.txt`
4. Copy the official data files into a `data` folder:
   - `UNETI2026_TRAIN_CHUNG_1200000.csv`
   - `UNETI14_TEST_3000.csv`

## Run training again
`python train_model.py --train data\UNETI2026_TRAIN_CHUNG_1200000.csv --test data\UNETI14_TEST_3000.csv`

The training script checks the required columns, IDs, target values, and final
submission before writing the model and predictions. `trip_id` and any
unrecognized columns are excluded from model features.

## Generate the EDA and results report
After training, create the two-page report with three EDA charts and validation
RMSLE comparison:
`python generate_report.py --train data\UNETI2026_TRAIN_CHUNG_1200000.csv --results outputs\validation_results.json --output outputs\UNETI14_report.pdf`

The report leaves a team-member section to complete before submission.

## Run the mini app
`streamlit run app.py`

## Validation result from this run
- Split: 85% train / 15% validation, random_state=42.
- Target transformation: `log1p(trip_duration)`.
- LightGBM validation RMSLE: see `outputs/validation_results.json`.
- Median baseline validation RMSLE: see `outputs/validation_results.json`.
- The final model was then fit on all 1,200,000 labeled rows.

Validation RMSLE is not the official test score. The official score is unknown until the organizer evaluates the submission.

LightGBM does not require feature standardization. The shared preprocessing
instead adds distance estimates from rounded coordinates and cyclical time
features; the chosen parameters were compared on the same held-out split.
The 3,000-row team test has no labels, so its score cannot be measured or used
to tune the model without risking leakage.

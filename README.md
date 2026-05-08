# Kaggle Titanic — Survival Prediction

Predicting passenger survival on the Titanic using a Random Forest classifier. Built as a modular Python pipeline with feature engineering and automated submission generation.

**Kaggle Score: 0.78947**

## Project Structure

```
kaggle-titanic/
├── data/
│   ├── train.csv           # Training data (891 passengers)
│   ├── test.csv            # Test data (418 passengers)
│   └── submission.csv      # Generated predictions
├── notebook/
│   └── EDA.ipynb           # Exploratory Data Analysis
├── scripts/
│   ├── run.py              # Main pipeline entry point
│   ├── data_loader.py      # CSV loading
│   ├── preprocessing.py    # Feature engineering & cleaning
│   ├── model.py            # Model training & evaluation
│   └── predict.py          # Submission file generation
└── requirements.txt
```

## Feature Engineering

| Feature | Description |
|---------|-------------|
| `FamilySize` | `SibSp + Parch + 1` |
| `IsAlone` | Binary flag for solo travelers |
| `Title` | Extracted from name, normalized to `Mr`, `Miss`, `Mrs`, `Master`, `Rare` |
| `Sex_Pclass` | Interaction feature between gender and ticket class |

**Preprocessing steps:**
- Drop low-signal columns: `Cabin`, `Ticket`, `PassengerId`
- Impute `Age` by title-group median
- Impute `Fare` with median, `Embarked` with mode
- One-hot encode `Sex`, `Embarked`, `Title`

## Model

**Random Forest Classifier** with tuned hyperparameters:
- `n_estimators=200`, `max_depth=5`, `min_samples_leaf=5`, `max_features="sqrt"`

**Local evaluation (80/20 split):**
- Train accuracy: **85.7%**
- Test accuracy: **82.7%**

The final model is retrained on the full training set before generating the submission.

## Setup & Run

```bash
# Create virtual environment
python -m venv kaggle
source kaggle/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python scripts/run.py
```

This generates `data/submission.csv` ready for Kaggle upload.

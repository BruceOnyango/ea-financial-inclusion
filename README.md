# Financial Inclusion in East Africa

![tests](https://img.shields.io/badge/tests-49%20passing-brightgreen)
![coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

Across Kenya, Rwanda, Tanzania and Uganda, only about **14 in every 100 adults hold a
bank account**. This project uses survey data to predict who is likely to be excluded,
shows where the gaps are widest, and serves a live screener two ways: an interactive
Dash dashboard and a JSON prediction API.

The winning model is served both through the dashboard and through a FastAPI service,
the latter wired to a Turn.io WhatsApp journey so a field officer can screen an
individual over chat. The emphasis throughout is a defensible, end to end machine
learning workflow: from data construction through leakage safe cross validation, honest
evaluation under class imbalance, and interfaces where every number shown is one a unit
test verified.

> **Headline result.** Five classifiers reach near identical accuracy (around 85%),
> but accuracy is misleading when only 14% of people are banked. Judged on the metrics
> that matter for finding the excluded, an **SVM (RBF)** leads, correctly ranking a
> banked and an unbanked person about 86 times in 100 (ROC-AUC 0.858) and recovering
> 64% of truly banked adults.

---

## Table of contents

- [The problem](#the-problem)
- [Data](#data)
- [Quickstart](#quickstart)
- [Reproducing the model](#reproducing-the-model)
- [The dashboard](#the-dashboard)
- [The prediction API](#the-prediction-api)
- [WhatsApp screening (Turn.io)](#whatsapp-screening-turnio)
- [Testing and coverage](#testing-and-coverage)
- [Project structure](#project-structure)
- [Deployment](#deployment)
- [Methodology](#methodology)
- [Key findings](#key-findings)

---

## The problem

Access to a bank account lets households save, make payments, and build the credit
history that unlocks further finance. It is a recognised contributor to long term
economic growth. Yet across these four East African countries the majority of adults
remain unbanked. Knowing **who** is excluded, and **which factors** drive that
exclusion, turns a broad policy goal into targeted, measurable action for a bank, an
NGO, or a financial regulator.

This is framed as a binary classification task: given a person's demographic profile,
predict whether they hold a bank account (Yes = 1, No = 0).

---

## Data

| Field | Detail |
| --- | --- |
| Source | FinScope and FinAccess household surveys, 2016 to 2018 |
| Coverage | Kenya, Rwanda, Tanzania, Uganda |
| Rows | 23,524 labelled respondents |
| Target | `bank_account` (14.1% positive) |
| Features | 10 demographic attributes after selection |

The dataset is public but sits behind a free login. Download `Train.csv` from the
[Zindi competition page](https://zindi.africa/competitions/financial-inclusion-in-africa/data)
or a Kaggle mirror, rename it to `financial_inclusion.csv`, and place it in `data/`.

**Feature selection notes.** `uniqueid` is dropped as a pure identifier. `year` is
dropped because it is perfectly collinear with `country` (each country was surveyed in
a single year). `education_level` is treated as ordinal, the rest of the categoricals
as one hot, and the two numeric fields are standardised.

---

## Quickstart

**Prerequisites:** Python 3.10 or newer, and Git.

### 1. Clone and create a virtual environment

**Windows (PowerShell)**

```powershell
git clone <your-repo-url>
cd ea-financial-inclusion
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS (bash)**

```bash
git clone <your-repo-url>
cd ea-financial-inclusion
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install with the dashboard, API and dev extras

**Windows (PowerShell)**, note the quotes, PowerShell globs the brackets otherwise:

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev,dash,api]"
```

**Linux / macOS**

```bash
pip install --upgrade pip setuptools wheel
pip install -e ".[dev,dash,api]"
```

> **Extras.** `dash` (dashboard), `api` (FastAPI service), `dev` (tests), `deploy`
> (gunicorn, server only). Install `.[dev,dash,api]` locally; on the server use
> `.[dash,api,deploy]` since the box does not run tests.

> **Version note.** `scikit-learn` and `imbalanced-learn` are pinned exactly, because a
> saved model only reloads reliably in the same library version that created it. If you
> retrain the model in a different environment, match the versions in `pyproject.toml`
> to the ones that produced your `model.joblib`.

---

## Reproducing the model

The machine learning workflow lives in the training notebook
(`notebook/financial_inclusion.ipynb`, exported alongside as `.py`). Running it top to
bottom performs the full pipeline and writes two artifacts:

- `artifacts/model.joblib`, the winning pipeline refit on all data, bundled with its
  tuned decision threshold and the feature schema.
- `artifacts/results.json`, every metric, confusion matrix, ROC curve, and dashboard
  input the app renders.

Both the dashboard and the API consume these files and compute no statistics of their
own, so the notebook, the app and the API never disagree.

---

## The dashboard

An interactive Dash application with a masthead that states the problem in one plain
sentence, four headline cards, a "where the gaps are widest" breakdown, a live
individual screener, and a "how reliable is this" evidence section. Every chart carries
a two line explainer: one in plain language, one technical.

```bash
python -m app.app
```

Open **http://127.0.0.1:8050**.

---

## The prediction API

A FastAPI service exposes the model as JSON so external channels can consume it:

```bash
uvicorn api.main:app --reload
```

- `GET /financial-inclusion/api/health` returns the service status and model name.
- `POST /financial-inclusion/api/predict` takes one respondent's ten attributes and
  returns the probability, a banked / not banked verdict, and a recommended action.
- Interactive docs at `/financial-inclusion/api/docs`.

Example:

```bash
curl -s -X POST http://127.0.0.1:8000/financial-inclusion/api/predict \
  -H "Content-Type: application/json" \
  -d '{"country":"Kenya","location_type":"Rural","cellphone_access":"No","household_size":6,"age_of_respondent":24,"gender_of_respondent":"Female","relationship_with_head":"Head of Household","marital_status":"Married/Living together","education_level":"No formal education","job_type":"Farming and Fishing"}'
```

The API reuses the same model artifact and prediction logic as the dashboard, so both
channels give identical answers.

---

## WhatsApp screening (Turn.io)

The API is wired to a Turn.io WhatsApp journey. A field officer answers ten quick
questions in chat; the journey POSTs them to `/predict` and replies in seconds with the
verdict and a recommended action. The flow is:

```
WhatsApp  ->  Turn.io journey  ->  prediction API  ->  SVM model  ->  reply
```

Long category names are shortened to fit WhatsApp's list rows and mapped back to the
exact training labels server side, so the model always receives values it recognises.
This puts the model where the audience already is, rather than behind a browser.

---

## Testing and coverage

The codebase is built test first. I/O boundaries are thin, the model handling, figure
builders and API logic are pure, and a small in process fixture stands in for the real
model, so the suite runs fully offline and never needs the artifacts to test the logic.

```bash
pytest
```

This runs the full suite and prints a coverage report to the terminal and to
`htmlcov/index.html`. Coverage is enforced at **100%** by `--cov-fail-under=100` in
`pyproject.toml`, so the suite fails if a single line goes uncovered.

**Current status: 49 tests passing, 100% line and branch coverage.**

Open the HTML report:

**Windows (PowerShell)**

```powershell
start htmlcov\index.html
```

**Linux / macOS**

```bash
xdg-open htmlcov/index.html   # or: open htmlcov/index.html on macOS
```

---

## Project structure

```
ea-financial-inclusion/
├── pyproject.toml              # deps, extras, pytest and coverage config
├── README.md
├── wsgi.py                     # WSGI entry point (Dash, gunicorn)
├── asgi.py                     # ASGI entry point (API, uvicorn / gunicorn)
├── artifacts/
│   ├── model.joblib            # winning pipeline + threshold + schema
│   └── results.json            # every number the dashboard renders
├── data/
│   └── financial_inclusion.csv # the labelled survey data
├── src/fia/
│   ├── config.py               # paths, labels, model ordering
│   ├── artifacts.py            # load and validate the two artifacts
│   └── predict.py              # single record prediction logic
├── app/
│   ├── theme.py                # design tokens + one Plotly template
│   ├── components.py           # pure figure builders (no statistics)
│   ├── data.py                 # cached artifact access for the view
│   ├── app.py                  # layout + callbacks (view only)
│   └── assets/style.css        # hand written dashboard styling
├── api/
│   ├── __init__.py
│   └── main.py                 # FastAPI /health and /predict
└── tests/                      # mirrors src/, app/ and api/, fully offline
```

**Design principle:** every module pairs a thin I/O boundary with pure functions, and
the pure functions carry the tests. Model building is separate from evaluation,
evaluation is separate from presentation, and the dashboard and API share one model.

---

## Deployment

Live at **https://bruceonyango.link/financial-inclusion**.

---

## Methodology

Choices are made for defensibility, not convenience.

- **Training data construction.** The target is encoded Yes to 1 and No to 0.
  Redundant columns are removed by feature selection. Education is encoded ordinally to
  respect its natural ranking, other categoricals are one hot encoded, and numeric
  fields are standardised so the distance and gradient based models are not dominated
  by scale.
- **Leakage control.** All preprocessing lives inside each model's pipeline, so it is
  refit on the training portion of every fold and never sees validation data.
- **Cross validation.** Five models are compared with **2-fold stratified cross
  validation**, which preserves the 14% positive rate in each fold.
- **Class imbalance.** SMOTE is applied **inside** the pipeline, so synthetic minority
  examples are generated only from training data within each fold, never leaking into
  evaluation.
- **Threshold tuning.** Each model's decision threshold is chosen on out of fold
  training predictions, then applied unchanged to the held out test set.
- **Honest evaluation.** Reported figures come from a 70/30 split, scored on the
  untouched 30%. Selection is on ROC-AUC because it is threshold free and measures
  ranking skill. Accuracy is reported but treated with caution, since a model that
  predicts "no account" for everyone already scores about 86%.

---

## Key findings

- **Accuracy hides the real story.** All five models land between 85% and 86%
  accuracy, but a do nothing baseline that always predicts "no account" reaches 85.9%
  while finding zero banked people. Accuracy is close to meaningless here.
- **The ranking flips by metric.** SVM (RBF) leads on ROC-AUC (0.858), recall (0.640)
  and F1 (0.538). The Decision Tree posts the highest raw accuracy yet the worst
  recall, winning the headline number by missing the most banked people.
- **Handling imbalance works.** SMOTE and threshold tuning lift mean recall across the
  five models from about 0.30 to about 0.55, trading a few points of accuracy to
  recover far more of the people the tool exists to find.
- **The strongest signals of inclusion** are cellphone access, education level, and
  formal employment. Rural residents, people with little schooling, and those
  dependent on farming or remittances are the most excluded.

---

## License

MIT.
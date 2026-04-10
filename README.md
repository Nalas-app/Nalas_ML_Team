# 🍽️ Recipe Data Processing Pipeline

A production-grade ETL pipeline that transforms unstructured recipe text into a
clean, validated, and enriched JSON dataset.

---

## 🚀 How to Run

### Step 1 — Open a terminal in this folder
Right-click the `nalas` folder → **Open in Terminal** (or PowerShell).

### Step 2 — Install dependencies
```powershell
pip install -r requirements.txt
```

### Step 3 — Run the pipeline
```powershell
python pipeline.py --stats
```

That's it! Four output files will be created:

| File | Description |
|------|-------------|
| `clean_recipes.json` | Final cleaned dataset |
| `validation_report.csv` | All validation issues |
| `imputation_log.json` | Audit trail of imputed values |
| `pipeline_summary.json` | Machine-readable run summary |

---

## 🔧 Other Ways to Run

```powershell
# Dry run (no files written)
python pipeline.py --dry-run --stats

# Custom input file
python pipeline.py --input my_recipes.json

# Silent (no statistics printed)
python pipeline.py --no-stats

# Verbose / debug logging
python pipeline.py --stats --verbose
```

---

## 📓 Jupyter Notebooks (Step-by-Step)

Open notebooks one at a time in order:

```powershell
jupyter notebook
```

Then open the `notebooks/` folder and run them **in order**:

| Notebook | What it does |
|----------|-------------|
| `01_parsing.ipynb` | Parse raw recipe text → structured dicts |
| `02_enrichment.ipynb` | SI unit conversion, tagging, duplicate detection |
| `03_validation.ipynb` | Schema checks, missing values heatmap, outliers |
| `04_imputation.ipynb` | Fill missing values (sklearn median + rule-based) |
| `05_analytics.ipynb` | Dataset statistics, charts, final clean output |

> ⚠️ Run notebooks **01 → 05 in order**. Each saves its output for the next one.

---

## 🧪 Run Tests

```powershell
python -m pytest tests/ -v
```

Expected result: **163 passed, 0 failed**

---

## 📁 Folder Structure

```
nalas/
├── pipeline.py              ← Main pipeline (run this)
├── mock_raw_data.json       ← Input: 12 raw recipes
├── recipe_schema.json       ← JSON validation schema
├── requirements.txt         ← Python dependencies
├── README.md
│
├── src/                     ← Pipeline modules
│   ├── config.py            ← Constants, unit maps, regex
│   ├── parser.py            ← Stage 1: Parsing
│   ├── enrichment.py        ← Stage 2: SI conversion, tags, dedup
│   ├── validation.py        ← Stage 3: Schema + outlier checks
│   ├── imputation.py        ← Stage 4: Missing value imputation
│   └── analytics.py         ← Stage 5: Statistics + summary
│
├── notebooks/               ← Jupyter notebooks (one per stage)
│   ├── 01_parsing.ipynb
│   ├── 02_enrichment.ipynb
│   ├── 03_validation.ipynb
│   ├── 04_imputation.ipynb
│   └── 05_analytics.ipynb
│
├── tests/                   ← Unit tests
│   ├── test_parser.py       ← 49 tests
│   ├── test_enrichment.py   ← 40 tests
│   ├── test_validation.py   ← 25 tests
│   ├── test_imputation.py   ← 17 tests
│   └── test_analytics.py    ← 27 tests
│
└── outputs (generated)
    ├── clean_recipes.json
    ├── validation_report.csv
    ├── imputation_log.json
    └── pipeline_summary.json
```

---

## ⚙️ Pipeline Architecture

```
Raw Text  →  [1] Parse  →  [2] Enrich  →  [3] Validate  →  [4] Impute  →  [5] Analyse  →  Clean JSON
```

| Stage | Module | What happens |
|-------|--------|-------------|
| 1 | `parser.py` | Extract name, ingredients, steps, time, servings from raw text |
| 2 | `enrichment.py` | Convert to SI units, normalise ingredients, generate tags, detect duplicates |
| 3 | `validation.py` | JSON schema check, outlier detection, duplicate flagging |
| 4 | `imputation.py` | Fill missing prep-time (sklearn median), fill missing servings (default=4) |
| 5 | `analytics.py` | Compute dataset statistics, export pipeline summary |

---

## 📦 Requirements

```
pandas>=1.5
numpy>=1.23
scikit-learn>=1.1
jsonschema>=4.0
spacy>=3.4
```

Install spaCy language model (optional but recommended):
```powershell
python -m spacy download en_core_web_sm
```
> Without it, the pipeline falls back to regex-only parsing.

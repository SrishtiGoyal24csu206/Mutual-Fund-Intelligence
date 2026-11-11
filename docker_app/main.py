
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


# ---------------- PROJECT PATHS ----------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Mutual Fund Intelligence",
    description="Interactive Mutual Fund Analytics and Risk Prediction",
    version="2.0.0"
)

STATIC_DIR.mkdir(exist_ok=True)

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)


# ---------------- DATA LOADING ----------------

def load_fund_data():

    possible_files = [
        "07_scheme_performance_clean.csv",
        "07_scheme_performance.csv",
        "01_fund_master.csv"
    ]

    for filename in possible_files:

        file_path = DATA_DIR / filename

        if file_path.exists():

            try:
                df = pd.read_csv(file_path)

                # Remove accidental spaces in column names
                df.columns = df.columns.astype(str).str.strip()

                return df

            except Exception as error:
                print(f"Error loading {filename}: {error}")

    return pd.DataFrame()


def clean_records(df):

    df = df.replace([np.inf, -np.inf], np.nan)

    return (
        df.astype(object)
        .where(pd.notna(df), None)
        .to_dict(orient="records")
    )


def find_column(df, possible_names):

    lookup = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for name in possible_names:

        if name.lower() in lookup:
            return lookup[name.lower()]

    return None


# ---------------- ML MODEL LOADING ----------------

def load_models():

    classifier_path = MODEL_DIR / "risk_classifier.pkl"
    preprocessor_path = MODEL_DIR / "risk_preprocessor.pkl"

    if not classifier_path.exists():
        return None, None

    if not preprocessor_path.exists():
        return None, None

    classifier = joblib.load(classifier_path)
    preprocessor = joblib.load(preprocessor_path)

    return classifier, preprocessor


# ---------------- PREDICTION INPUT ----------------

class RiskInput(BaseModel):

    beta: float = 1.0
    sharpe_ratio: float = 0.5
    sortino_ratio: float = 0.7
    std_dev_ann_pct: float = 15.0
    max_drawdown_pct: float = -20.0
    aum_crore: float = Field(default=1000, ge=0)
    expense_ratio_pct: float = Field(default=1.0, ge=0)
    morningstar_rating: int = Field(default=4, ge=1, le=5)

    category: str = "Equity"
    plan: str = "Direct"
    fund_house: str = "Other"


# ---------------- HOME PAGE ----------------

@app.get("/", response_class=HTMLResponse)
def home():

    html_file = STATIC_DIR / "index.html"

    if not html_file.exists():
        return "<h1>Mutual Fund Intelligence</h1><p>Dashboard file not found.</p>"

    return html_file.read_text(encoding="utf-8")


# ---------------- HEALTH CHECK ----------------

@app.get("/api/health")
def health():

    return {
        "status": "running",
        "project": "Mutual Fund Intelligence"
    }


# ---------------- OVERVIEW ----------------

@app.get("/api/overview")
def overview():

    df = load_fund_data()

    if df.empty:
        return {
            "available": False,
            "message": "Dataset not found."
        }

    category_col = find_column(df, ["category", "scheme_category"])
    house_col = find_column(df, ["fund_house", "fund_house_name", "amc"])

    return {
        "available": True,
        "total_schemes": int(len(df)),
        "fund_houses": (
            int(df[house_col].nunique())
            if house_col else None
        ),
        "categories": (
            int(df[category_col].nunique())
            if category_col else None
        ),
        "columns": list(df.columns)
    }


# ---------------- FUND FILTER OPTIONS ----------------

@app.get("/api/funds/filters")
def fund_filter_options():

    df = load_fund_data()

    if df.empty:
        return {
            "categories": [],
            "fund_houses": [],
            "risk_levels": []
        }

    category_col = find_column(df, ["category", "scheme_category"])
    house_col = find_column(df, ["fund_house", "fund_house_name", "amc"])
    risk_col = find_column(df, ["risk_level", "risk", "risk_category"])

    def unique_values(column):

        if column is None:
            return []

        return sorted(
            df[column].dropna().astype(str).unique().tolist()
        )

    return {
        "categories": unique_values(category_col),
        "fund_houses": unique_values(house_col),
        "risk_levels": unique_values(risk_col)
    }


# ---------------- FUND EXPLORER ----------------

@app.get("/api/funds")
def get_funds(
    search: str = "",
    category: str = "",
    fund_house: str = "",
    risk: str = "",
    sort_by: str = "",
    order: str = "desc",
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=50, ge=1, le=300)
):

    df = load_fund_data()

    if df.empty:
        return {
            "total": 0,
            "page": page,
            "limit": limit,
            "rows": [],
            "columns": []
        }

    # Identify available columns
    scheme_col = find_column(
        df, ["scheme_name", "fund_name", "scheme"]
    )

    category_col = find_column(
        df, ["category", "scheme_category"]
    )

    house_col = find_column(
        df, ["fund_house", "fund_house_name", "amc"]
    )

    risk_col = find_column(
        df, ["risk_level", "risk", "risk_category"]
    )

    # Search across all text columns
    if search.strip():

        text_columns = df.select_dtypes(
            include=["object", "string"]
        ).columns

        mask = pd.Series(False, index=df.index)

        for col in text_columns:

            mask |= df[col].astype(str).str.contains(
                search.strip(),
                case=False,
                na=False,
                regex=False
            )

        df = df[mask]

    # Category filter
    if category and category_col:

        df = df[
            df[category_col].astype(str).str.lower()
            == category.lower()
        ]

    # Fund house filter
    if fund_house and house_col:

        df = df[
            df[house_col].astype(str).str.lower()
            == fund_house.lower()
        ]

    # Risk filter
    if risk and risk_col:

        df = df[
            df[risk_col].astype(str).str.lower()
            == risk.lower()
        ]

    # Sorting
    if sort_by and sort_by in df.columns:

        numeric_values = pd.to_numeric(
            df[sort_by], errors="coerce"
        )

        if numeric_values.notna().any():

            df = df.assign(
                _sort_value=numeric_values
            ).sort_values(
                "_sort_value",
                ascending=(order.lower() == "asc"),
                na_position="last"
            ).drop(columns="_sort_value")

        else:

            df = df.sort_values(
                sort_by,
                ascending=(order.lower() == "asc"),
                na_position="last"
            )

    total = len(df)

    start = (page - 1) * limit

    df = df.iloc[start:start + limit]

    return {
        "total": int(total),
        "page": page,
        "limit": limit,
        "total_pages": int(np.ceil(total / limit)) if total else 0,
        "columns": list(df.columns),
        "rows": clean_records(df)
    }


# ---------------- RISK PREDICTION ----------------

@app.post("/api/predict-risk")
def predict_risk(data: RiskInput):

    try:

        classifier, preprocessor = load_models()

        if classifier is None:
            raise HTTPException(
                status_code=503,
                detail="Saved model files not found."
            )

        input_data = pd.DataFrame([data.model_dump()])

        processed_data = preprocessor.transform(input_data)

        prediction = classifier.predict(processed_data)

        result = {
            "predicted_class": int(prediction[0])
        }

        if hasattr(classifier, "predict_proba"):

            probabilities = classifier.predict_proba(
                processed_data
            )[0]

            result["probabilities"] = {
                str(int(cls)): float(prob)
                for cls, prob in zip(
                    classifier.classes_,
                    probabilities
                )
            }

        result["message"] = (
            "Educational model output, not investment advice."
        )

        return result

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=422,
            detail=f"Prediction failed: {str(error)}"
        )



    
# ---------------- NAV ANALYSIS ----------------

NAV_DATASETS = {
    "Axis Bluechip": "axis_bluechip_live_nav.csv",
    "HDFC Top 100": "hdfc_top100_live_nav.csv",
    "ICICI Bluechip": "icici_bluechip_live_nav.csv",
    "Kotak Bluechip": "kotak_bluechip_live_nav.csv",
    "Nippon Large Cap": "nippon_largecap_live_nav.csv",
    "SBI Bluechip": "sbi_bluechip_live_nav.csv"
}


def load_nav_dataset(fund_name):

    filename = NAV_DATASETS.get(fund_name)

    if not filename:
        raise HTTPException(
            status_code=404,
            detail="Selected NAV dataset not found."
        )

    file_path = DATA_DIR / filename

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"NAV file missing: {filename}"
        )

    df = pd.read_csv(file_path)

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
    )

    if not {"date", "nav"}.issubset(df.columns):
        raise HTTPException(
            status_code=422,
            detail="NAV dataset must contain date and nav columns."
        )

    df["date"] = pd.to_datetime(
        df["date"], errors="coerce"
    )

    df["nav"] = pd.to_numeric(
        df["nav"], errors="coerce"
    )

    df = (
        df.dropna(subset=["date", "nav"])
          .sort_values("date")
          .drop_duplicates(subset=["date"], keep="last")
    )

    return df


@app.get("/api/nav/options")
def nav_options():

    available = []

    for fund_name, filename in NAV_DATASETS.items():

        if (DATA_DIR / filename).exists():
            available.append(fund_name)

    return {
        "funds": available,
        "total_available": len(available)
    }


@app.get("/api/nav/data")
def nav_data(
    fund: str,
    start_date: str = "",
    end_date: str = ""
):

    df = load_nav_dataset(fund)

    start = None
    end = None

    if start_date:
        start = pd.to_datetime(start_date, errors="coerce")

        if pd.isna(start):
            raise HTTPException(
                status_code=400,
                detail="Invalid start date."
            )

    if end_date:
        end = pd.to_datetime(end_date, errors="coerce")

        if pd.isna(end):
            raise HTTPException(
                status_code=400,
                detail="Invalid end date."
            )

    if start is not None and end is not None and start > end:
        raise HTTPException(
            status_code=400,
            detail="Start date cannot be after end date."
        )

    if start is not None:
        df = df[df["date"] >= start]

    if end is not None:
        df = df[df["date"] <= end]

    if df.empty:
        return {
            "fund": fund,
            "total_records": 0,
            "first_date": None,
            "last_date": None,
            "first_nav": None,
            "latest_nav": None,
            "change_pct": None,
            "records": []
        }

    first_nav = float(df.iloc[0]["nav"])
    latest_nav = float(df.iloc[-1]["nav"])

    change_pct = (
        ((latest_nav - first_nav) / first_nav) * 100
        if first_nav != 0 else None
    )

    records = [
        {
            "date": row["date"].strftime("%Y-%m-%d"),
            "nav": float(row["nav"])
        }
        for _, row in df.iterrows()
    ]

    return {
        "fund": fund,
        "total_records": len(records),
        "first_date": records[0]["date"],
        "last_date": records[-1]["date"],
        "first_nav": first_nav,
        "latest_nav": latest_nav,
        "change_pct": (
            round(change_pct, 2)
            if change_pct is not None else None
        ),
        "records": records
    }

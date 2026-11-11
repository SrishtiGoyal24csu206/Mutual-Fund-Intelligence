import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load all datasets
fund_master = pd.read_csv("data/01_fund_master.csv")
nav_history = pd.read_csv("data/02_nav_history.csv")
aum = pd.read_csv("data/03_aum_by_fund_house.csv")
sip = pd.read_csv("data/04_monthly_sip_inflows.csv")
category_inflows = pd.read_csv("data/05_category_inflows.csv")
folio = pd.read_csv("data/06_industry_folio_count.csv")
performance = pd.read_csv("data/07_scheme_performance.csv")
transactions = pd.read_csv("data/08_investor_transactions.csv")
holdings = pd.read_csv("data/09_portfolio_holdings.csv")
benchmark = pd.read_csv("data/10_benchmark_indices.csv")

axis = pd.read_csv("data/axis_bluechip_live_nav.csv")
hdfc = pd.read_csv("data/hdfc_top100_live_nav.csv")
icici = pd.read_csv("data/icici_bluechip_live_nav.csv")
kotak = pd.read_csv("data/kotak_bluechip_live_nav.csv")
nippon = pd.read_csv("data/nippon_largecap_live_nav.csv")
sbi = pd.read_csv("data/sbi_bluechip_live_nav.csv")

print("All 16 datasets loaded successfully!")
datasets = {
    "Fund Master": fund_master,
    "NAV History": nav_history,
    "AUM": aum,
    "SIP": sip,
    "Category Inflows": category_inflows,
    "Folio": folio,
    "Performance": performance,
    "Transactions": transactions,
    "Holdings": holdings,
    "Benchmark": benchmark,
    "Axis NAV": axis,
    "HDFC NAV": hdfc,
    "ICICI NAV": icici,
    "Kotak NAV": kotak,
    "Nippon NAV": nippon,
    "SBI NAV": sbi
}

print("\n--- DATASET SHAPES ---")
for name, df in datasets.items():
    print(name, ":", df.shape)
    # ==========================================
# EDA - BASIC DATA UNDERSTANDING
# ==========================================

datasets = {
    "Fund Master": fund_master,
    "NAV History": nav_history,
    "AUM by Fund House": aum,
    "Monthly SIP Inflows": sip,
    "Category Inflows": category_inflows,
    "Industry Folio Count": folio,
    "Scheme Performance": performance,
    "Investor Transactions": transactions,
    "Portfolio Holdings": holdings,
    "Benchmark Indices": benchmark,
    "Axis Bluechip NAV": axis,
    "HDFC Top 100 NAV": hdfc,
    "ICICI Bluechip NAV": icici,
    "Kotak Bluechip NAV": kotak,
    "Nippon Large Cap NAV": nippon,
    "SBI Bluechip NAV": sbi
}

print("\n========== DATASET OVERVIEW ==========\n")

for name, df in datasets.items():

    print("\n--------------------------------")
    print(name)
    print("--------------------------------")

    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("Missing Values:", df.isnull().sum().sum())
    print("Duplicate Rows:", df.duplicated().sum())
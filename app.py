
import joblib

import streamlit as st
import pandas as pd
from pathlib import Path

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Mutual Fund Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS - WHITE & GREEN THEME
# =====================================================
st.markdown("""
<style>
/* Main background */
.stApp {
    background-color: #f5faf6;
    color: #183b2b;
}

/* Main content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #e1f0e5;
    border-right: 1px solid #c8e3d0;
}

[data-testid="stSidebar"] * {
    color: #183b2b !important;
}

/* Headings */
h1, h2, h3 {
    color: #176b3a !important;
}

/* Main body text */
p, label {
    color: #244b35;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #c8e3d0;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 4px 12px rgba(20, 90, 45, 0.08);
}

/* Metric labels */
div[data-testid="stMetric"] label {
    color: #50705b !important;
}

/* Metric values */
div[data-testid="stMetricValue"] {
    color: #176b3a !important;
    font-weight: 700;
}

/* Metric delta */
div[data-testid="stMetricDelta"] {
    color: #218c4a !important;
}

/* Buttons */
div.stButton > button,
div.stDownloadButton > button {
    background-color: #218c4a;
    color: white !important;
    border: none;
    border-radius: 9px;
    padding: 0.5rem 1rem;
    font-weight: 600;
}

div.stButton > button:hover,
div.stDownloadButton > button:hover {
    background-color: #176b3a;
    color: white !important;
}

/* Input fields */
.stTextInput input,
.stNumberInput input {
    background-color: #ffffff;
    color: #183b2b;
    border: 1px solid #b8d8c2;
}

/* Selectbox */
div[data-baseweb="select"] {
    background-color: #ffffff;
}

/* Alerts */
div[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid #c8e3d0;
    border-radius: 10px;
}

/* Divider */
hr {
    border-color: #c8e3d0;
}

/* Sidebar radio */
[data-testid="stSidebar"] [role="radiogroup"] label {
    color: #183b2b !important;
    font-weight: 500;
}

/* Make dashboard text clearer and bolder */

.stApp {
    font-weight: 500;
}

/* Main headings */
h1, h2, h3 {
    color: #176B3A !important;
    font-weight: 750 !important;
}

/* Table text */
[data-testid="stDataFrame"] {
    font-size: 16px !important;
    font-weight: 600 !important;
}

/* Table headers */
[data-testid="stDataFrame"] th {
    font-weight: 700 !important;
    color: #174C32 !important;
}

/* Table cells */
[data-testid="stDataFrame"] td {
    font-weight: 600 !important;
    color: #174C32 !important;
}

/* Metric values */
[data-testid="stMetricValue"] {
    font-weight: 750 !important;
    color: #176B3A !important;
}

/* General text */
p, label, .stMarkdown {
    color: #244C36;
    font-weight: 500;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# DATA LOADING
# =====================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"


@st.cache_data
def load_csv(filename):
    file_path = DATA_DIR / filename

    if file_path.exists():
        return pd.read_csv(file_path)

    return pd.DataFrame()


funds = load_csv("07_scheme_performance_clean.csv")
nav = load_csv("02_nav_history_clean.csv")
aum = load_csv("03_aum_by_fund_house.csv")
category_flows = load_csv("05_category_inflows.csv")


# =====================================================
# LOAD TRAINED ML MODEL
# =====================================================

MODEL_DIR = BASE_DIR / "models"

@st.cache_resource
def load_ml_model():
    model_path = MODEL_DIR / "risk_classifier.pkl"
    preprocessor_path = MODEL_DIR / "risk_preprocessor.pkl"

    if model_path.exists() and preprocessor_path.exists():
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)

        return model, preprocessor

    return None, None


risk_model, risk_preprocessor = load_ml_model()

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================
st.sidebar.markdown("# 🌿 MF Intelligence")
st.sidebar.markdown("### Mutual Fund Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Overview",
        "🔎 Fund Explorer",
        "📈 Return Analysis",
        "⚖️ Risk Analysis",
        "🤖 ML Prediction",
        "ℹ️ About Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("---")

st.sidebar.markdown("**Developed by**")

st.sidebar.markdown("### Srishti Goyal")
st.sidebar.markdown("### Sonika Gupta")

st.sidebar.caption("BTech CSE | AIML Project")
st.sidebar.caption("Mutual Fund Intelligence")
# =====================================================
# MUTUAL FUND VISUALS
# =====================================================

st.markdown("""
<style>
.hero-banner {
    background: linear-gradient(120deg, #e0f5e8, #ffffff);
    border: 1px solid #b9e3c7;
    border-radius: 18px;
    padding: 28px;
    margin-bottom: 25px;
}

.hero-banner h2 {
    color: #176b3a !important;
    font-size: 30px;
}

.hero-banner p {
    color: #315b42;
    font-size: 16px;
}

.image-card {
    background: #ffffff;
    border: 1px solid #c8e3d0;
    border-radius: 14px;
    padding: 15px;
    box-shadow: 0 4px 12px rgba(20, 90, 45, 0.07);
}

.image-card h4 {
    color: #176b3a;
}

.image-card p {
    color: #50705b;
}

/* Attractive chart theme */
.stApp {
    background-color: #F1F8F3;
}

[data-testid="stVegaLiteChart"],
[data-testid="stArrowVegaLiteChart"],
[data-testid="stLineChart"],
[data-testid="stBarChart"] {
    background-color: #FFFFFF;
    border: 1px solid #D4E8D9;
    border-radius: 14px;
    padding: 12px;
}

/* Chart headings */
h2, h3 {
    color: #176B3A !important;
}

/* Chart containers */
div[data-testid="stChart"] {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)




# =====================================================
# THREE ATTRACTIVE IMAGE CARDS
# =====================================================

st.markdown("## 🌿 Explore Mutual Fund Intelligence")

img1, img2, img3 = st.columns(3)

with img1:
    st.image(
        "https://images.unsplash.com/photo-1579621970563-ebec7560ff3e?auto=format&fit=crop&w=700&q=80",
        use_container_width=True
    )

    st.markdown("""
    <div class="image-card">
        <h4>💰 Smart Investing</h4>
        <p>Build long-term wealth with mutual funds.</p>
    </div>
    """, unsafe_allow_html=True)

with img2:
    st.image(
        "https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?auto=format&fit=crop&w=700&q=80",
        use_container_width=True
    )

    st.markdown("""
    <div class="image-card">
        <h4>📈 Market Performance</h4>
        <p>Analyze fund returns and market trends.</p>
    </div>
    """, unsafe_allow_html=True)

with img3:
    st.image(
        "https://images.unsplash.com/photo-1677442136019-21780ecad995?auto=format&fit=crop&w=700&q=80",
        use_container_width=True
    )

    st.markdown("""
    <div class="image-card">
        <h4>🤖 ML Insights</h4>
        <p>Explore AI-powered mutual fund risk analysis.</p>
    </div>
    """, unsafe_allow_html=True)


# MUTUAL FUND IMAGE CARDS

# =====================================================
# PAGE-WISE HERO IMAGE
# =====================================================

page_images = {
    "🏠 Overview": (
        "https://images.unsplash.com/photo-1579621970563-ebec7560ff3e"
        "?auto=format&fit=crop&w=1400&q=85"
    ),

    "🔎 Fund Explorer": (
        "https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3"
        "?auto=format&fit=crop&w=1400&q=85"
    ),

    "📈 Return Analysis": (
        "https://images.unsplash.com/photo-1460925895917-afdad011f9b1"
        "?auto=format&fit=crop&w=1400&q=85"
    ),

    "⚖️ Risk Analysis": (
        "https://images.unsplash.com/photo-163 previously"
    ),

    "🤖 ML Prediction": (
        "https://images.unsplash.com/photo-1677442136019-21780ecad995"
        "?auto=format&fit=crop&w=1400&q=85"
    ),

    "ℹ️ About Project": (
        "https://images.unsplash.com/photo-1451187580459-43490279c0fa"
        "?auto=format&fit=crop&w=1400&q=85"
    )
}

# Page title and subtitle
page_titles = {
    "🏠 Overview": "Smart Mutual Fund Analytics",
    "🔎 Fund Explorer": "Explore Mutual Funds",
    "📈 Return Analysis": "Track Fund Performance",
    "⚖️ Risk Analysis": "Understand Investment Risk",
    "🤖 ML Prediction": "AI-Powered Risk Prediction",
    "ℹ️ About Project": "About Mutual Fund Intelligence"
}

page_subtitles = {
    "🏠 Overview": "Explore investment growth and mutual fund insights.",
    "🔎 Fund Explorer": "Discover and compare mutual fund schemes.",
    "📈 Return Analysis": "Analyze returns and performance trends.",
    "⚖️ Risk Analysis": "Understand risk metrics and market volatility.",
    "🤖 ML Prediction": "Explore machine learning based risk classification.",
    "ℹ️ About Project": "A BTech CSE AIML mutual fund analytics project."
}

# Display image and heading for selected page
st.image(
    page_images[page],
    width=650
)
st.markdown(
    f"""
    <div class="hero-banner">
        <h2>{page_titles[page]}</h2>
        <p>{page_subtitles[page]}</p>
    </div>
    """,
    unsafe_allow_html=True
)


# =====================================================
# OVERVIEW PAGE
# =====================================================
if page == "🏠 Overview":

    st.subheader("📊 Dashboard Overview")

    if funds.empty:

        st.error(
            "Fund performance dataset load nahi hua. "
            "Check data/07_scheme_performance_clean.csv"
        )

    else:

        st.success("Fund performance dataset loaded successfully.")

        total_schemes = len(funds)

        fund_houses = (
            funds["fund_house"].nunique()
            if "fund_house" in funds.columns else 0
        )

        categories = (
            funds["category"].nunique()
            if "category" in funds.columns else 0
        )

        return_cols = [
            col for col in funds.columns
            if "return" in col.lower()
            and pd.api.types.is_numeric_dtype(funds[col])
        ]

        average_return = (
            funds[return_cols[0]].mean()
            if return_cols else None
        )

        # KPI CARDS
        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Schemes",
            f"{total_schemes:,}"
        )

        col2.metric(
            "Fund Houses",
            f"{fund_houses:,}"
        )

        col3.metric(
            "Categories",
            f"{categories:,}"
        )

        col4.metric(
            "Average Return",
            f"{average_return:.2f}%"
            if average_return is not None else "N/A"
        )

        st.markdown("---")

        # RETURN CHART
        if return_cols:

            st.subheader("📈 Return Distribution")

            selected_return = st.selectbox(
                "Select Return Metric",
                return_cols
            )

            st.bar_chart(
                funds[selected_return].dropna()
            )

        # CATEGORY DISTRIBUTION
        if "category" in funds.columns:

            st.subheader("📂 Category Distribution")

            category_count = funds["category"].value_counts()

            st.bar_chart(category_count)

        # DATA PREVIEW
        st.subheader("📋 Fund Dataset Preview")

        st.dataframe(
            funds.head(10),
            use_container_width=True
        )


# =====================================================
# FUND EXPLORER PAGE
# =====================================================
elif page == "🔎 Fund Explorer":

    st.subheader("🔎 Explore Mutual Funds")

    if funds.empty:

        st.warning("Fund dataset available nahi hai.")

    else:

        filtered = funds.copy()

        # CATEGORY FILTER
        if "category" in filtered.columns:

            categories = sorted(
                filtered["category"]
                .dropna()
                .astype(str)
                .unique()
            )

            selected_category = st.selectbox(
                "Select Fund Category",
                ["All Categories"] + categories
            )

            if selected_category != "All Categories":

                filtered = filtered[
                    filtered["category"].astype(str)
                    == selected_category
                ]

        # FUND HOUSE FILTER
        if "fund_house" in filtered.columns:

            houses = sorted(
                filtered["fund_house"]
                .dropna()
                .astype(str)
                .unique()
            )

            selected_house = st.selectbox(
                "Select Fund House",
                ["All Fund Houses"] + houses
            )

            if selected_house != "All Fund Houses":

                filtered = filtered[
                    filtered["fund_house"].astype(str)
                    == selected_house
                ]

        # SCHEME SEARCH
        search = st.text_input(
            "🔍 Search Scheme Name"
        )

        if "scheme_name" in filtered.columns and search:

            filtered = filtered[
                filtered["scheme_name"].str.contains(
                    search,
                    case=False,
                    na=False
                )
            ]

        st.markdown("---")

        st.write(
            f"### Matching Schemes: {len(filtered)}"
        )

        st.dataframe(
            filtered,
            use_container_width=True
        )

        # DOWNLOAD FILTERED DATA
        csv_data = filtered.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Filtered Funds",
            data=csv_data,
            file_name="filtered_mutual_funds.csv",
            mime="text/csv"
        )


# =====================================================
# RETURN ANALYSIS PAGE
# =====================================================
elif page == "📈 Return Analysis":

    st.subheader("📈 Mutual Fund Return Analysis")

    if funds.empty:

        st.warning("Fund dataset available nahi hai.")

    else:

        return_cols = [
            col for col in funds.columns
            if "return" in col.lower()
            and pd.api.types.is_numeric_dtype(funds[col])
        ]

        if return_cols:

            selected_return = st.selectbox(
                "Choose Return Metric",
                return_cols
            )

            st.markdown("### Return Distribution")

            st.bar_chart(
                funds[selected_return].dropna()
            )

            st.markdown("### Return Statistics")

            st.dataframe(
                funds[selected_return].describe().to_frame(),
                use_container_width=True
            )

            if "scheme_name" in funds.columns:

                top_funds = funds[
                    ["scheme_name", selected_return]
                ].dropna().sort_values(
                    selected_return,
                    ascending=False
                ).head(10)

                st.markdown(
                    "### 🏆 Top 10 Funds by Selected Return"
                )

                st.dataframe(
                    top_funds,
                    use_container_width=True
                )

        else:

            st.info(
                "Numeric return columns dataset mein nahi mile."
            )


# =====================================================
# RISK ANALYSIS PAGE
# =====================================================
elif page == "⚖️ Risk Analysis":

    st.subheader("⚖️ Mutual Fund Risk Analysis")

    if funds.empty:

        st.warning("Fund dataset available nahi hai.")

    else:

        risk_cols = [
            col for col in funds.columns
            if any(term in col.lower() for term in [
                "risk",
                "sharpe",
                "sortino",
                "drawdown",
                "beta",
                "std_dev"
            ])
        ]

        if risk_cols:

            selected_risk = st.selectbox(
                "Choose Risk Metric",
                risk_cols
            )

            if pd.api.types.is_numeric_dtype(
                funds[selected_risk]
            ):

                st.markdown(
                    f"### {selected_risk} Distribution"
                )

                st.bar_chart(
                    funds[selected_risk].dropna()
                )

                st.markdown("### Risk Statistics")

                st.dataframe(
                    funds[selected_risk].describe().to_frame(),
                    use_container_width=True
                )

            else:

                st.markdown(
                    "### Risk Category Distribution"
                )

                st.bar_chart(
                    funds[selected_risk].value_counts()
                )

        else:

            st.info(
                "Risk-related columns dataset mein nahi mile."
            )


# =====================================================
# ML PREDICTION PAGE
# =====================================================


    # =====================================================
# ML PREDICTION PAGE
# =====================================================
elif page == "🤖 ML Prediction":

    st.subheader("🤖 Machine Learning Prediction")

    st.markdown("""
    ### Mutual Fund Risk Classification

    Predict the risk category of a mutual fund using a trained ML model
    """)

    if risk_model is None or risk_preprocessor is None:

        st.error(
            "Trained model load nahi hua. "
            "Check karein ki models folder mein "
            "risk_classifier.pkl aur risk_preprocessor.pkl available hain."
        )

    else:

        # Dataset se dropdown options lena
        def get_options(column_name):
            if column_name in funds.columns:
                values = (
                    funds[column_name]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                if values:
                    return sorted(values)

            return ["Other"]

        with st.form("risk_prediction_form"):

            st.markdown("### Enter Fund Details")

            category = st.selectbox(
                "Fund Category",
                get_options("category")
            )

            plan = st.selectbox(
                "Plan",
                get_options("plan")
            )

            fund_house = st.selectbox(
                "Fund House",
                get_options("fund_house")
            )

            beta = st.number_input(
                "Beta",
                value=1.0,
                step=0.1
            )

            sharpe_ratio = st.number_input(
                "Sharpe Ratio",
                value=0.5,
                step=0.1
            )

            sortino_ratio = st.number_input(
                "Sortino Ratio",
                value=0.7,
                step=0.1
            )

            std_dev_ann_pct = st.number_input(
                "Annual Standard Deviation (%)",
                value=15.0,
                step=1.0
            )

            max_drawdown_pct = st.number_input(
                "Maximum Drawdown (%)",
                value=-20.0,
                step=1.0
            )

            aum_crore = st.number_input(
                "AUM (Crore)",
                value=1000.0,
                min_value=0.0,
                step=100.0
            )

            expense_ratio_pct = st.number_input(
                "Expense Ratio (%)",
                value=1.0,
                min_value=0.0,
                step=0.1
            )

            morningstar_rating = st.selectbox(
                "Morningstar Rating",
                [1, 2, 3, 4, 5],
                index=3
            )

            predict_button = st.form_submit_button(
                "🔍 Predict Risk Class"
            )

        if predict_button:

            # Model training ke feature names ke saath input
            input_data = pd.DataFrame([{
                "beta": beta,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "std_dev_ann_pct": std_dev_ann_pct,
                "max_drawdown_pct": max_drawdown_pct,
                "aum_crore": aum_crore,
                "expense_ratio_pct": expense_ratio_pct,
                "morningstar_rating": morningstar_rating,
                "category": category,
                "plan": plan,
                "fund_house": fund_house
            }])

            
            try:

                # Saved preprocessing pipeline use karna
                processed_data = risk_preprocessor.transform(
                    input_data
                )

                # Risk class prediction
                prediction = risk_model.predict(
                    processed_data
                )[0]

                # ==========================================
                # PROFESSIONAL PREDICTION RESULT
                # ==========================================

                st.markdown("---")
                st.subheader("📊 Machine Learning Prediction Result")

                predicted_class = int(prediction)

                # Class labels: verify from training target
                risk_class_labels = {
                    0: "Risk Class 0",
                    1: "Risk Class 1"
                }

                predicted_label = risk_class_labels.get(
                    predicted_class,
                    f"Risk Class {predicted_class}"
                )

                st.success(
                    f"### Predicted Category: {predicted_label}"
                )

                st.metric(
                    label="Predicted Risk Class",
                    value=predicted_class
                )

                # Risk class explanation
                st.markdown("### 📘 Understanding Risk Classes")

                st.markdown("""
                - **Risk Class 0:** One category learned by the model.
                - **Risk Class 1:** Another category learned by the model.

                The exact meaning (Low Risk or High Risk) must be
                confirmed from the original training labels.
                """)

                # Probability chart
                if hasattr(risk_model, "predict_proba"):

                    probabilities = risk_model.predict_proba(
                        processed_data
                    )[0]

                    probability_data = pd.DataFrame({
                        "Risk Class": [
                            f"Class {int(class_label)}"
                            for class_label in risk_model.classes_
                        ],
                        "Probability (%)": probabilities * 100
                    })

                    st.markdown("### 📈 Prediction Probability Graph")

                    st.bar_chart(
                        probability_data.set_index("Risk Class"),
                        color="#23834A"
                    )

                    st.caption(
                        "The graph shows the model's estimated probability "
                        "for each class. It does not guarantee future "
                        "investment performance."
                    )

                # Explanation of input features
                st.markdown("### 💡 What Does This Result Mean?")

                st.markdown("""
                The model predicts a risk category based on the
                mutual fund parameters entered.

                - **Beta:** Sensitivity to market movements.
                - **Standard Deviation:** Variation in returns.
                - **Maximum Drawdown:** Largest historical decline.
                - **Sharpe Ratio:** Return relative to total risk.
                - **Sortino Ratio:** Return relative to downside risk.
                - **Expense Ratio:** Annual fund expense.
                - **Morningstar Rating:** Rating-based input feature.
                """)

                # Input summary
                st.markdown("### 🧾 Prediction Input Summary")

                st.dataframe(
                    input_data,
                    use_container_width=True,
                    hide_index=True
                )

                # Disclaimer
                st.warning(
                    "This is an educational Machine Learning project. "
                    "The predicted risk class is based on historical data "
                    "and does not guarantee future returns or investment "
                    "outcomes. It is not personal financial advice."
                )

            except Exception as e:

                st.error(
                    f"Prediction generate nahi hui: {e}"
                )

# =====================================================
# ABOUT PROJECT PAGE
# =====================================================
elif page == "ℹ️ About Project":

    st.subheader("ℹ️ About This Project")

    st.markdown("""
    # Mutual Fund Intelligence

    An interactive mutual fund analytics dashboard
    developed as a BTech CSE AIML project.

    ### Project Objectives

    - Understand mutual fund performance
    - Analyze returns and risk
    - Explore mutual fund categories
    - Apply machine learning techniques
    - Build an interactive analytics dashboard

    ### Technology Stack

    - Python
    - Pandas
    - Scikit-learn
    - Streamlit
    - Jupyter Notebook

    ### Machine Learning Tasks

    - Risk Classification
    - Return Regression
    - Model Evaluation
    - Feature Analysis
    """)

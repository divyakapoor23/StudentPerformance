"""
OULAD Student Performance Dashboard
Run with:  streamlit run dashboard.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OULAD Student Performance",
    page_icon="🎓",
    layout="wide",
)

# ── Data loading (cached) ────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "oulad"


@st.cache_data
def load_data() -> dict[str, pd.DataFrame]:
    student_info = pd.read_csv(DATA_DIR / "studentInfo.csv")
    student_info["imd_band"] = student_info["imd_band"].replace("?", "Unknown")

    assessments = pd.read_csv(DATA_DIR / "assessments.csv")
    assessments["date"] = pd.to_numeric(assessments["date"], errors="coerce")

    student_assessments = pd.read_csv(DATA_DIR / "studentAssessment.csv")
    student_assessments["score"] = pd.to_numeric(
        student_assessments["score"], errors="coerce"
    )

    student_vle = pd.read_csv(DATA_DIR / "studentVle.csv")

    # ── Engagement features ──────────────────────────────────────────────────
    engagement_features = (
        student_vle.groupby(["id_student", "code_module", "code_presentation"])
        .agg(total_clicks=("sum_click", "sum"), active_days=("date", "nunique"))
        .reset_index()
    )

    early_engagement = (
        student_vle.assign(
            date_num=pd.to_numeric(student_vle["date"], errors="coerce")
        )
        .groupby(["id_student", "code_module", "code_presentation"], as_index=False)
        .agg(engaged_before_start=("date_num", lambda x: bool((x < 0).any())))
    )

    # ── Average assessment score per student-course ──────────────────────────
    merge_sa = pd.merge(student_assessments, assessments, on="id_assessment", how="left")
    merge_full = pd.merge(
        merge_sa,
        student_info,
        on=["id_student", "code_module", "code_presentation"],
        how="inner",
    )
    avg_score = (
        merge_full.groupby(["id_student", "code_module", "code_presentation"])["score"]
        .mean()
        .reset_index()
        .rename(columns={"score": "avg_score"})
    )

    # ── Timeliness ───────────────────────────────────────────────────────────
    merged_assess = pd.merge(
        student_assessments,
        assessments[["id_assessment", "code_module", "code_presentation", "date"]],
        on="id_assessment",
        how="left",
    )
    merged_assess["days_late"] = merged_assess["date_submitted"] - merged_assess["date"]
    timeliness = (
        merged_assess.groupby(["id_student", "code_module", "code_presentation"])
        .agg(
            avg_days_late=("days_late", "mean"),
            num_late_submissions=("days_late", lambda x: (x > 0).sum()),
        )
        .reset_index()
    )

    # ── Completion rate ──────────────────────────────────────────────────────
    assigned_counts = (
        assessments.groupby(["code_module", "code_presentation"])
        .size()
        .reset_index(name="num_assigned")
    )
    submitted_counts = (
        merged_assess.groupby(["id_student", "code_module", "code_presentation"])
        .size()
        .reset_index(name="num_submitted")
    )
    completion = pd.merge(
        submitted_counts, assigned_counts, on=["code_module", "code_presentation"], how="left"
    )
    completion["completion_rate"] = completion["num_submitted"] / completion["num_assigned"]

    # ── Master table ─────────────────────────────────────────────────────────
    master = student_info.merge(
        engagement_features, on=["id_student", "code_module", "code_presentation"], how="left"
    )
    master = master.merge(
        avg_score, on=["id_student", "code_module", "code_presentation"], how="left"
    )
    master = master.merge(
        early_engagement[["id_student", "code_module", "code_presentation", "engaged_before_start"]],
        on=["id_student", "code_module", "code_presentation"],
        how="left",
    )
    master = master.merge(
        timeliness, on=["id_student", "code_module", "code_presentation"], how="left"
    )
    master = master.merge(
        completion[["id_student", "code_module", "code_presentation", "completion_rate"]],
        on=["id_student", "code_module", "code_presentation"],
        how="left",
    )
    master["total_clicks"] = master["total_clicks"].fillna(0)
    master["active_days"] = master["active_days"].fillna(0)
    master["engaged_before_start"] = master["engaged_before_start"].fillna(False)
    master["passed"] = master["final_result"].isin(["Pass", "Distinction"])

    return {
        "student_info": student_info,
        "assessments": assessments,
        "student_assessments": student_assessments,
        "student_vle": student_vle,
        "merge_full": merge_full,
        "master": master,
        "engagement_features": engagement_features,
    }


data = load_data()
master = data["master"]
student_info = data["student_info"]
merge_full = data["merge_full"]

# ── Sidebar filters ──────────────────────────────────────────────────────────
st.sidebar.title("🎓 OULAD Dashboard")
st.sidebar.markdown("---")

modules = ["All"] + sorted(master["code_module"].unique().tolist())
selected_module = st.sidebar.selectbox("Module", modules)

presentations = ["All"] + sorted(master["code_presentation"].unique().tolist())
selected_presentation = st.sidebar.selectbox("Presentation", presentations)

genders = ["All"] + sorted(master["gender"].dropna().unique().tolist())
selected_gender = st.sidebar.selectbox("Gender", genders)

# Apply filters
filtered = master.copy()
if selected_module != "All":
    filtered = filtered[filtered["code_module"] == selected_module]
if selected_presentation != "All":
    filtered = filtered[filtered["code_presentation"] == selected_presentation]
if selected_gender != "All":
    filtered = filtered[filtered["gender"] == selected_gender]

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing **{len(filtered):,}** of **{len(master):,}** records")

# ── Navigation tabs ──────────────────────────────────────────────────────────
tab_overview, tab_demographics, tab_engagement, tab_assessments, tab_model = st.tabs(
    ["📊 Overview", "👥 Demographics", "📈 Engagement", "📝 Assessments", "🤖 Model"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.header("Overview")

    total_students = filtered["id_student"].nunique()
    pass_rate = filtered["passed"].mean() * 100
    withdrawal_rate = (filtered["final_result"] == "Withdrawn").mean() * 100
    avg_clicks = filtered["total_clicks"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students", f"{total_students:,}")
    c2.metric("Pass Rate", f"{pass_rate:.1f}%")
    c3.metric("Withdrawal Rate", f"{withdrawal_rate:.1f}%")
    c4.metric("Avg VLE Clicks", f"{avg_clicks:,.0f}")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        result_counts = filtered["final_result"].value_counts().reset_index()
        result_counts.columns = ["Result", "Count"]
        fig = px.bar(
            result_counts,
            x="Result",
            y="Count",
            color="Result",
            title="Final Result Distribution",
            text_auto=True,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        fig = px.pie(
            result_counts,
            names="Result",
            values="Count",
            title="Result Share",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.35,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Average Engagement Metrics by Outcome")
    engagement_by_result = (
        filtered.groupby("final_result")[["total_clicks", "active_days", "avg_score"]]
        .mean()
        .round(1)
        .sort_values("total_clicks", ascending=False)
        .reset_index()
    )
    engagement_by_result.columns = ["Final Result", "Avg Total Clicks", "Avg Active Days", "Avg Score"]
    st.dataframe(engagement_by_result, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DEMOGRAPHICS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_demographics:
    st.header("Demographics")

    col1, col2 = st.columns(2)

    with col1:
        gender_counts = (
            filtered.groupby(["gender", "final_result"])
            .size()
            .reset_index(name="count")
        )
        fig = px.bar(
            gender_counts,
            x="gender",
            y="count",
            color="final_result",
            barmode="group",
            title="Final Result by Gender",
            labels={"gender": "Gender", "count": "Count", "final_result": "Result"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        age_counts = (
            filtered.groupby(["age_band", "final_result"])
            .size()
            .reset_index(name="count")
        )
        fig = px.bar(
            age_counts,
            x="age_band",
            y="count",
            color="final_result",
            barmode="stack",
            title="Final Result by Age Band",
            labels={"age_band": "Age Band", "count": "Count", "final_result": "Result"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        pass_by_edu = (
            filtered.groupby("highest_education")["passed"]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
            .reset_index()
        )
        pass_by_edu.columns = ["Education Level", "Pass Rate (%)"]
        fig = px.bar(
            pass_by_edu,
            x="Education Level",
            y="Pass Rate (%)",
            title="Pass Rate by Highest Prior Education",
            text_auto=".1f",
            color="Pass Rate (%)",
            color_continuous_scale="Blues",
        )
        fig.update_layout(xaxis_tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        pass_by_imd = (
            filtered.groupby("imd_band")["passed"]
            .mean()
            .mul(100)
            .reset_index()
        )
        pass_by_imd.columns = ["IMD Band", "Pass Rate (%)"]
        imd_order = [
            "0-10%", "10-20", "20-30%", "30-40%", "40-50%",
            "50-60%", "60-70%", "70-80%", "80-90%", "90-100%", "Unknown",
        ]
        pass_by_imd["IMD Band"] = pd.Categorical(
            pass_by_imd["IMD Band"], categories=imd_order, ordered=True
        )
        pass_by_imd = pass_by_imd.sort_values("IMD Band")
        fig = px.line(
            pass_by_imd.dropna(subset=["IMD Band"]),
            x="IMD Band",
            y="Pass Rate (%)",
            markers=True,
            title="Pass Rate by IMD Band (Deprivation)",
        )
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Pass Rate by Region")
    pass_by_region = (
        filtered.groupby("region")["passed"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .reset_index()
    )
    pass_by_region.columns = ["Region", "Pass Rate (%)"]
    fig = px.bar(
        pass_by_region,
        x="Region",
        y="Pass Rate (%)",
        title="Pass Rate by Region",
        text_auto=".1f",
        color="Pass Rate (%)",
        color_continuous_scale="Teal",
    )
    fig.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ENGAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_engagement:
    st.header("VLE Engagement Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.box(
            filtered[filtered["total_clicks"] > 0],
            x="final_result",
            y="total_clicks",
            color="final_result",
            log_y=True,
            title="Total VLE Clicks by Final Result (log scale)",
            labels={"final_result": "Result", "total_clicks": "Total Clicks (log)"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            filtered[filtered["active_days"] > 0],
            x="final_result",
            y="active_days",
            color="final_result",
            title="Active Days by Final Result",
            labels={"final_result": "Result", "active_days": "Active Days"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Early Engagement vs Outcome")
    early_pct = (
        filtered.groupby(["engaged_before_start", "final_result"])
        .size()
        .reset_index(name="count")
    )
    early_pct["label"] = early_pct["engaged_before_start"].map(
        {True: "Engaged Before Start", False: "Not Engaged Before Start"}
    )
    total_per_group = early_pct.groupby("label")["count"].transform("sum")
    early_pct["pct"] = (early_pct["count"] / total_per_group * 100).round(1)

    fig = px.bar(
        early_pct,
        x="label",
        y="pct",
        color="final_result",
        barmode="stack",
        text_auto=".1f",
        title="Outcome Distribution: Early vs. Non-Early Engagers (%)",
        labels={"label": "Engagement Group", "pct": "Percentage (%)", "final_result": "Result"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Clicks vs. Average Assessment Score")
    scatter_df = filtered[(filtered["total_clicks"] > 0) & filtered["avg_score"].notna()]
    fig = px.scatter(
        scatter_df,
        x="total_clicks",
        y="avg_score",
        color="final_result",
        log_x=True,
        opacity=0.4,
        title="Total Clicks (log) vs. Average Assessment Score",
        labels={
            "total_clicks": "Total Clicks (log scale)",
            "avg_score": "Average Assessment Score",
            "final_result": "Result",
        },
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ASSESSMENTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_assessments:
    st.header("Assessment Analysis")

    col1, col2 = st.columns(2)

    with col1:
        merge_filtered = merge_full[
            merge_full["id_student"].isin(filtered["id_student"])
        ]
        fig = px.box(
            merge_filtered.dropna(subset=["score"]),
            x="final_result",
            y="score",
            color="final_result",
            title="Assessment Score by Final Result",
            labels={"final_result": "Result", "score": "Score"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            merge_filtered.dropna(subset=["score"]),
            x="score",
            color="final_result",
            nbins=40,
            barmode="overlay",
            opacity=0.6,
            title="Score Distribution by Outcome",
            labels={"score": "Assessment Score", "final_result": "Result"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        timeliness_df = filtered.dropna(subset=["avg_days_late"])
        fig = px.box(
            timeliness_df,
            x="final_result",
            y="avg_days_late",
            color="final_result",
            title="Avg Days Late (submission vs. deadline) by Outcome",
            labels={"final_result": "Result", "avg_days_late": "Avg Days Late"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        completion_df = filtered.dropna(subset=["completion_rate"])
        fig = px.box(
            completion_df,
            x="final_result",
            y="completion_rate",
            color="final_result",
            title="Assessment Completion Rate by Outcome",
            labels={"final_result": "Result", "completion_rate": "Completion Rate"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Average Key Metrics by Outcome")
    summary_table = (
        filtered.groupby("final_result")[[
            "total_clicks", "active_days", "avg_score",
            "avg_days_late", "num_late_submissions", "completion_rate"
        ]]
        .mean()
        .round(2)
        .sort_values("total_clicks", ascending=False)
        .reset_index()
    )
    summary_table.columns = [
        "Final Result", "Avg Clicks", "Avg Active Days", "Avg Score",
        "Avg Days Late", "Avg Late Submissions", "Avg Completion Rate",
    ]
    st.dataframe(summary_table, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — MODEL
# ═══════════════════════════════════════════════════════════════════════════════
with tab_model:
    st.header("Logistic Regression — Pass Prediction")
    st.markdown(
        "Trains a logistic regression on the **filtered** data using demographic + "
        "engagement features to predict whether a student passes (Pass or Distinction)."
    )

    feature_cols = [
        "gender", "region", "highest_education", "imd_band", "age_band",
        "num_of_prev_attempts", "studied_credits", "disability", "engaged_before_start",
    ]

    model_df = filtered[feature_cols + ["passed"]].dropna(subset=feature_cols).copy()
    model_df["engaged_before_start"] = model_df["engaged_before_start"].astype(int)
    model_df = pd.get_dummies(
        model_df,
        columns=["gender", "region", "highest_education", "imd_band", "age_band", "disability"],
        drop_first=True,
    )

    if len(model_df) < 50:
        st.warning("Not enough data to train the model with current filters. Please broaden your selection.")
    else:
        X = model_df.drop(columns=["passed"])
        y = model_df["passed"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        numeric_cols = ["num_of_prev_attempts", "studied_credits"]
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        m1, m2, m3 = st.columns(3)
        report = classification_report(y_test, y_pred, target_names=["Not Passed", "Passed"], output_dict=True)
        m1.metric("Accuracy", f"{acc:.1%}")
        m2.metric("Precision (Passed)", f"{report['Passed']['precision']:.1%}")
        m3.metric("Recall (Passed)", f"{report['Passed']['recall']:.1%}")

        st.subheader("Top Feature Importances (Coefficient Magnitude)")
        coef_df = (
            pd.DataFrame({"Feature": X.columns, "Coefficient": clf.coef_[0]})
            .assign(AbsCoef=lambda d: d["Coefficient"].abs())
            .sort_values("AbsCoef", ascending=False)
            .head(20)
        )
        coef_df["Direction"] = coef_df["Coefficient"].apply(
            lambda v: "Positive (↑ pass)" if v > 0 else "Negative (↓ pass)"
        )
        fig = px.bar(
            coef_df.sort_values("Coefficient"),
            x="Coefficient",
            y="Feature",
            color="Direction",
            orientation="h",
            title="Logistic Regression Coefficients (top 20 by magnitude)",
            color_discrete_map={
                "Positive (↑ pass)": "#2ca02c",
                "Negative (↓ pass)": "#d62728",
            },
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

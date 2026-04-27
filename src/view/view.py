import pandas as pd
import streamlit as st
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

st.set_page_config(page_title="RAGAS Dashboard", layout="wide")
st.title("RAGAS CSV 시각화")

experiments_dir = PROJECT_ROOT / "ragas_store" / "experiments"
csv_files = sorted(experiments_dir.glob("*.csv"))

if not csv_files:
    st.error(f"CSV 파일을 찾을 수 없습니다: {experiments_dir}")
    st.stop()

selected_csv_name = st.selectbox(
    "분석할 CSV 파일",
    options=[file.name for file in csv_files],
)
csv_path = experiments_dir / selected_csv_name
st.caption(f"선택된 파일: `{selected_csv_name}`")

df = pd.read_csv(csv_path)

metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
for c in metrics:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["avg_score"] = df[metrics].mean(axis=1)

THRESHOLD = 0.8

COLUMN_DESCRIPTIONS = {
    "user_input": "사용자 질문",
    "response": "AI 답변",
    "reference": "테스트셋 기준 정답 답변",
    "retrieved_contexts": "AI 답변 생성에 사용한 TOP-K 컨텍스트",
    "reference_contexts": "테스트셋 생성에 사용한 TOP-K 컨텍스트",
    "faithfulness": "근거 기반 답변 일치도",
    "answer_relevancy": "질문-답변 관련성",
    "context_precision": "검색 컨텍스트 정밀도",
    "context_recall": "검색 컨텍스트 재현율",
}


def _mean_html(label: str, value: float) -> str:
    if pd.isna(value):
        color = "#6c757d"
        value_str = "—"
    else:
        color = "#198754" if value > THRESHOLD else "#dc3545"
        value_str = f"{value:.3f}"
    return (
        f'<p style="margin:0 0 0.25rem 0;font-size:0.9rem;">{label} '
        f'<span style="color:#6c757d;">(기준 {THRESHOLD})</span></p>'
        f'<p style="margin:0;font-size:1.75rem;font-weight:600;color:{color}">{value_str}</p>'
    )


# 상단 요약: 지표별 평균은 0.8을 넘으면 초록, 못 넘으면 빨강
st.caption(
    f"지표 **평균**은 **{THRESHOLD}점**을 기준으로, 초과는 초록·이하는 빨강으로 표시합니다."
)
c1, c2, c3, c4 = st.columns(4)
c1.markdown(_mean_html("Faithfulness 평균", float(df["faithfulness"].mean())), unsafe_allow_html=True)
c2.markdown(_mean_html("Relevancy 평균", float(df["answer_relevancy"].mean())), unsafe_allow_html=True)
c3.markdown(_mean_html("Precision 평균", float(df["context_precision"].mean())), unsafe_allow_html=True)
c4.markdown(_mean_html("Recall 평균", float(df["context_recall"].mean())), unsafe_allow_html=True)

# 하위 케이스
st.subheader("낮은 점수 질문 확인")
selected_metric = st.selectbox("확인할 지표", metrics)
base_columns = [
    "user_input",
    "response",
    "reference",
    "retrieved_contexts",
    "reference_contexts",
]
display_columns = [col for col in base_columns if col in df.columns] + [selected_metric]
low_score_df = df[df[selected_metric] <= THRESHOLD][display_columns]
display_low_score_df = low_score_df.rename(
    columns={
        col: f"{col} ({COLUMN_DESCRIPTIONS[col]})"
        for col in low_score_df.columns
        if col in COLUMN_DESCRIPTIONS
    }
)

if low_score_df.empty:
    st.info(f"`{selected_metric}` 지표에서 {THRESHOLD} 이하인 항목이 없습니다.")
else:
    st.dataframe(display_low_score_df, use_container_width=True)

# streamlit run src/view/view.py
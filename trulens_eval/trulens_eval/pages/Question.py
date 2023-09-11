import argparse

import asyncio
import json
import math
import sys

# https://github.com/jerryjliu/llama_index/issues/7244:
asyncio.set_event_loop(asyncio.new_event_loop())

from millify import millify
import numpy as np
import streamlit as st
from streamlit_extras.switch_page_button import switch_page

from trulens_eval.db_migration import MIGRATION_UNKNOWN_STR
from trulens_eval.ux.styles import CATEGORY

st.runtime.legacy_caching.clear_cache()

from trulens_eval import Tru
from trulens_eval.ux import styles
from trulens_eval.ux.components import draw_metadata
from trulens_eval.ux.add_logo import add_logo
from time import sleep
st.set_page_config(page_title="Question", layout="wide")

st.runtime.legacy_caching.clear_cache()
state = st.session_state

add_logo()

tru = Tru()
lms = tru.db

if 'question' not in state:
    st.write("No question selected yet. Please select one")
    sleep(3.0)
    switch_page('Leaderboard')



st.title("Evaluations of - " + state.question)

if 'all_records' not in state or 'feedback_cols' not in state:
    df_results, feedback_cols = lms.get_records_and_feedback([])
    state.all_records = df_results
    state.feedback_cols = feedback_cols

st.write('Average feedback values displayed in the range from 0 (worst) to 1 (best).')
    
# Filter records based on selected question
selected_question = state.question
df = state.all_records[state.all_records['question'] == selected_question]

feedback_col_names = state.feedback_cols

if df.empty:
    st.write("No records yet for the selected question.")
    switch_page('Leaderboard')

# Sort apps by their average 'is_correct' in descending order
sorted_apps = df.groupby('app_id')['is_correct'].mean().sort_values(ascending=False).index.tolist()

# Display leaderboard for the selected question
st.markdown("""---""")

for app_id in sorted_apps:
    app_df = df[df['app_id'] == app_id]
    
    # Display the question as header
    st.header(selected_question)

    col1, col2, col3, col4, *feedback_cols, col_records = st.columns(
        5 + len(feedback_col_names)
    )

    # Metrics computation and display
    latency_mean = app_df['latency'].apply(
        lambda td: td if td != MIGRATION_UNKNOWN_STR else None
    ).mean()

    col1.metric("Records", len(app_df))
    col2.metric(
        "Average Latency (Seconds)",
        f"{millify(round(latency_mean, 5), precision=2)}"
        if not math.isnan(latency_mean) else "nan"
    )
    col3.metric(
        "Total Cost (USD)",
        f"${millify(round(sum(cost for cost in app_df.total_cost if cost is not None), 5), precision = 2)}"
    )
    col4.metric(
        "Total Tokens",
        millify(
            sum(
                tokens for tokens in app_df.total_tokens
                if tokens is not None
            ),
            precision=2
        )
    )

    for i, col_name in enumerate(feedback_col_names):
        mean = app_df[col_name].mean()

        if math.isnan(mean):
            pass
        else:
            if col_name in ['false_n', 'false_p']:
                cat = CATEGORY.of_score(1 - mean)
                if (mean) < 0.2:
                    adjective = 'low'
                    delta_c = "normal"
                else:
                    adjective = 'high'
                    delta_c = "inverse"
            else:
                cat = CATEGORY.of_score(mean)
                adjective = cat.adjective
                delta_c = "normal" if mean >= CATEGORY.PASS.threshold else "inverse"
            feedback_cols[i].metric(
                label=col_name,
                value=f'{round(mean, 2)}',
                delta=f'{cat.icon} {adjective}',
                delta_color=delta_c
            )

    # Adding the button for each app to view its evaluations
    with col_records:
        if st.button('Select App', key=f"app-selector-{app_id}"):
            state.app_id = app_id
            switch_page('Model Details')

    st.markdown("""---""")


   
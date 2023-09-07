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

st.set_page_config(page_title="Leaderboard", layout="wide")

from trulens_eval.ux.add_logo import add_logo

add_logo()

parser = argparse.ArgumentParser()
parser.add_argument('--database-url', default=None)

try:
    args = parser.parse_args()
except SystemExit as e:
    # This exception will be raised if --help or invalid command line arguments
    # are used. Currently, streamlit prevents the program from exiting normally,
    # so we have to do a hard exit.
    sys.exit(e.code)

tru = Tru(database_url=args.database_url)
lms = tru.db


def streamlit_app():
    st.title('Question Leaderboard')
    st.write('Average feedback values displayed in the range from 0 (worst) to 1 (best).')

    state = st.session_state

    if 'all_records' not in state or 'feedback_cols' not in state:
        df_results, feedback_cols = lms.get_records_and_feedback([])
        state.all_records = df_results
        state.feedback_cols = feedback_cols

    df = state.all_records
    feedback_col_names = state.feedback_cols

    if df.empty:
        st.write("No records yet...")
        return

    # Selecting top apps based on 'is_correct' feedback
    grouped = df.groupby(['question', 'app_id'])['is_correct'].mean()
    top_apps = grouped.groupby('question').idxmax().values

    # Display leaderboard
    st.markdown("""---""")
    
    for question, app in top_apps:
        app_df = df[(df['question'] == question) & (df['app_id'] == app)]
        
        # Display the question as header instead of app
        st.header(question)

        col1, col2, col3, col4, *feedback_cols, col_records, col_all_apps = st.columns(
            6 + len(feedback_col_names)
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

            st.write(
                styles.stmetricdelta_hidearrow,
                unsafe_allow_html=True,
            )

            if math.isnan(mean):
                pass
            else:
                if col_name in ['false_n', 'false_p']:
                    cat = CATEGORY.of_score(1 - mean)
                    if (mean) < 0.2:  # Replace SOME_THRESHOLD with an actual value
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

        # Adding two buttons for each row
        with col_records:
            if st.button('Select App', key=f"app-selector-{app}"):
                st.session_state.question = question
                st.session_state.app = app
                switch_page('Evaluations')

        with col_all_apps:
            if st.button(f"View All Apps for {question}", key=f"all-apps-{question}"):
                st.session_state.question = question
                st.session_state.app = None
                switch_page("Question")

        st.markdown("""---""")
            

        #with st.expander("Model metadata"):
        #    st.markdown(draw_metadata(metadata))



# Define the main function to run the app
def main():
    streamlit_app()


if __name__ == '__main__':
    main()

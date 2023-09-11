import json
from typing import Iterable, List, Tuple
import asyncio
from time import sleep

# https://github.com/jerryjliu/llama_index/issues/7244:
asyncio.set_event_loop(asyncio.new_event_loop())

import matplotlib.pyplot as plt
import pandas as pd
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
from st_aggrid.shared import JsCode
import streamlit as st
from streamlit_javascript import st_javascript
from ux.add_logo import add_logo
from ux.styles import CATEGORY

from trulens_eval import Tru
from trulens_eval.app import ComponentView
from trulens_eval.app import instrumented_component_views
from trulens_eval.app import LLM
from trulens_eval.app import Other
from trulens_eval.app import Prompt
from trulens_eval.db import MULTI_CALL_NAME_DELIMITER
from trulens_eval.react_components.record_viewer import record_viewer
from trulens_eval.schema import Record
from trulens_eval.schema import Select
from trulens_eval.util import jsonify
from trulens_eval.util import JSONPath
from trulens_eval.ux.components import draw_call
from trulens_eval.ux.components import draw_llm_info
from trulens_eval.ux.components import draw_metadata
from trulens_eval.ux.components import draw_prompt_info
from trulens_eval.ux.components import render_selector_markdown
from trulens_eval.ux.components import write_or_json
from streamlit_extras.switch_page_button import switch_page
from trulens_eval.ux.styles import cellstyle_jscode
from trulens_eval.utils.shared_info import (
    question_info,
    system_message_explanation,
    user_message_explanation,
    temperature_explanation,
    model_name_explanation,
)

st.set_page_config(page_title="Model Details", layout="wide")



st.runtime.legacy_caching.clear_cache()

add_logo()

tru = Tru()
lms = tru.db

state = st.session_state

# CHECK TO MAKE SURE MODEL IS SELECTED FIRST AND IF NOT REDIRECT
if 'app_id' not in state:
    st.write("Please select a model first.")
    sleep(3.0)
    switch_page('Leaderboard')


full_df = state.all_records
model_df = full_df[full_df['app_id'] == state.app_id]
feedback_cols = state.feedback_cols

# now extract out record details
model_df['output'] = model_df['output'].apply(lambda x: json.loads(x))
model_df['answer'] = model_df['output'].apply(lambda x: x['answer'])
model_df['citation'] = model_df['output'].apply(lambda x: x['citation'])

# Reorder the columns as desired
desired_columns_order = ['transcript_id', 'answer', 'citation', 'is_correct', 'false_n', 'false_p']
all_columns = model_df.columns.tolist()
rest_columns = [col for col in all_columns if col not in desired_columns_order]
new_columns_order = desired_columns_order + rest_columns
model_df = model_df[new_columns_order]


# get relevant info from app json
app_json = json.loads(model_df.iloc[0]['app_json'])
system_message = app_json['app']['chains'][0]['prompt']['messages'][0]['prompt']['template']
user_message = app_json['app']['chains'][0]['prompt']['messages'][1]['prompt']['partial_variables']['question']
temperature = app_json['app']['chains'][0]['llm']['temperature']
model_name = app_json['app']['chains'][0]['llm']['model_name']

# now display the model details
st.title("Model Details")

# Display the system message
st.header("System Message:")
st.write(f"**System Message:** {system_message}\n\n{system_message_explanation}")

# Display the user message
st.header("User Message:")
st.write(f"**CallMiner Metric:** {state['question']}")
st.write(f"**Prompt Question:** {user_message}\n\n{user_message_explanation}")

# Display the temperature
st.header("Temperature:")
st.write(f"**Temperature:** {temperature}\n\n{temperature_explanation}")

# Display the model name
st.header("Model Name:")
st.write(f"**Model Name:** {model_name}\n\n{model_name_explanation}")


if "clipboard" not in state:
    state.clipboard = "nothing"

if state.clipboard:
    ret = st_javascript(
        f"""navigator.clipboard.writeText("{state.clipboard}")
    .then(
        function() {{
            console.log('success?')
        }},
        function(err) {{
            console.error("Async: Could not copy text: ", err)
        }}
    )
"""
    )


def jsonify_for_ui(*args, **kwargs):
    return jsonify(*args, **kwargs, redact_keys=True, skip_specials=True)


def render_component(query, component, header=True):
    # Draw the accessor/path within the wrapped app of the component.
    if header:
        st.subheader(
            f"Component {render_selector_markdown(Select.for_app(query))}"
        )

    # Draw the python class information of this component.
    cls = component.cls
    base_cls = cls.base_class()
    label = f"__{repr(cls)}__"
    if str(base_cls) != str(cls):
        label += f" < __{repr(base_cls)}__"
    st.write("Python class: " + label)

    # Per-component-type drawing routines.
    if isinstance(component, LLM):
        draw_llm_info(component=component, query=query)

    elif isinstance(component, Prompt):
        draw_prompt_info(component=component, query=query)

    elif isinstance(component, Other):
        with st.expander("Uncategorized Component Details:"):
            st.json(jsonify_for_ui(component.json))

    else:
        with st.expander("Unhandled Component Details:"):
            st.json(jsonify_for_ui(component.json))


if model_df.empty:
    st.write("No records yet...")

else:

    tab1, tab2 = st.tabs(["Records", "Feedback Functions"])

    with tab1:

        gridOptions = {'alwaysShowHorizontalScroll': True,
                       'pagination': True,
                       'paginationPageSize': 10,}
        gb = GridOptionsBuilder.from_dataframe(model_df)

        cellstyle_jscode = JsCode(cellstyle_jscode)
        gb.configure_column('type', header_name='App Type', hide=True)
        gb.configure_column('question', header_name='Question', hide=True)
        gb.configure_column('record_json', header_name='Record JSON', hide=True)
        gb.configure_column('app_json', header_name='App JSON', hide=True)
        gb.configure_column('cost_json', header_name='Cost JSON', hide=True)
        gb.configure_column('perf_json', header_name='Perf. JSON', hide=True)

        gb.configure_column('record_id', header_name='Record ID', hide=True)
        gb.configure_column('app_id', header_name='App ID', hide=True)
        gb.configure_column('transcript_id', header_name='Transcript ID')
        gb.configure_column('answer', header_name='Answer')
        gb.configure_column('citation', header_name='Citation')

        gb.configure_column('feedback_id', header_name='Feedback ID', hide=True)
        gb.configure_column('input', header_name='User Input', hide=True)
        gb.configure_column('output', header_name='Response', hide=True)
        gb.configure_column('total_tokens', header_name='Total Tokens (#)')
        gb.configure_column('total_cost', header_name='Total Cost (USD)')
        gb.configure_column('latency', header_name='Latency (Seconds)')
        gb.configure_column('tags', header_name='Tags', hide=True)
        gb.configure_column('ts', header_name='Time Stamp', hide=True)

        non_feedback_cols = [
            'app_id', 'type', 'ts', 'transcript_id' ,'total_tokens', 'total_cost', 'record_json',
            'latency', 'record_id', 'app_id', 'cost_json', 'app_json', 'input',
            'output', 'perf_json', 'question', 'tags', 'answer', 'citation'
        ]

        for feedback_col in model_df.columns.drop(non_feedback_cols):
            gb.configure_column(
                feedback_col,
                cellStyle=cellstyle_jscode,
                hide=feedback_col.endswith("_calls")
            )
        gb.configure_pagination()
        gb.configure_side_bar()
        gb.configure_selection(selection_mode="single", use_checkbox=False)
        #gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
        gridOptions = gb.build()
        data = AgGrid(
            model_df,
            gridOptions=gridOptions,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            allow_unsafe_jscode=True
        )

        selected_rows = data['selected_rows']
        selected_rows = pd.DataFrame(selected_rows)

        if len(selected_rows) == 0:
            st.write("Hint: select a row to display details of a record")

        else:
            st.text(f"Selected Record ID: {selected_rows['record_id'][0]}")

            prompt = selected_rows['input'][0]
            response = selected_rows['output'][0]
            details = selected_rows['app_json'][0]

            with st.expander(
                    f"Input {render_selector_markdown(Select.RecordInput)}",
                    expanded=True):
                write_or_json(st, obj=prompt)

            with st.expander(
                    f"Response {render_selector_markdown(Select.RecordOutput)}",
                    expanded=True):
                write_or_json(st, obj=response)

            metadata = app_json.get('metadata')
            if metadata:
                with st.expander("Metadata"):
                    st.markdown(draw_metadata(metadata))

            row = selected_rows.head().iloc[0]

            st.header("Feedback")
            for fcol in feedback_cols:
                feedback_name = fcol
                feedback_result = row[fcol]
                if MULTI_CALL_NAME_DELIMITER in fcol:
                    fcol = fcol.split(MULTI_CALL_NAME_DELIMITER)[0]
                feedback_calls = row[f"{fcol}_calls"]

                def display_feedback_call(call):

                    def highlight(s):
                        cat = CATEGORY.of_score(s.result)
                        return [f'background-color: {cat.color}'] * len(s)

                    if call is not None and len(call) > 0:

                        df = pd.DataFrame.from_records(
                            [call[i]["args"] for i in range(len(call))]
                        )
                        df["result"] = pd.DataFrame(
                            [
                                float(call[i]["ret"] or -1)
                                for i in range(len(call))
                            ]
                        )
                        df["meta"] = pd.Series(
                            [call[i]["meta"] for i in range(len(call))]
                        )
                        df = df.join(df.meta.apply(lambda m: pd.Series(m))
                                    ).drop(columns="meta")

                        st.dataframe(
                            df.style.apply(highlight, axis=1
                                          ).format("{:.2}", subset=["result"])
                        )

                    else:
                        st.text("No feedback details.")

                with st.expander(f"{feedback_name} = {feedback_result}",
                                 expanded=True):
                    display_feedback_call(feedback_calls)

            record_str = selected_rows['record_json'][0]
            record_json = json.loads(record_str)
            record = Record(**record_json)

            classes: Iterable[Tuple[JSONPath, ComponentView]
                             ] = list(instrumented_component_views(app_json))
            classes_map = {path: view for path, view in classes}

            st.header('Timeline')
            val = record_viewer(record_json, app_json)

            match_query = None

            # Assumes record_json['perf']['start_time'] is always present
            if val != record_json['perf']['start_time'] and val != '':
                match = None
                for call in record.calls:
                    if call.perf.start_time.isoformat() == val:
                        match = call
                        break

                if match:
                    length = len(match.stack)
                    app_call = match.stack[length - 1]

                    match_query = match.top().path

                    st.subheader(
                        f"{app_call.method.obj.cls.name} {render_selector_markdown(Select.for_app(match_query))}"
                    )

                    draw_call(match)

                    view = classes_map.get(match_query)
                    if view is not None:
                        render_component(
                            query=match_query, component=view, header=False
                        )
                    else:
                        st.write(
                            f"Call by {match_query} was not associated with any instrumented component."
                        )
                        # Look up whether there was any data at that path even if not an instrumented component:
                        app_component_json = list(match_query(app_json))[0]
                        if app_component_json is not None:
                            with st.expander(
                                    "Uninstrumented app component details."):
                                st.json(app_component_json)

                else:
                    st.text('No match found')
            else:
                st.subheader(f"App {render_selector_markdown(Select.App)}")
                with st.expander("App Details:"):
                    st.json(jsonify_for_ui(app_json))

            if match_query is not None:
                st.header("Subcomponents:")

                for query, component in classes:
                    if not match_query.is_immediate_prefix_of(query):
                        continue

                    if len(query.path) == 0:
                        # Skip App, will still list App.app under "app".
                        continue

                    render_component(query, component)

            st.header("More options:")

            if st.button("Display full app json"):

                st.write(jsonify_for_ui(app_json))

            if st.button("Display full record json"):

                st.write(jsonify_for_ui(record_json))

    with tab2:
        feedback = feedback_cols
        cols = 4
        rows = len(feedback) // cols + 1

        for row_num in range(rows):
            with st.container():
                columns = st.columns(cols)
                for col_num in range(cols):
                    with columns[col_num]:
                        ind = row_num * cols + col_num
                        if ind < len(feedback):
                            # Generate histogram
                            fig, ax = plt.subplots()
                            bins = [
                                0, 0.2, 0.4, 0.6, 0.8, 1.0
                            ]  # Quintile buckets
                            ax.hist(
                                model_df[feedback[ind]],
                                bins=bins,
                                edgecolor='black',
                                color='#2D736D'
                            )
                            ax.set_xlabel('Feedback Value')
                            ax.set_ylabel('Frequency')
                            ax.set_title(feedback[ind], loc='center')
                            st.pyplot(fig)

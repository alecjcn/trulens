# Import necessary libraries
import streamlit as st
from trulens_eval.utils.custom_chain import create_tru_chain
from trulens_eval.utils.transcriptprocessing import label_transcript
from trulens_eval import Tru
import glob
import os
from streamlit_extras.switch_page_button import switch_page
import plotly.graph_objects as go
from trulens_eval.utils.shared_info import (
    question_info,
    system_message_explanation,
    user_message_explanation,
    temperature_explanation,
    model_name_explanation,
)
# Be sure to wrap all calls to track our records
tru = Tru()
lms = tru.db

state = st.session_state

# Instructions for the user
st.markdown("## Instructions")
st.write("This dashboard allows you to process transcripts using a language model.")
st.write("Follow the instructions below to get started:")

############################################################################################################
st.markdown("### Step 1: Select Question To Evaluate")
st.write("Specify which CallMiner evaluation metric to score below.")

# Check if we already have a default question to focus on
if 'question' in state:
    default_question = state['question']
else:
    default_question = None

selected_index = None
if default_question is not None and default_question in question_info:
    selected_index = list(question_info.keys()).index(default_question)

if selected_index is not None:
    callminer_metric = st.selectbox("Select Question Metric", list(question_info.keys()), key="question_metric", index=selected_index)
else:
    callminer_metric = st.selectbox("Select Question Metric", list(question_info.keys()), key="question_metric")


# Extract the question number and whether to invert the output from the question_info dictionary
selected_question_info = question_info.get(callminer_metric)
if selected_question_info:
    question_number = selected_question_info['question']
    invert_answer = selected_question_info['invert_llm_output']

############################################################################################################
st.markdown("### Step 2: Devise Prompt")

# Write instructions for the user
st.write(
    "While technically everything you feed into the language model constitutes *the prompt*, including the "
    "transcript, any labels of the transcript, formatting instructions, the question, etc. we will focus on two main components. "
    + system_message_explanation + " " + user_message_explanation + "\n"
)

# Create a text area for the system message
system_message_default = (
    "A transcript of a marketing call will be presented to you, enclosed within triple quotes, followed by a question."
    " The question provided will focus solely on the interactions between the marketer and the patient."
)
system_message = st.text_area("System Message", system_message_default)

# Create a text area for the question
question = st.text_area("Question", value="", placeholder= "Enter your question here...")

# Write instructions for the user regarding the invert_answer checkbox
st.write(
    "Lastly, we need to specify whether to invert the output of the language model. "
    "Many of our scoring metrics are in the negative form, e.g. 'Rep Didn't Claim No Cost'. "
    "However, we often ask questions in the affirmative form, e.g. 'Did the rep claim no cost?' "
    "In this case, we need to invert the output of the language model. Please specify whether to invert the output of the language model."
)
# Create a checkbox to allow the user to invert the output
invert_answer = st.checkbox("Invert Output", value=invert_answer)

############################################################################################################

st.markdown("### Step 3: Model Parameters")
st.write(
    "Configure model parameters to control the behavior of the language model." + temperature_explanation + " " + model_name_explanation +
    " In this format, we can only use a single model for now. We will add more models in the future."
)
temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
model_options = ["gpt-3.5-turbo-16k"]  # Add more models if needed
model = st.selectbox("Select Model", model_options, index=0)

############################################################################################################

st.markdown("### Step 4: Answer and Citation Descriptions")
st.write(
    "As you're likely familiar, these chat models do not generally output structured data. "
    "To get it to produce structured data and extract it, we had to define a model for its responses. "
    "We have defined this model to include an *answer* and a *citation*. "
    "For both pieces of information that we expect in the model's response, we can provide a description "
    "of what we expect the model to produce, and this will be used to add instructions onto the prompt we "
    "have already devised. This description will be used to instruct the model on what "
    "to produce. Assuming it produces something that matches the description, we can extract it. "
    "Behind the scenes, there's even more magic happening, including verifying that the model's output "
    "matches both the format and additional rules. For example, if the answer is true, the citation must "
    "be non-empty. If the output doesn't meet these criteria, the model may be prompted again.\n"
)


answer_field_description = st.text_input("Answer Field Description", "true/false answer to the provided question")
citation_field_description = st.text_input("Citation Field Description", "citation from transcript of marketer if answer is True")

# Create counters to keep track of performance metrics
false_n = 0
false_p = 0
correct = 0
total = 0

# Create placeholders for the grouped bar chart
performance_fig = go.FigureWidget(
    layout={'title': 'Performance Metrics'},
    data=[
        go.Bar(x=['Accuracy', 'False Negative', 'False Positive'], y=[0, 0, 0], name='Metrics', marker_color=['blue', 'red', 'green'])
    ]
)
# Set the y-axis range to always be from 0 to 100
performance_fig.update_yaxes(range=[0, 100])

pl = st.empty()

# Before your loop
if 'stop_execution' not in st.session_state:
    st.session_state.stop_execution = False

# Function to update the performance chart
def update_performance_chart():
    accuracy = correct / total
    false_n_rate = false_n / total
    false_p_rate = false_p / total
    performance_fig.data[0].y = [accuracy * 100, false_n_rate * 100, false_p_rate * 100]
    pl.plotly_chart(performance_fig)

# Initialize the chart
pl.plotly_chart(performance_fig)

state.model_finished = False
if st.button("Run Chain"):
    st.write("system_message: ", system_message)
    st.write("question: ", question)
    st.write("callminer_metric: ", callminer_metric)
    st.write("invert_answer: ", invert_answer)
    st.write("model: ", model)
    st.write("temperature: ", temperature)

    # Combine the user's message with the format_instructions
    system_message = system_message + "\n\n{format_instructions}"

    # create truchain app
    truchain = create_tru_chain(answer_field_description=answer_field_description, citation_field_description=citation_field_description, system_message=system_message, 
                                question=question, callminer_metric=callminer_metric, invert_answer=invert_answer, 
                                model=model, temperature=temperature)
    
    # st.write("truchain: ", truchain)
    
    # st.session_state.tru.add_app(app=truchain)

    transcripts_path = "/Users/alec/Documents/Documents - Alec’s MacBook Pro/trulens_testing/transcripts"
    txt_files = glob.glob(os.path.join(transcripts_path, '*.txt'))


    # Initialize a dictionary to store errors
    # The keys will be file names, and the values will be error messages
    error_dict = {}

    # iterate over files in directory
    for file_path in txt_files:
        
        try:
            labeled_transcript, answer, eureka_id = label_transcript(file_path, question_number)
            print("answer: ", answer)
            print("eureka_id: ", eureka_id)
            print("label_transcript: ", labeled_transcript)
            
            response, rec = truchain.call_with_record({"transcript": labeled_transcript, "expected_value": answer}, transcript_id=eureka_id)
            print("response: ", response)
            is_correct = rec.main_output['answer'] == answer
            print("is_correct: ", is_correct)

            if is_correct:
                correct += 1
            elif answer == 'true':
                false_n += 1
            else:
                false_p += 1
            total += 1
            
            update_performance_chart()
            
        except Exception as e:
            # Handle the error and continue with the next file
            print(f"Error processing file {file_path}: {e}")

        if total == 6:
            total = 0
            correct = 0
            false_n = 0
            false_p = 0
            state.model_finished = True
            break

if state.model_finished:
    if st.button("View Model Details"):
        state.app_id = truchain.app_id
        switch_page('Model Details')
        
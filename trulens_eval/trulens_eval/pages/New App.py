# Import necessary libraries
import streamlit as st
from utils.custom_chain import create_tru_chain
from utils.transcriptprocessing import label_transcript
import glob
import os
import shutil

# Title of the page
st.title("Test New Question")

# Input fields for answer and citation descriptions
answer_field_description = st.text_input("Answer Field Description", "true/false answer to the provided question")
citation_field_description = st.text_input("Citation Field Description", "citation from transcript of marketer if answer is True")

# Sliders and selectors for model parameters
temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
model_options = ["gpt-3.5-turbo-16k"]  # Add more models if needed
model = st.selectbox("Select Model", model_options, index=0)

# Text area for system message and question
system_message_default = "A transcript of a marketing call will be presented to you, enclosed within triple quotes, followed by a question. The question provided will focuses solely on the interactions between the marketer and the patient. \n{format_instructions}"
system_message = st.text_area("System Message", system_message_default)

question_default = '''
Does the marketer explicitly state that the test will be entirely covered by insurance, or that the test is somehow a program of the patient's insurance or Medicare program?
'''
question = st.text_area("Question", question_default)

# Dictionary to map the labels to their descriptions
question_info = [
    {'question': "Q3", 'invert_llm_output': False, 'callminer_metric': "Rep Asks If Patient Makes Medical Decisions"},
    {'question': "Q4", 'invert_llm_output': True, 'callminer_metric': "Patient Didn`t State Not Making Medical Decisions"},
    {'question': "Q5", 'invert_llm_output': False, 'callminer_metric': "Rep Asks If Test Taken Before"},
    {'question': "Q6", 'invert_llm_output': True, 'callminer_metric': "Patient Didn`t State Has Taken Test"},
    {'question': "Q7", 'invert_llm_output': True, 'callminer_metric': "Rep Didn`t State Test Required"},
    {'question': "Q8", 'invert_llm_output': True, 'callminer_metric': "Rep Didn`t Claim Affiliation"},
    {'question': "Q9", 'invert_llm_output': True, 'callminer_metric': "Rep Didn`t Claim Benefit of Insurance"},
    {'question': "Q10", 'invert_llm_output': True, 'callminer_metric': "Rep Didn`t Claim No Cost"}
]
# Selectbox for question number using callminer_metric as display
selected_metric = st.selectbox("Select Question Metric", [q['callminer_metric'] for q in question_info])
selected_question_info = next((q for q in question_info if q['callminer_metric'] == selected_metric), None)

question_number = selected_question_info['question']
callminer_metric = selected_question_info['callminer_metric']
invert_answer = selected_question_info['invert_llm_output']

if st.button("Run Chain"):
    st.write("button clicked")
    # create truchain app
    truchain = create_tru_chain(answer_field_description=answer_field_description, citation_field_description=citation_field_description, system_message=system_message, 
                                question=question, callminer_metric=callminer_metric, invert_answer=invert_answer, 
                                model=model, temperature=temperature)

    transcripts_path = "/Users/alec/Documents/trulens_testing/transcripts"
    new_transcripts_path = "/Users/alec/Documents/trulens_testing/transcripts"
    txt_files = glob.glob(os.path.join(transcripts_path, '*.txt'))


    # Initialize a dictionary to store errors
    # The keys will be file names, and the values will be error messages
    error_dict = {}

    # iterate over files in directory
    for file_path in txt_files:
        try:
            labeled_transcript, answer, eureka_id = label_transcript(file_path, question_number)
            # Display the first couple of lines of the labeled_transcript
            first_few_lines = '\n'.join(labeled_transcript.split('\n')[:2])  # Adjust the number '2' to display more or fewer lines
            st.write(f"First few lines of {os.path.basename(file_path)}:\n{first_few_lines}")
            st.write(f"answer: {answer}")
            st.write(f"eureka_id: {eureka_id}")
            
            response, rec = truchain.call_with_record({"transcript": labeled_transcript, "answer": answer}, transcript_id= eureka_id)
            
            st.write(f"response: {response}")
            # Copy the successful transcript to the new folder
            shutil.copy(file_path, new_transcripts_path)
            
            st.write(f"response: {response}")
            print(f"rec: {rec}")
        except Exception as e:
            # Handle the error and continue with the next file
            st.write(f"Error processing file {file_path}: {e}")

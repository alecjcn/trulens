# Dictionary to map the labels to their descriptions
question_info = {
    "Rep Asks If Patient Makes Medical Decisions": {'question': "Q3", 'invert_llm_output': False},
    "Patient Didn't State Not Making Medical Decisions": {'question': "Q4", 'invert_llm_output': True},
    "Rep Asks If Test Taken Before": {'question': "Q5", 'invert_llm_output': False},
    "Patient Didn't State Has Taken Test": {'question': "Q6", 'invert_llm_output': True},
    "Rep Didn't State Test Required": {'question': "Q7", 'invert_llm_output': True},
    "Rep Didn't Claim Affiliation": {'question': "Q8", 'invert_llm_output': True},
    "Rep Didn't Claim Benefit of Insurance": {'question': "Q9", 'invert_llm_output': True},
    "Rep Didn't Claim No Cost": {'question': "Q10", 'invert_llm_output': True}
}

system_message_explanation = (
    "The system message is a message or instruction provided to the language model before generating a response. "
    "It can be used to set the context or behavior of the model for the conversation. It's a way to fine-tune the "
    "interaction without needing to modify the underlying code or model."
)

user_message_explanation = (
    "The question is what we are asking the model to answer using the provided transcript. We actually combine these into "
    "a user message, which is the input provided by the user to the language model. It serves as the starting point "
    "for generating responses based on the user's input."
)

temperature_explanation = (
    "Temperature controls the randomness of the model's responses. It ranges from 0 to 1. Higher values make the output "
    "more random or creative, while lower values make it more deterministic."
)

model_name_explanation = (
    "The model name specifies the version or variant of the language model you are using."
)

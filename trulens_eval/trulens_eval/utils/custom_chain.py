from pydantic import BaseModel, Field, validator
from langchain.chains import LLMChain, TransformChain, SequentialChain
from langchain.prompts.chat import ChatPromptTemplate, PromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from trulens_eval import TruChain, Feedback, Tru, Provider, Select
from pydantic import BaseModel, Field, validator
from trulens_eval.utils.customparsers import ChatRetryWithErrorOutputParser


class Answer(BaseModel):
    '''Answer to a question about a transcript'''

    # class variables for dynamic descriptions
    answer_desc = "true/false answer to the provided question"
    citation_desc = "citation from transcript of marketer if answer is True"
    invert_answer = False

    original_answer: bool = Field(..., description="Original LLM answer")
    answer: bool = Field(..., description="Processed answer (may be inverted)")
    citation: str = Field(..., description="Dynamic description will be set")

    @validator("answer", pre=True, always=True)
    def set_and_invert_answer(cls, v, values):
        values["original_answer"] = v
        if cls.invert_answer:
            return not v
        return v

    @validator("citation", always=True)
    def true_answers_have_citation(cls, citation, values):
        original_answer = values.get('original_answer')
        if original_answer is True:
            if not citation or citation.strip() == "":
                raise ValueError("You previously answered true, but did not provide a citation from the provided transcript.")
        return citation

class Accuracy_positive(Provider):
    def is_correct(self, response: dict) -> float:
        if response['answer'] == response['expected_value']:
            print("llm correct")
            return 1.
        else:
            print("llm incorrect")
            return 0.
    def false_p(self, response: dict) -> float:
        print("false_p function entered")
        if response['answer'] and not response['expected_value']:
            return 1.
        else:
            return 0.
    def false_n(self, response: dict) -> float:
        print("false_n function entered")
        if not response['answer'] and response['expected_value']:
            return 1.
        else:
            return 0.

def create_tru_chain(answer_field_description, citation_field_description, system_message, question, callminer_metric ,invert_answer, model, temperature):

    Answer.answer_desc = answer_field_description
    Answer.citation_desc = citation_field_description
    Answer.invert_answer = invert_answer

    # intialize the parser
    pydantic_parser = PydanticOutputParser(pydantic_object=Answer)

    # intialize the message prompt templates
    # Define the prompt templates
    system_template = PromptTemplate(
        template=system_message,
        input_variables=[],
        partial_variables={"format_instructions": pydantic_parser.get_format_instructions()}
    )
    system_prompt_template = SystemMessagePromptTemplate(prompt=system_template)

    system_template = PromptTemplate(
        template=system_message,
        input_variables=[],
        partial_variables={"format_instructions": pydantic_parser.get_format_instructions()}
    )

    system_prompt_template = SystemMessagePromptTemplate(prompt=system_template)

    user_message = '''
    """{transcript}"""
    Question: {question}'''
    user_full_template = PromptTemplate(
        template=user_message,
        input_variables=["transcript"],
        partial_variables={"question": question}
    )
    user_prompt_template = HumanMessagePromptTemplate(prompt=user_full_template)

    chat_template = ChatPromptTemplate.from_messages([system_prompt_template, user_prompt_template])

    print(chat_template)

    # define the model, and the llm chain
    chat = ChatOpenAI(temperature=temperature,model=model,verbose=True)

    llm_chain = LLMChain(llm=chat, prompt=chat_template, output_key="json_string")

    parser = ChatRetryWithErrorOutputParser.from_llm(parser=pydantic_parser, llm=chat, prompt=chat_template)

    # define the transformation
    def parse_outputs(inputs: dict) -> dict:
        text = inputs['json_string']
        transcript = inputs['transcript']
        expected_value = inputs['expected_value']
        parsed_answer = parser.parse_with_prompt(text, {'transcript': transcript})  # Assume this returns an instance of the Answer class
        
        # Construct the output dictionary
        output = {
            'answer': parsed_answer.answer,
            'citation': parsed_answer.citation,
            'expected_value': expected_value,
        }
        return {'result': output}

    # define the transform chain
    transformChain = TransformChain(
        input_variables=['json_string', 'transcript', 'expected_value'],
        output_variables=['result'],
        transform=parse_outputs
    )

    # define the full chain
    chain = SequentialChain(
        input_variables=['transcript', 'expected_value'],
        output_variables=['result'],
        chains=[llm_chain, transformChain],
    )

    custom = Accuracy_positive()

    f_correct = Feedback(custom.is_correct).on(Select.Record.main_output)
    f_false_p = Feedback(custom.false_p).on(Select.Record.main_output)
    f_false_n = Feedback(custom.false_n).on(Select.Record.main_output)

    tru = Tru()
    # wrap the chain in a truchain to record the results
    truchain = TruChain(chain,
                        question=callminer_metric,
                        feedbacks=[f_correct,f_false_n,f_false_p],
                        tru=tru)
    
    return truchain
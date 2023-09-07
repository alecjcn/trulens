from __future__ import annotations
from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import BaseOutputParser, BasePromptTemplate, OutputParserException, PromptValue
from langchain.schema.language_model import BaseLanguageModel
from langchain.output_parsers import  RetryWithErrorOutputParser
from typing import Any, Dict, TypeVar
import copy

T = TypeVar("T")

class ChatRetryWithErrorOutputParser(RetryWithErrorOutputParser[T]):

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        parser: BaseOutputParser[T],
        prompt: BasePromptTemplate,
    ) -> ChatRetryWithErrorOutputParser[T]:
        """Create a CustomRetryWithErrorOutputParser from an LLM.

        Args:
            llm: The LLM to use to retry the completion.
            parser: The parser to use to parse the output.
            prompt: The prompt to use to retry the completion.

        Returns:
            A CustomRetryWithErrorOutputParser.
        """
        # Check if the provided prompt is an instance of ChatPromptTemplate
        if isinstance(prompt, ChatPromptTemplate):


            new_prompt = copy.deepcopy(prompt)

            new_prompt.input_variables.append('completion')
            new_prompt.input_variables.append('error')
            # Append the AI feedback message
            new_prompt.append(("ai", "{completion}"))
            
            new_prompt.append(("human", 
            """The answer you provided above did not satisfy the constraints given in the Prompt.
            Details: {error}
            Please revisit the transcript and the question, and reanswer the question."""))

        chain = LLMChain(llm=llm, prompt=new_prompt)
        return cls(parser=parser, retry_chain=chain)

    
    def parse_with_prompt(self, completion: str, prompt_values: Dict[str,Any]) -> T:
        """Parse the completion with the prompt values.

        Args:
            completion: The completion to parse.
            prompt_values: The prompt values to use.

        Returns:
            The parsed completion.
        """
        attempts = 0
        max_attempts = 3
        parsed_completion = None

        while attempts < max_attempts:
            try:
                parsed_completion = self.parser.parse(completion)
                break
            except OutputParserException as e:
                new_completion = self.retry_chain.run(
                    **prompt_values, completion=completion, error=repr(e)
                )
                completion = new_completion
                attempts += 1

        if attempts == max_attempts:
            raise OutputParserException(
                f"Failed to parse completion after {max_attempts} attempts."
            )

        return parsed_completion
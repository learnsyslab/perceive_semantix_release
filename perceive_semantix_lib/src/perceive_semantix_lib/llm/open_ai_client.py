import asyncio
from logging import getLogger
from typing import Callable, Optional, TypeVar

from joblib import Memory
from openai import AsyncOpenAI, OpenAI

InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")

logger = getLogger(__name__)


class OpenAIParallelClient:
    """Class for making multiple requests to OpenAI LLM in parallel."""

    client: Optional[AsyncOpenAI] = None
    memory = Memory(location="__openai_cache__", verbose=0)

    @staticmethod
    def api_key_is_valid() -> bool:
        try:
            client = OpenAI()
            client.models.list()
            return True
        except Exception as e:
            logger.warning(f"OpenAI API key is not set or invalid: {e}")
            return False

    @classmethod
    async def _make_request(
        cls,
        input_item: InputType,
        system_prompt: str,
        user_prompt: str,
        assistant_prompt: str,
        input_formatter: Callable[[InputType], str],
        output_parser: Callable[[str], OutputType],
        model: str,
    ) -> OutputType:
        formatted_input = input_formatter(input_item)

        completion = await cls.client.chat.completions.create(  # type: ignore
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_prompt},
                {"role": "user", "content": formatted_input},
            ],
        )

        response: str = completion.choices[0].message.content  # type: ignore
        return output_parser(response)

    @classmethod
    async def _run_parallel_requests(
        cls,
        input_list: list[InputType],
        system_prompt: str,
        user_prompt: str,
        assistant_prompt: str,
        input_formatter: Callable[[InputType], str],
        output_parser: Callable[[str], OutputType],
        model: str,
    ) -> list[OutputType]:
        tasks = [
            cls._make_request(item, system_prompt, user_prompt, assistant_prompt, input_formatter, output_parser, model)
            for item in input_list
        ]
        results = await asyncio.gather(*tasks)

        return results

    @classmethod
    @memory.cache
    def run(
        cls,
        input_list: list[InputType],
        system_prompt: str,
        user_prompt: str,
        assistant_prompt: str,
        input_formatter: Callable[[InputType], str],
        output_parser: Callable[[str], OutputType],
        model: str = "gpt-4o",
    ) -> list[OutputType]:
        """Run the OpenAI parallel client with the given parameters.

        Args:
            input_list (list[InputType]): list of input items to process.
            system_prompt (str): System prompt for the OpenAI model.
            user_prompt (str): User prompt for the OpenAI model.
            assistant_prompt (str): Assistant prompt for the OpenAI model.
            input_formatter (Callable[[InputType], str]): Function to format each item in the input list for the prompt.
            output_parser (Callable[[str], OutputType]): Function to parse each request's response.
            model (str): Model to use for the OpenAI API.

        Returns:
            list (OutputType): Accumulated parsed results from all requests.

        """
        if len(input_list) == 0:
            return []

        if cls.client is None:
            cls.client = AsyncOpenAI()

        results = asyncio.run(
            cls._run_parallel_requests(
                input_list,
                system_prompt,
                user_prompt,
                assistant_prompt,
                input_formatter,
                output_parser,
                model,
            )
        )

        return results

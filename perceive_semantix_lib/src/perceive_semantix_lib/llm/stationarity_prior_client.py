from logging import getLogger
from typing import Optional

from perceive_semantix_lib.core.utils.pocd_datatypes import POCDObjectTypes
from perceive_semantix_lib.llm.open_ai_client import OpenAIParallelClient

logger = getLogger(__name__)

OBJECT_STATIONARITY_PROMPT_SYSTEM = """
You are an intelligent assistant for analyzing everyday environments.
"""

OBJECT_STATIONARITY_PROMPT_USER = """
Your task is to classify whether an object is dynamic or static in typical human-centered environments (e.g., homes, kitchens, offices, workspaces).

- A **dynamic object** is something that can be moved and is regularly moved, repositioned, or reoriented (e.g., bottles, laptops, keyboards, doors).
- A **static object** is something that is not easily movable or is rarely moved (e.g., walls, desks, large appliances).

You will be given a single object. Respond with either:

- "Dynamic." followed by one short sentence explaining why the object is considered dynamic,
**OR**
- "Static." followed by one short sentence explaining why the object is considered static.

Examples:
laptop
Dynamic. Laptops are often carried around and used in different locations.

wall
Static. Walls are fixed parts of the building structure and never moved.

chair
Dynamic. Chairs are commonly moved and repositioned by users.

Done examples.

desk"""

OBJECT_STATIONARITY_PROMPT_ASSISTANT = """
Static. Desks are typically heavy and stay in one place for long periods in homes and offices.
"""


class StationarityPriorOpenAIAsyncClient:
    """Static class to query OpenAI for semantic stationarity prior of objects."""

    @staticmethod
    def _object_stationarity_prompt(object_input) -> str:
        return f"{object_input.lower()}"

    @staticmethod
    def _process_stationarity_response(response: str) -> tuple[Optional[POCDObjectTypes], str]:
        try:
            type, explanation = response.split(".", 1)
        except ValueError:
            return None, response

        type = type.strip().lower()
        explanation = explanation.strip()
        if type == "dynamic":
            return POCDObjectTypes.DYNAMIC, explanation
        elif type == "static":
            return POCDObjectTypes.STATIC, explanation
        else:
            return None, explanation

    @staticmethod
    def query_semantic_stationarity(object_list: list[str]) -> dict[str, POCDObjectTypes]:
        if len(object_list) == 0:
            return {}

        results = OpenAIParallelClient.run(
            input_list=object_list,
            system_prompt=OBJECT_STATIONARITY_PROMPT_SYSTEM,
            user_prompt=OBJECT_STATIONARITY_PROMPT_USER,
            assistant_prompt=OBJECT_STATIONARITY_PROMPT_ASSISTANT,
            input_formatter=StationarityPriorOpenAIAsyncClient._object_stationarity_prompt,
            output_parser=StationarityPriorOpenAIAsyncClient._process_stationarity_response,
            model="gpt-4o",
        )

        result_types = {}
        for obj, (type, response) in zip(object_list, results):
            if type is None:
                logger.warning(f"Stationarity prior is None. Input: '{obj}'. Reason: '{response}'")
                result_types[obj] = POCDObjectTypes.DYNAMIC
            else:
                result_types[obj] = type

        return result_types

from logging import getLogger
from typing import Optional

from perceive_semantix_lib.llm.open_ai_client import OpenAIParallelClient

logger = getLogger(__name__)

OBJECT_SIMILARITY_PROMPT_SYSTEM = """
You are an intelligent assistant for analyzing everyday environments.
"""

OBJECT_SIMILARITY_PROMPT_USER = """
Your task is to estimate how likely it is that two given objects are typically located near each other in everyday human environments (such as homes, offices, or workshops).

You will be given two objects, each represented by a single word. Respond with a single integer from 0 to 100, where:

0 means extremely unlikely to be found near each other,

100 means almost always found near each other.

Then, in one short sentence, explain your reasoning based on typical spatial arrangements in human environments.

Example:
keyboard; monitor

In this case, you should output: 95. Keyboards are commonly placed directly in front of monitors on desks.

Example:
toothbrush; refrigerator
In this case, you should output: 5. These items are found in separate rooms—bathroom and kitchen, respectively.

Example:
mug; coffee_machine

In this case, you should output: 90. Mugs are typically kept near coffee machines for convenience.

Done examples.

stapler; printer"""

OBJECT_SIMILARITY_PROMPT_ASSISTANT = (
    """70. Staplers are often kept near printers in office settings for assembling printed documents."""
)

OBJECT_STATIONARITY_PROMPT_SYSTEM = """
You are an intelligent assistant for analyzing everyday environments.
"""


class SimilarityOpenAIAsyncClient:
    """Static Class to query OpenAI for semantic similarity of objects."""

    @staticmethod
    def _object_similarity_prompt(object_input_pair: tuple[str, str]) -> str:
        return f"{object_input_pair[0]}; {object_input_pair[1]}"

    @staticmethod
    def _process_similarity_response(response: str) -> tuple[Optional[float], str]:
        try:
            score_str, explanation = response.split(".", 1)
            score = int(score_str.strip())
            return score, explanation.strip()
        except ValueError:
            return None, response

    @staticmethod
    def query_semantic_similarity(query_object: str, object_list: list[str]) -> dict[str, float]:
        """Query OpenAI for semantic similarity of the given object to all objects in the object list.

        Args:
            query_object (str): The object to compare against the object list.
            object_list (list[str]): List of objects to compare to the query object.

        Returns:
            list[float]: List of similarity scores (0.0-1.0) for each object in `self.object_list`.

        """
        if len(object_list) == 0:
            return {}
        input_items = [(o, query_object) for o in object_list]

        results = OpenAIParallelClient.run(
            input_list=input_items,
            system_prompt=OBJECT_SIMILARITY_PROMPT_SYSTEM,
            user_prompt=OBJECT_SIMILARITY_PROMPT_USER,
            assistant_prompt=OBJECT_SIMILARITY_PROMPT_ASSISTANT,
            input_formatter=SimilarityOpenAIAsyncClient._object_similarity_prompt,
            output_parser=SimilarityOpenAIAsyncClient._process_similarity_response,
            model="gpt-4o",
        )

        result_scores: dict[str, float] = {}
        for obj, (score, response) in zip(object_list, results):
            if score is None:
                logger.warning(f"Similarity score was None. Input: '{obj}'. Reason: '{response}'")
                result_scores[obj] = 0.0
            else:
                result_scores[obj] = score / 100.0

        return result_scores

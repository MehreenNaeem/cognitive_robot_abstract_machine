"""
Language models a prompt is sent to, and the chat completion payload they answer with.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from requests import RequestException
from typing_extensions import Protocol

from llm_planner.ollama_config import DEFAULT_MODEL, chat_with_model, get_response

# %% language models


class LanguageModel(Protocol):
    """
    A model that answers a prompt with text.
    """

    def complete(self, prompt: str) -> str:
        """
        :param prompt: The prompt to answer.
        :return: The model's answer.
        """


class ChatCompletionField(StrEnum):
    """
    The keys of the chat completion payload an OpenAI-compatible endpoint answers with.
    """

    CHOICES = "choices"
    """
    The answers the model produced.
    """

    MESSAGE = "message"
    """
    One choice's message.
    """

    CONTENT = "content"
    """
    A message's text.
    """


class EmptyChatCompletionError(RuntimeError):
    """
    Raised when a chat completion payload carries no answer to read.
    """

    def __init__(self, payload: dict[str, object]) -> None:
        super().__init__(f"Chat completion carries no choice: {payload}")
        self.payload = payload
        """
        The payload that was answered with.
        """


@dataclass
class ChatCompletion:
    """
    The part of an OpenAI-compatible chat completion payload that is read back.
    """

    content: str
    """
    The text of the model's first answer.
    """

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> ChatCompletion:
        """
        Read a chat completion payload.

        :param payload: The endpoint's decoded JSON answer.
        :return: The completion it carries.
        :raises EmptyChatCompletionError: If the payload carries no choice.
        """
        choices = payload.get(ChatCompletionField.CHOICES, [])
        if not choices:
            raise EmptyChatCompletionError(payload)
        return cls(
            content=choices[0][ChatCompletionField.MESSAGE][ChatCompletionField.CONTENT]
        )


class LanguageModelUnreachable(RuntimeError):
    """
    Raised when the language model's endpoint could not be reached at all.
    """

    def __init__(self, url: str, reason: Exception) -> None:
        super().__init__(f"the language model at {url} could not be reached: {reason}")
        self.url = url
        """
        The endpoint that did not answer.
        """

        self.reason = reason
        """
        What the request failed with.
        """


@dataclass
class OllamaLanguageModel:
    """
    A :class:`LanguageModel` served by an Ollama chat completions endpoint.
    """

    url: str
    """
    Address of the chat completions endpoint.
    """

    user_id: str
    """
    Authorization header value the endpoint expects.
    """

    model: str = DEFAULT_MODEL
    """
    Name of the model to prompt.
    """

    def complete(self, prompt: str) -> str:
        """
        :param prompt: The prompt to answer.
        :return: The model's answer, with any reasoning block removed.
        :raises LanguageModelUnreachable: If the endpoint could not be reached.
        """
        try:
            payload = chat_with_model(prompt, self.url, self.user_id, self.model)
        except RequestException as error:
            raise LanguageModelUnreachable(self.url, error) from error
        return get_response(ChatCompletion.from_json(payload).content)

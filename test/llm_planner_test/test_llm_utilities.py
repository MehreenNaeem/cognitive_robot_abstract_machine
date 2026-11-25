"""
Tests for :mod:`llm_planner.env_reasoner.LLM_utilities`.
"""

from __future__ import annotations

import pytest
import requests

from llm_planner.env_reasoner import LLM_utilities
from llm_planner.env_reasoner.LLM_utilities import (
    ChatCompletion,
    ChatCompletionField,
    EmptyChatCompletionError,
    LanguageModelUnreachable,
    OllamaLanguageModel,
)
from llm_planner.ollama_config import DEFAULT_MODEL, get_response

# %% chat completion and language models


class TestChatCompletion:
    def test_reads_the_first_choices_message_content(self) -> None:
        payload = {
            ChatCompletionField.CHOICES: [
                {ChatCompletionField.MESSAGE: {ChatCompletionField.CONTENT: "answer"}}
            ]
        }

        assert ChatCompletion.from_json(payload).content == "answer"

    def test_rejects_a_payload_without_a_choice(self) -> None:
        payload = {ChatCompletionField.CHOICES: []}

        with pytest.raises(EmptyChatCompletionError) as excinfo:
            ChatCompletion.from_json(payload)

        assert excinfo.value.payload is payload


class TestOllamaLanguageModel:
    def test_completes_with_the_content_the_endpoint_answered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        answered = "<think>hmm</think>[]"
        asked = {}

        def chat_with_model(query, url, user_id, model):
            asked.update(query=query, url=url, user_id=user_id, model=model)
            return {
                ChatCompletionField.CHOICES: [
                    {
                        ChatCompletionField.MESSAGE: {
                            ChatCompletionField.CONTENT: answered
                        }
                    }
                ]
            }

        monkeypatch.setattr(LLM_utilities, "chat_with_model", chat_with_model)
        model = OllamaLanguageModel(url="http://model", user_id="Bearer token")

        assert model.complete("prompt") == get_response(answered)
        assert asked == {
            "query": "prompt",
            "url": "http://model",
            "user_id": "Bearer token",
            "model": DEFAULT_MODEL,
        }


class TestUnreachableLanguageModel:
    def test_a_transport_failure_names_the_endpoint_that_did_not_answer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        A refused connection must arrive as a language-model failure, not as the
        request library's own exception leaking through every caller.
        """
        failure = requests.exceptions.ConnectionError("connection refused")

        def chat_with_model(query, url, user_id, model):
            raise failure

        monkeypatch.setattr(LLM_utilities, "chat_with_model", chat_with_model)
        model = OllamaLanguageModel(url="http://model", user_id="Bearer token")

        with pytest.raises(LanguageModelUnreachable) as excinfo:
            model.complete("prompt")

        assert excinfo.value.url == "http://model"
        assert excinfo.value.reason is failure

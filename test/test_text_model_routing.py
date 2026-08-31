from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException

from services.account_service import AccountService
from services.model_service import ModelRoute, ModelUnavailableError
from services.openai_backend_api import ChatRequirements, OpenAIBackendAPI
from services.protocol import (
    anthropic_v1_messages,
    conversation,
    openai_v1_chat_complete,
    openai_v1_response,
)
from services.storage.json_storage import JSONStorageBackend


class TextAccountRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.service = AccountService(
            JSONStorageBackend(Path(self.temp_dir.name) / "accounts.json")
        )
        self.service.add_account_items(
            [
                {"access_token": "free", "type": "free", "status": "正常"},
                {"access_token": "plus", "type": "free", "status": "正常"},
                {"access_token": "pro", "type": "Pro", "status": "正常"},
                {"access_token": "pro-disabled", "type": "Pro", "status": "禁用"},
            ]
        )
        self.service.refresh_access_token = lambda token, **_kwargs: token

    def test_explicit_model_selects_only_advertising_account(self) -> None:
        route = ModelRoute(access_tokens=frozenset({"pro"}), allow_anonymous=False)
        with mock.patch(
            "services.model_service.model_catalog_service.route_for_model",
            return_value=route,
        ):
            token = self.service.get_text_access_token(model="pro-only")

        self.assertEqual(token, "pro")

    def test_account_level_route_works_when_plan_types_match(self) -> None:
        route = ModelRoute(access_tokens=frozenset({"plus"}), allow_anonymous=False)
        with mock.patch(
            "services.model_service.model_catalog_service.route_for_model",
            return_value=route,
        ):
            token = self.service.get_text_access_token(model="misclassified-team-only")

        self.assertEqual(token, "plus")

    def test_auto_model_keeps_existing_unfiltered_rotation(self) -> None:
        with mock.patch(
            "services.model_service.model_catalog_service.route_for_model",
            side_effect=AssertionError("auto must not load the model catalog"),
        ):
            tokens = {
                self.service.get_text_access_token(model="auto"),
                self.service.get_text_access_token(model="auto"),
                self.service.get_text_access_token(model="auto"),
            }

        self.assertEqual(tokens, {"free", "plus", "pro"})

    def test_anonymous_model_uses_anonymous_backend(self) -> None:
        route = ModelRoute(access_tokens=frozenset(), allow_anonymous=True)
        with mock.patch(
            "services.model_service.model_catalog_service.route_for_model",
            return_value=route,
        ):
            token = self.service.get_text_access_token(model="anon-only")

        self.assertEqual(token, "")

    def test_model_without_eligible_account_fails_closed(self) -> None:
        route = ModelRoute(access_tokens=frozenset({"team"}), allow_anonymous=False)
        with mock.patch(
            "services.model_service.model_catalog_service.route_for_model",
            return_value=route,
        ), self.assertRaisesRegex(ModelUnavailableError, "team-only"):
            self.service.get_text_access_token(model="team-only")


class TextProtocolRoutingTests(unittest.TestCase):
    def test_conversation_translates_public_model_for_selected_account(self) -> None:
        backend = mock.Mock(access_token="team")
        backend.stream_conversation.return_value = []
        with mock.patch(
            "services.model_service.model_catalog_service.upstream_model_for",
            return_value="gpt-5-6-thinking",
        ) as resolver:
            result = list(conversation.conversation_events(
                backend,
                messages=[{"role": "user", "content": "hello"}],
                model="gpt-5.6-sol",
                thinking_effort="high",
            ))

        self.assertEqual(result, [])
        resolver.assert_called_once_with("gpt-5.6-sol", "team", "high", "")
        self.assertEqual(backend.stream_conversation.call_args.kwargs["model"], "gpt-5-6-thinking")

    def test_text_backend_passes_requested_model_to_account_selector(self) -> None:
        backend = mock.Mock()
        with (
            mock.patch.object(
                conversation.account_service,
                "get_text_access_token",
                return_value="pro",
            ) as selector,
            mock.patch.object(conversation, "OpenAIBackendAPI", return_value=backend),
        ):
            result = conversation.text_backend("pro-only")

        self.assertIs(result, backend)
        selector.assert_called_once_with(model="pro-only")

    def test_chat_completions_passes_requested_model_to_text_backend(self) -> None:
        body = {
            "model": "pro-chat",
            "messages": [{"role": "user", "content": "route chat"}],
        }
        with (
            mock.patch.object(openai_v1_chat_complete, "text_backend", return_value=object()) as backend,
            mock.patch.object(openai_v1_chat_complete, "collect_text", return_value="ok"),
        ):
            openai_v1_chat_complete.handle(body)

        backend.assert_called_once_with("pro-chat", "")

    def test_chat_completions_none_effort_selects_instant_route(self) -> None:
        body = {
            "model": "gpt-5.6-sol",
            "reasoning_effort": "none",
            "messages": [{"role": "user", "content": "route instant"}],
        }
        captured_requests = []

        def fake_collect(_backend, request):
            captured_requests.append(request)
            return "ok"

        with (
            mock.patch.object(openai_v1_chat_complete, "text_backend", return_value=object()) as backend,
            mock.patch.object(openai_v1_chat_complete, "collect_text", side_effect=fake_collect),
        ):
            openai_v1_chat_complete.handle(body)

        backend.assert_called_once_with("gpt-5.6-sol", "none")
        self.assertEqual(captured_requests[0].thinking_effort, "none")

    def test_chat_completions_rejects_pro_reasoning_mode(self) -> None:
        body = {
            "model": "gpt-5.6-sol",
            "reasoning": {"mode": "pro", "effort": "medium"},
            "messages": [{"role": "user", "content": "route pro"}],
        }

        with self.assertRaises(HTTPException) as raised:
            openai_v1_chat_complete.handle(body)

        self.assertEqual(raised.exception.status_code, 400)

    def test_responses_passes_requested_model_to_text_backend(self) -> None:
        body = {"model": "pro-response", "input": "route response"}
        with (
            mock.patch.object(openai_v1_response, "text_backend", return_value=object()) as backend,
            mock.patch.object(openai_v1_response, "stream_text_deltas", return_value=iter(["ok"])),
        ):
            openai_v1_response.handle(body)

        backend.assert_called_once_with("pro-response", "", "")

    def test_responses_pro_mode_selects_pro_route_and_preserves_public_model(self) -> None:
        body = {
            "model": "gpt-5.6-sol",
            "input": "route pro response",
            "reasoning": {"mode": "pro"},
        }
        captured_requests = []

        def fake_stream(_backend, request):
            captured_requests.append(request)
            yield "ok"

        with (
            mock.patch.object(openai_v1_response, "text_backend", return_value=object()) as backend,
            mock.patch.object(openai_v1_response, "stream_text_deltas", side_effect=fake_stream),
        ):
            openai_v1_response.handle(body)

        backend.assert_called_once_with("gpt-5.6-sol", "medium", "pro")
        self.assertEqual(captured_requests[0].model, "gpt-5.6-sol")
        self.assertEqual(captured_requests[0].thinking_effort, "medium")
        self.assertEqual(captured_requests[0].reasoning_mode, "pro")

    def test_responses_pro_mode_rejects_low_effort(self) -> None:
        body = {
            "model": "gpt-5.6-sol",
            "input": "route pro response",
            "reasoning": {"mode": "pro", "effort": "low"},
        }

        with self.assertRaises(HTTPException) as raised:
            openai_v1_response.handle(body)

        self.assertEqual(raised.exception.status_code, 400)

    def test_anthropic_messages_passes_requested_model_to_account_selector(self) -> None:
        with (
            mock.patch.object(
                anthropic_v1_messages.account_service,
                "get_text_access_token",
                return_value="pro",
            ) as selector,
            mock.patch.object(anthropic_v1_messages, "OpenAIBackendAPI"),
        ):
            request = anthropic_v1_messages.message_request({
                "model": "pro-anthropic",
                "messages": [{"role": "user", "content": "route anthropic"}],
            })

        self.assertEqual(request.model, "pro-anthropic")
        selector.assert_called_once_with(model="pro-anthropic")

    def test_invalid_token_retry_keeps_requested_model_filter(self) -> None:
        initial_backend = SimpleNamespace(access_token="bad")
        request = conversation.ConversationRequest(
            model="pro-only",
            messages=[{"role": "user", "content": "hello"}],
        )

        def fake_events(backend, **_kwargs):
            if backend.access_token == "bad":
                raise RuntimeError("token_invalidated")
            yield {"type": "conversation.delta", "delta": "ok"}

        with (
            mock.patch.object(conversation, "OpenAIBackendAPI", side_effect=lambda access_token: SimpleNamespace(
                access_token=access_token,
                close=lambda: None,
            )),
            mock.patch.object(conversation, "conversation_events", side_effect=fake_events),
            mock.patch.object(
                conversation.account_service,
                "refresh_access_token",
                return_value="bad",
            ),
            mock.patch.object(conversation.account_service, "remove_invalid_token"),
            mock.patch.object(
                conversation.account_service,
                "get_text_access_token",
                return_value="pro",
            ) as selector,
            mock.patch.object(conversation.account_service, "mark_text_used"),
        ):
            result = list(conversation.stream_text_deltas(initial_backend, request))

        self.assertEqual(result, ["ok"])
        selector.assert_called_once_with(
            excluded_tokens={"bad"},
            model="pro-only",
        )


class BackendConversationRoutingTests(unittest.TestCase):
    def test_thinking_effort_uses_chatgpt_web_values(self) -> None:
        expected = {
            "": "",
            "none": "",
            "low": "min",
            "min": "min",
            "medium": "standard",
            "standard": "standard",
            "high": "extended",
            "extended": "extended",
            "xhigh": "max",
            "max": "max",
        }

        for effort, upstream in expected.items():
            with self.subTest(effort=effort):
                self.assertEqual(OpenAIBackendAPI._normalize_thinking_effort(effort), upstream)

    def test_payload_forces_supported_defaults_for_pro_and_work_models(self) -> None:
        backend = object.__new__(OpenAIBackendAPI)
        messages = [{"role": "user", "content": "hello"}]

        with mock.patch(
            "services.openai_backend_api.config",
            SimpleNamespace(default_thinking_effort=""),
        ):
            pro = backend._conversation_payload(messages, "gpt-5-6-pro", "UTC", "max")
            thinking = backend._conversation_payload(messages, "gpt-5-6-thinking", "UTC")
            work_mode = backend._conversation_payload(messages, "gpt-5-6-sol-wm", "UTC")

        self.assertEqual(pro["thinking_effort"], "standard")
        self.assertEqual(thinking["thinking_effort"], "standard")
        self.assertEqual(work_mode["thinking_effort"], "standard")

    def test_stream_conversation_resumes_after_work_mode_handoff(self) -> None:
        class FakeSSE:
            status_code = 200

            def __init__(self, lines: list[str]) -> None:
                self.lines = lines
                self.closed = False

            def iter_lines(self):
                return iter(self.lines)

            def close(self) -> None:
                self.closed = True

        initial = FakeSSE([
            'data: {"type":"resume_conversation_token","conversation_id":"conv-1","token":"resume-token"}',
            'data: {"type":"stream_handoff","options":[{"type":"resume_sse_endpoint","topic_id":"topic-1"}]}',
            "data: [DONE]",
        ])
        resumed = FakeSSE([
            'data: {"type":"message","message":{"content":{"parts":["OK"]}}}',
            "data: [DONE]",
        ])
        backend = object.__new__(OpenAIBackendAPI)
        backend.base_url = "https://chatgpt.com"
        backend.session = mock.Mock()
        backend.session.post.side_effect = [initial, resumed]
        requirements = ChatRequirements(token="requirements-token")

        with (
            mock.patch.object(backend, "_bootstrap"),
            mock.patch.object(backend, "_get_chat_requirements", return_value=requirements),
            mock.patch.object(backend, "_chat_target", return_value=("/backend-api/conversation", "UTC")),
            mock.patch.object(backend, "_conversation_payload", return_value={"model": "gpt-5-6-sol-wm"}),
            mock.patch.object(backend, "_conversation_headers", return_value={"Authorization": "Bearer token"}),
        ):
            events = list(backend.stream_conversation(model="gpt-5-6-sol-wm", prompt="hello"))

        self.assertEqual(events[-2:], [
            '{"type":"message","message":{"content":{"parts":["OK"]}}}',
            "[DONE]",
        ])
        self.assertEqual(events.count("[DONE]"), 1)
        resume_call = backend.session.post.call_args_list[1]
        self.assertEqual(resume_call.args[0], "https://chatgpt.com/backend-api/f/conversation/resume")
        self.assertEqual(resume_call.kwargs["json"], {"conversation_id": "conv-1", "offset": 0})
        self.assertEqual(resume_call.kwargs["headers"]["X-Conduit-Token"], "resume-token")
        self.assertTrue(initial.closed)
        self.assertTrue(resumed.closed)


if __name__ == "__main__":
    unittest.main()

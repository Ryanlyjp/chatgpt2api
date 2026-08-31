from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from services.account_service import AccountService
from services.model_service import ModelCatalogService
from services.storage.json_storage import JSONStorageBackend


def advertised_model(
    model_id: str,
    *,
    title: str = "",
    reasoning_type: str = "",
    is_work_mode_model: bool = False,
) -> dict:
    return {
        "id": model_id,
        "object": "model",
        "created": 0,
        "owned_by": "chatgpt",
        "permission": [],
        "root": model_id,
        "parent": None,
        "_chatgpt_title": title,
        "_chatgpt_reasoning_type": reasoning_type,
        "_chatgpt_is_work_mode_model": is_work_mode_model,
    }


def model_list(*models: str | dict) -> dict:
    return {
        "object": "list",
        "data": [
            advertised_model(model) if isinstance(model, str) else model
            for model in models
        ],
    }


class FakeBackend:
    def __init__(self, access_token: str, outcomes: dict[str, object], calls: list[str], closed: list[str]) -> None:
        self.access_token = access_token
        self._outcomes = outcomes
        self._calls = calls
        self._closed = closed

    def list_models(self) -> dict:
        self._calls.append(self.access_token)
        outcome = self._outcomes[self.access_token]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def close(self) -> None:
        self._closed.append(self.access_token)


class ModelCatalogServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.accounts = AccountService(
            JSONStorageBackend(Path(self.temp_dir.name) / "accounts.json")
        )
        self.accounts.add_account_items(
            [
                {"access_token": "free-bad", "type": "free", "status": "正常"},
                {"access_token": "free-good", "type": "FREE", "status": "正常"},
                {"access_token": "plus", "type": "free", "status": "正常"},
                {"access_token": "pro", "type": "pro", "status": "正常"},
                {"access_token": "team-disabled", "type": "Team", "status": "禁用"},
            ]
        )
        self.accounts.refresh_access_token = lambda token, **_kwargs: token
        self.now = 1000.0
        self.calls: list[str] = []
        self.closed: list[str] = []
        self.outcomes: dict[str, object] = {
            "": model_list("anon", "shared"),
            "free-bad": RuntimeError("expired"),
            "free-good": model_list("free-only", "shared"),
            "plus": model_list("plus-only", "shared"),
            "pro": model_list("pro-only"),
        }
        self.catalog = ModelCatalogService(
            self.accounts,
            backend_factory=lambda access_token="": FakeBackend(
                access_token, self.outcomes, self.calls, self.closed
            ),
            cache_ttl_seconds=300,
            clock=lambda: self.now,
        )

    def test_catalog_unions_anonymous_and_every_active_account(self) -> None:
        result = self.catalog.list_models()

        self.assertEqual(
            [item["id"] for item in result["data"]],
            ["anon", "free-only", "plus-only", "pro-only", "shared"],
        )
        self.assertCountEqual(self.calls, ["", "free-bad", "free-good", "plus", "pro"])
        self.assertCountEqual(self.closed, self.calls)
        self.assertNotIn("team-disabled", self.calls)

        pro_route = self.catalog.route_for_model("pro-only")
        self.assertEqual(pro_route.access_tokens, frozenset({"pro"}))
        self.assertFalse(pro_route.allow_anonymous)

        shared_route = self.catalog.route_for_model("shared")
        self.assertEqual(shared_route.access_tokens, frozenset({"free-good", "plus"}))
        self.assertTrue(shared_route.allow_anonymous)

    def test_gpt_5_6_names_and_routes_follow_each_accounts_metadata(self) -> None:
        self.outcomes["free-good"] = model_list(
            advertised_model("gpt-5-6", title="GPT-5.6 Luna", reasoning_type="auto"),
            advertised_model("gpt-5-6-mini", title="GPT-5.6 Luna", reasoning_type="none"),
            advertised_model("gpt-5-6-t-mini", title="GPT-5.6 Luna", reasoning_type="reasoning"),
        )
        self.outcomes["plus"] = model_list(
            advertised_model("gpt-5-6", title="GPT-5.6 Sol", reasoning_type="auto"),
            advertised_model("gpt-5-6-instant", title="GPT-5.6 Sol", reasoning_type="none"),
            advertised_model("gpt-5-6-thinking", title="GPT-5.6 Sol", reasoning_type="reasoning"),
            advertised_model("gpt-5-6-pro", title="GPT-5.6 Pro", reasoning_type="pro"),
            advertised_model(
                "gpt-5.6-luna-wm",
                title="GPT-5.6 Luna",
                reasoning_type="reasoning",
                is_work_mode_model=True,
            ),
            advertised_model(
                "gpt-5.6-terra-wm",
                title="GPT-5.6 Terra",
                reasoning_type="reasoning",
                is_work_mode_model=True,
            ),
            advertised_model(
                "gpt-5.6-sol-wm",
                title="GPT-5.6 Sol",
                reasoning_type="reasoning",
                is_work_mode_model=True,
            ),
        )

        result = self.catalog.list_models()
        model_ids = {item["id"] for item in result["data"]}

        self.assertTrue({
            "gpt-5.6-luna",
            "gpt-5.6-sol",
            "gpt-5.6-luna-wm",
            "gpt-5.6-terra-wm",
            "gpt-5.6-sol-wm",
        }.issubset(model_ids))
        self.assertNotIn("gpt-5.6-luna-instant", model_ids)
        self.assertNotIn("gpt-5.6-luna-thinking", model_ids)
        self.assertNotIn("gpt-5.6-sol-instant", model_ids)
        self.assertNotIn("gpt-5.6-sol-thinking", model_ids)
        self.assertNotIn("gpt-5.6-pro", model_ids)
        self.assertNotIn("gpt-5-6", model_ids)
        self.assertNotIn("gpt-5-6-mini", model_ids)
        self.assertEqual(
            self.catalog.route_for_model("gpt-5.6-luna").access_tokens,
            frozenset({"free-good"}),
        )
        self.assertEqual(
            self.catalog.route_for_model("gpt-5.6-sol").access_tokens,
            frozenset({"plus"}),
        )
        self.assertEqual(
            self.catalog.route_for_model("gpt-5.6-sol-thinking").access_tokens,
            frozenset({"plus"}),
        )
        self.assertEqual(
            self.catalog.route_for_model("gpt-5.6-pro").access_tokens,
            frozenset({"plus"}),
        )
        self.assertEqual(
            self.catalog.route_for_model("gpt-5-6").access_tokens,
            frozenset({"free-good", "plus"}),
        )
        self.assertEqual(
            self.catalog.upstream_model_for("gpt-5.6-luna", "free-good"),
            "gpt-5-6",
        )
        self.assertEqual(
            self.catalog.upstream_model_for("gpt-5.6-luna", "free-good", "none"),
            "gpt-5-6-mini",
        )
        self.assertEqual(
            self.catalog.upstream_model_for("gpt-5.6-luna", "free-good", "high"),
            "gpt-5-6-t-mini",
        )
        self.assertEqual(
            self.catalog.upstream_model_for("gpt-5.6-sol", "plus", "none"),
            "gpt-5-6-instant",
        )
        self.assertEqual(
            self.catalog.upstream_model_for("gpt-5.6-sol", "plus", "high"),
            "gpt-5-6-thinking",
        )
        self.assertEqual(
            self.catalog.upstream_model_for("gpt-5.6-sol", "plus", "medium", "pro"),
            "gpt-5-6-pro",
        )
        self.assertEqual(
            self.catalog.upstream_model_for("gpt-5.6-sol-wm", "plus", "high"),
            "gpt-5.6-sol-wm",
        )
        self.assertEqual(
            self.catalog.upstream_model_for("gpt-5.6-sol-thinking", "plus"),
            "gpt-5-6-thinking",
        )
        self.assertEqual(
            self.catalog.upstream_model_for("gpt-5-6", "plus"),
            "gpt-5-6",
        )

    def test_catalog_is_cached_until_ttl_expires(self) -> None:
        self.catalog.list_models()
        self.catalog.list_models()
        self.catalog.route_for_model("pro-only")

        self.assertEqual(self.calls.count("pro"), 1)
        self.assertEqual(self.calls.count(""), 1)

    def test_concurrent_readers_share_one_catalog_refresh(self) -> None:
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda _index: self.catalog.list_models(), range(8)))

        self.assertTrue(all(result == results[0] for result in results))
        self.assertEqual(self.calls.count(""), 1)
        self.assertEqual(self.calls.count("free-good"), 1)
        self.assertEqual(self.calls.count("plus"), 1)
        self.assertEqual(self.calls.count("pro"), 1)

    def test_failed_refresh_keeps_last_successful_models_for_that_type(self) -> None:
        self.catalog.list_models()
        self.outcomes["pro"] = RuntimeError("temporary upstream failure")
        self.now += 301

        result = self.catalog.list_models()

        self.assertIn("pro-only", {item["id"] for item in result["data"]})
        self.assertEqual(
            self.catalog.route_for_model("pro-only").access_tokens,
            frozenset({"pro"}),
        )
        self.assertEqual(self.calls.count("pro"), 2)

    def test_removed_account_type_drops_its_stale_capabilities(self) -> None:
        self.catalog.list_models()
        self.accounts.delete_accounts(["pro"])

        result = self.catalog.list_models()

        self.assertNotIn("pro-only", {item["id"] for item in result["data"]})
        self.assertEqual(
            self.catalog.route_for_model("pro-only").access_tokens,
            frozenset(),
        )


if __name__ == "__main__":
    unittest.main()

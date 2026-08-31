from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import RLock
from typing import Any

from services.account_service import AccountService, account_service
from services.openai_backend_api import OpenAIBackendAPI
from utils.log import logger


@dataclass(frozen=True)
class ModelRoute:
    access_tokens: frozenset[str]
    allow_anonymous: bool = False


@dataclass
class AccountModelCatalog:
    models: dict[str, dict[str, Any]]
    upstream_by_model: dict[str, str]


class ModelUnavailableError(RuntimeError):
    pass


class ModelCatalogService:
    """Caches the model catalog advertised to each active account."""

    def __init__(
        self,
        accounts: AccountService,
        *,
        backend_factory: Callable[..., Any] = OpenAIBackendAPI,
        cache_ttl_seconds: float = 300,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._accounts = accounts
        self._backend_factory = backend_factory
        self._cache_ttl_seconds = max(1.0, float(cache_ttl_seconds))
        self._clock = clock
        self._lock = RLock()
        self._expires_at = 0.0
        self._account_signature: tuple[str, ...] = ()
        self._anonymous_catalog = AccountModelCatalog({}, {})
        self._catalogs_by_access_token: dict[str, AccountModelCatalog] = {}

    @staticmethod
    def _public_model_id(item: dict[str, Any], upstream_model: str) -> str:
        title = str(item.get("_chatgpt_title") or "").strip().lower()
        family_model = {
            "gpt-5.6 luna": "gpt-5.6-luna",
            "gpt-5.6 terra": "gpt-5.6-terra",
            "gpt-5.6 sol": "gpt-5.6-sol",
            "gpt-5.6 pro": "gpt-5.6-sol",
        }.get(title)
        if not family_model:
            return upstream_model
        if bool(item.get("_chatgpt_is_work_mode_model")):
            return f"{family_model}-wm"
        return family_model

    @staticmethod
    def _variant_model_id(item: dict[str, Any], public_model: str) -> str:
        if bool(item.get("_chatgpt_is_work_mode_model")):
            return public_model
        reasoning_type = str(item.get("_chatgpt_reasoning_type") or "").strip().lower()
        if reasoning_type == "none":
            return f"{public_model}-instant"
        if reasoning_type == "reasoning":
            return f"{public_model}-thinking"
        if reasoning_type == "pro":
            return "gpt-5.6-pro" if public_model == "gpt-5.6-sol" else f"{public_model}-pro"
        return public_model

    @staticmethod
    def routing_model_for(model: str, thinking_effort: str = "", reasoning_mode: str = "") -> str:
        model = str(model or "").strip()
        if model not in {"gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"}:
            return model
        if str(reasoning_mode or "").strip().lower() == "pro":
            return "gpt-5.6-pro" if model == "gpt-5.6-sol" else f"{model}-pro"
        effort = str(thinking_effort or "").strip().lower()
        if effort == "none":
            return f"{model}-instant"
        if effort:
            return f"{model}-thinking"
        return model

    @classmethod
    def _model_catalog(cls, result: object) -> AccountModelCatalog:
        if not isinstance(result, dict) or not isinstance(result.get("data"), list):
            raise TypeError("upstream model response has no data list")
        models: dict[str, dict[str, Any]] = {}
        upstream_by_model: dict[str, str] = {}
        for item in result["data"]:
            if not isinstance(item, dict):
                continue
            upstream_model = str(item.get("id") or "").strip()
            if not upstream_model:
                continue
            public_model = cls._public_model_id(item, upstream_model)
            variant_model = cls._variant_model_id(item, public_model)
            upstream_by_model.setdefault(upstream_model, upstream_model)
            upstream_by_model.setdefault(variant_model, upstream_model)
            if variant_model != public_model or public_model in models:
                continue
            public_item = {
                key: value
                for key, value in item.items()
                if not str(key).startswith("_chatgpt_")
            }
            public_item["id"] = public_model
            public_item["root"] = public_model
            models[public_model] = public_item
        return AccountModelCatalog(models, upstream_by_model)

    def _active_access_tokens(self) -> list[str]:
        tokens: list[str] = []
        for account in self._accounts.list_accounts():
            if not isinstance(account, dict) or account.get("status") in {"禁用", "异常"}:
                continue
            access_token = str(account.get("access_token") or "").strip()
            if access_token and access_token not in tokens:
                tokens.append(access_token)
        return tokens

    @staticmethod
    def _signature(access_tokens: list[str]) -> tuple[str, ...]:
        return tuple(sorted(access_tokens))

    def _fetch_models(self, access_token: str = "") -> AccountModelCatalog:
        backend = self._backend_factory(access_token=access_token)
        try:
            return self._model_catalog(backend.list_models())
        finally:
            backend.close()

    def _fetch_account_models(
        self, access_token: str
    ) -> tuple[str, AccountModelCatalog | None]:
        resolved_token = access_token
        try:
            resolved_token = self._accounts.refresh_access_token(
                access_token,
                event="list_models",
            ) or access_token
            return resolved_token, self._fetch_models(resolved_token)
        except Exception as exc:  # noqa: BLE001 - retain this account's cached catalog
            logger.warning({
                "event": "model_catalog_account_failed",
                "error_type": type(exc).__name__,
            })
            return resolved_token, None

    def _refresh(self, access_tokens: list[str], signature: tuple[str, ...]) -> None:
        catalogs_by_access_token: dict[str, AccountModelCatalog] = {}
        with ThreadPoolExecutor(max_workers=min(4, len(access_tokens) + 1)) as executor:
            anonymous_future = executor.submit(self._fetch_models)
            account_futures = {
                access_token: executor.submit(self._fetch_account_models, access_token)
                for access_token in access_tokens
            }
            try:
                anonymous_catalog = anonymous_future.result()
            except Exception as exc:  # noqa: BLE001 - retain cached models on upstream failure
                logger.warning({
                    "event": "model_catalog_anonymous_failed",
                    "error_type": type(exc).__name__,
                })
                anonymous_catalog = self._anonymous_catalog

            for access_token, future in account_futures.items():
                resolved_token, catalog = future.result()
                if catalog is not None:
                    catalogs_by_access_token[resolved_token] = catalog
                elif access_token in self._catalogs_by_access_token:
                    catalogs_by_access_token[resolved_token] = self._catalogs_by_access_token[access_token]

        self._anonymous_catalog = anonymous_catalog
        self._catalogs_by_access_token = catalogs_by_access_token
        self._account_signature = signature
        self._expires_at = self._clock() + self._cache_ttl_seconds

    def _ensure_catalog(self) -> None:
        access_tokens = self._active_access_tokens()
        signature = self._signature(access_tokens)
        with self._lock:
            if signature == self._account_signature and self._clock() < self._expires_at:
                return
            self._refresh(access_tokens, signature)

    def list_models(self) -> dict[str, Any]:
        self._ensure_catalog()
        with self._lock:
            union: dict[str, dict[str, Any]] = {
                model_id: dict(item)
                for model_id, item in self._anonymous_catalog.models.items()
            }
            for access_token in sorted(self._catalogs_by_access_token):
                for model_id, item in self._catalogs_by_access_token[access_token].models.items():
                    union.setdefault(model_id, dict(item))
        return {
            "object": "list",
            "data": [union[model_id] for model_id in sorted(union)],
        }

    def route_for_model(self, model: str) -> ModelRoute:
        model = str(model or "").strip()
        self._ensure_catalog()
        with self._lock:
            access_tokens = frozenset(
                access_token
                for access_token, catalog in self._catalogs_by_access_token.items()
                if model in catalog.upstream_by_model
            )
            return ModelRoute(
                access_tokens=access_tokens,
                allow_anonymous=model in self._anonymous_catalog.upstream_by_model,
            )

    def upstream_model_for(
        self,
        model: str,
        access_token: str = "",
        thinking_effort: str = "",
        reasoning_mode: str = "",
    ) -> str:
        model = str(model or "").strip()
        if not model or model == "auto":
            return model or "auto"
        routing_model = self.routing_model_for(model, thinking_effort, reasoning_mode)
        self._ensure_catalog()
        with self._lock:
            if access_token:
                resolved_token = self._accounts.resolve_access_token(access_token)
                catalog = self._catalogs_by_access_token.get(resolved_token)
            else:
                catalog = self._anonymous_catalog
            if catalog is None:
                return model
            return catalog.upstream_by_model.get(
                routing_model,
                catalog.upstream_by_model.get(model, model),
            )


model_catalog_service = ModelCatalogService(account_service)

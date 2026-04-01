"""litellm custom provider for slullama.

Registers ``slullama/`` as a model prefix so that::

    import slullama  # auto-registers
    import litellm

    resp = litellm.completion(model="slullama/qwen3.5:9b", messages=[...])

works seamlessly — the SSH tunnel and Slurm backend are managed
automatically under the hood.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

log = logging.getLogger("slullama.litellm")

_registered = False


def register() -> None:
    """Register slullama as a litellm custom provider (idempotent)."""
    global _registered
    if _registered:
        return

    try:
        import litellm
        from litellm import CustomLLM, ModelResponse
        from litellm.types.utils import (
            Choices,
            Delta,
            Message,
            StreamingChoices,
            Usage,
        )
    except ImportError:
        log.debug("litellm not installed, skipping provider registration")
        return

    class SlulamaLLM(CustomLLM):
        """litellm custom handler that routes through slullama."""

        def __init__(self) -> None:
            self._client: Any = None

        def _get_client(self) -> Any:
            if self._client is None:
                from slullama.client.client import SlulamaClient

                self._client = SlulamaClient.get_default()
            return self._client

        def completion(
            self,
            model: str,
            messages: list[dict[str, Any]],
            api_base: str | None = None,
            custom_prompt_dict: dict | None = None,
            model_response: ModelResponse | None = None,
            print_verbose: Any = None,
            encoding: Any = None,
            logging_obj: Any = None,
            optional_params: dict | None = None,
            litellm_params: dict | None = None,
            logger_fn: Any = None,
            headers: dict | None = None,
            timeout: float | None = None,
            acompletion: bool = False,
            **kwargs: Any,
        ) -> ModelResponse:
            """Synchronous completion via slullama."""

            client = self._get_client()
            client._ensure_connected()

            # Extract model name after "slullama/"
            model_name = model.split("/", 1)[-1] if "/" in model else model

            # Build ollama-compatible request
            import httpx

            url = f"{client.ollama_url}/api/chat"
            payload = {
                "model": model_name,
                "messages": messages,
                "stream": False,
            }
            if optional_params:
                if "max_tokens" in optional_params:
                    payload["options"] = payload.get("options", {})
                    payload["options"]["num_predict"] = optional_params["max_tokens"]
                if "temperature" in optional_params:
                    payload["options"] = payload.get("options", {})
                    payload["options"]["temperature"] = optional_params["temperature"]

            resp = httpx.post(
                url,
                json=payload,
                headers=client._auth_headers(),
                timeout=httpx.Timeout(600.0),
            )
            resp.raise_for_status()
            data = resp.json()

            msg = data.get("message", {})
            content = msg.get("content", "")

            response = model_response or ModelResponse()
            response.choices = [
                Choices(
                    message=Message(role="assistant", content=content),
                    index=0,
                    finish_reason="stop",
                )
            ]
            response.model = model_name
            response.usage = Usage(
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                total_tokens=(
                    data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                ),
            )
            return response

        def streaming(
            self,
            model: str,
            messages: list[dict[str, Any]],
            api_base: str | None = None,
            custom_prompt_dict: dict | None = None,
            model_response: ModelResponse | None = None,
            print_verbose: Any = None,
            encoding: Any = None,
            logging_obj: Any = None,
            optional_params: dict | None = None,
            litellm_params: dict | None = None,
            logger_fn: Any = None,
            headers: dict | None = None,
            timeout: float | None = None,
            **kwargs: Any,
        ) -> Iterator[ModelResponse]:
            """Synchronous streaming via slullama."""
            client = self._get_client()
            client._ensure_connected()

            model_name = model.split("/", 1)[-1] if "/" in model else model

            import httpx

            url = f"{client.ollama_url}/api/chat"
            payload = {
                "model": model_name,
                "messages": messages,
                "stream": True,
            }
            if optional_params:
                if "max_tokens" in optional_params:
                    payload["options"] = payload.get("options", {})
                    payload["options"]["num_predict"] = optional_params["max_tokens"]
                if "temperature" in optional_params:
                    payload["options"] = payload.get("options", {})
                    payload["options"]["temperature"] = optional_params["temperature"]

            with httpx.stream(
                "POST",
                url,
                json=payload,
                headers=client._auth_headers(),
                timeout=httpx.Timeout(600.0),
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    msg = chunk.get("message", {})
                    content = msg.get("content", "")
                    done = chunk.get("done", False)

                    delta_resp = ModelResponse(
                        stream=True,
                        model=model_name,
                    )
                    delta_resp.choices = [
                        StreamingChoices(
                            delta=Delta(content=content, role="assistant"),
                            index=0,
                            finish_reason="stop" if done else None,
                        )
                    ]
                    if done:
                        delta_resp.usage = Usage(
                            prompt_tokens=chunk.get("prompt_eval_count", 0),
                            completion_tokens=chunk.get("eval_count", 0),
                            total_tokens=(
                                chunk.get("prompt_eval_count", 0)
                                + chunk.get("eval_count", 0)
                            ),
                        )
                    yield delta_resp

    handler = SlulamaLLM()
    if (
        not hasattr(litellm, "custom_provider_map")
        or litellm.custom_provider_map is None
    ):
        litellm.custom_provider_map = []

    # Don't double-register
    existing = {
        e.get("provider") for e in litellm.custom_provider_map if isinstance(e, dict)
    }
    if "slullama" not in existing:
        litellm.custom_provider_map.append(
            {"provider": "slullama", "custom_handler": handler}
        )
        log.debug("Registered slullama litellm provider")

    _registered = True

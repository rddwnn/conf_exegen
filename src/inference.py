"""Inference transport and operator-supplied remote provenance, shared by all methods."""

import json
import re
import socket
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from common import read_json, resolve


def inference_mode(cfg):
    mode = cfg.get("inference", {}).get("mode", "local")
    if mode not in ("local", "remote"):
        raise ValueError("inference.mode must be local or remote")
    return mode


def api_url(cfg, suffix):
    base = cfg["model"]["base_url"].rstrip("/")
    parsed = urlparse(base)
    if (
        parsed.hostname not in ("localhost", "127.0.0.1", "::1")
        or parsed.scheme != "http"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path != "/v1"
    ):
        raise ValueError(
            "Use a loopback HTTP /v1 endpoint; remote GPUs must be reached through SSH forwarding"
        )
    return base.removesuffix("/v1") + suffix


def api_get(cfg, suffix, optional=False):
    url = api_url(cfg, suffix)
    timeout = min(30, cfg["model"]["timeout_seconds"])
    try:
        with urlopen(
            Request(url, headers={"Authorization": "Bearer local"}), timeout=timeout
        ) as response:
            return json.load(response)
    except HTTPError as exc:
        if optional and exc.code in (404, 405):
            return None
        if exc.code in (401, 403):
            raise ValueError(
                f"Inference authentication failure at {suffix} (HTTP {exc.code}); check server access settings"
            ) from exc
        if exc.code == 503:
            raise ValueError(
                f"Inference server is not ready at {suffix} (HTTP 503); wait for model loading"
            ) from exc
        raise ValueError(f"Inference HTTP failure at {suffix}: {exc.code}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise ValueError(
            f"Inference timeout at {suffix}; check the SSH tunnel and remote GPU server"
        ) from exc
    except URLError as exc:
        category = (
            "timeout"
            if isinstance(exc.reason, (TimeoutError, socket.timeout))
            else "connectivity failure"
        )
        raise ValueError(
            f"Inference {category} at {suffix}; check the SSH tunnel and server: {exc.reason}"
        ) from exc
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid inference response at {suffix}: expected JSON") from exc


def generation_settings(cfg):
    model = cfg["model"]
    result = {key: model[key] for key in ("max_tokens", "temperature", "top_p")}
    for key in ("top_k", "min_p"):
        if key in model:
            result[key] = model[key]
    result["chat_template_kwargs"] = model.get("chat_template_kwargs", {})
    return result


def request_extensions(cfg):
    model = cfg["model"]
    result = {key: model[key] for key in ("top_k", "min_p") if key in model}
    if model.get("chat_template_kwargs"):
        result["chat_template_kwargs"] = model["chat_template_kwargs"]
    return result


def validate_endpoint(cfg):
    health = api_get(cfg, "/health")
    if not isinstance(health, dict) or health.get("status") != "ok":
        raise ValueError("Inference server is not ready: /health must report status=ok")
    models = api_get(cfg, "/v1/models")
    if not isinstance(models, dict) or not isinstance(models.get("data"), list):
        raise ValueError("Invalid model-list response: expected a data list")
    if cfg["model"]["name"] not in {
        item.get("id") for item in models["data"] if isinstance(item, dict)
    }:
        raise ValueError(f"Inference model mismatch: alias {cfg['model']['name']!r} is not served")
    properties = api_get(cfg, "/props", optional=True)
    if properties is not None and not isinstance(properties, dict):
        raise ValueError("Invalid /props response: expected an object")
    if properties is not None:
        defaults = properties.get("default_generation_settings", {})
        if not isinstance(defaults, dict):
            raise ValueError("Invalid /props default_generation_settings")
        context = defaults.get("n_ctx", properties.get("n_ctx_per_seq"))
        if context is not None and context != cfg["model"]["context_size"]:
            raise ValueError(
                f"Inference context mismatch: configured {cfg['model']['context_size']}, server reports {context}"
            )
        kwargs = properties.get("chat_template_kwargs", defaults.get("chat_template_kwargs"))
        if isinstance(kwargs, dict) and "enable_thinking" in kwargs:
            expected = cfg["model"].get("chat_template_kwargs", {}).get("enable_thinking")
            if expected is not None and kwargs["enable_thinking"] != expected:
                raise ValueError("Inference thinking-mode mismatch in /props")
    return health, properties


def remote_server(cfg):
    """Validate a receipt; do not imply that API responses prove the remote file hash."""
    path = cfg.get("inference", {}).get("receipt")
    if not path:
        raise ValueError("Remote inference requires inference.receipt from the GPU server")
    receipt = read_json(resolve(path))
    try:
        if receipt["schema_version"] != 1:
            raise ValueError("Unsupported remote receipt schema")
        model, server, generation = receipt["model"], receipt["server"], receipt["generation"]
        parameters = server["runtime_parameters"]
        if not isinstance(model["name"], str) or not model["name"].strip():
            raise ValueError("Remote receipt must name the exact model")
        if model["alias"] != cfg["model"]["name"]:
            raise ValueError("Remote receipt model alias mismatch")
        if not re.fullmatch(r"[a-fA-F0-9]{64}", model["gguf_sha256"]):
            raise ValueError(
                "Remote receipt requires a real GGUF SHA-256, computed on the GPU host"
            )
        if (
            not isinstance(model["quantization"], str)
            or not model["quantization"].strip()
            or not isinstance(server["version"], str)
            or not server["version"].strip()
            or not re.fullmatch(r"[a-fA-F0-9]{7,40}", server["commit"])
        ):
            raise ValueError("Remote receipt requires quantization, server version, and commit")
        binary_hash = server.get("binary_sha256")
        if binary_hash is not None and not re.fullmatch(r"[a-fA-F0-9]{64}", binary_hash):
            raise ValueError("Remote binary SHA-256 must be a real hash or null")
        if parameters["host"] != "127.0.0.1":
            raise ValueError("Remote receipt must describe a server bound to 127.0.0.1")
        if parameters["context_size"] != cfg["model"]["context_size"]:
            raise ValueError("Remote receipt context length mismatch")
        if parameters.get("parallel") != 1:
            raise ValueError("Remote reproduction requires one server slot")
        actual_kwargs = parameters.get("chat_template_kwargs", {})
        expected_kwargs = cfg["model"].get("chat_template_kwargs")
        if expected_kwargs is not None and actual_kwargs != expected_kwargs:
            raise ValueError("Remote receipt chat-template settings mismatch")
        if "qwen3" in model["name"].lower() and actual_kwargs.get("enable_thinking") is not False:
            raise ValueError(
                "Qwen3 reproduction requires enable_thinking=false in the remote receipt"
            )
        for key, value in generation_settings(cfg).items():
            if key != "chat_template_kwargs" and generation.get(key) != value:
                raise ValueError(f"Remote receipt generation setting mismatch: {key}")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"Malformed remote inference receipt: {exc}") from exc
    return {
        "mode": "remote",
        "pid": 0,
        "model_sha256": model["gguf_sha256"].lower(),
        "binary_sha256": binary_hash.lower() if binary_hash else None,
        "version": server["version"],
        "commit": server["commit"],
        "model_name": model["name"],
        "alias": model["alias"],
        "quantization": model["quantization"],
        "runtime_parameters": parameters,
        "generation": generation,
        "receipt": receipt,
        "verification": "Operator-supplied remote receipt. Hashes are not recomputed locally; alias, available metadata and chat are checked through the tunnel. This is not cryptographic attestation of the live server.",
    }


def server_identity(server):
    """Stable scientific identity excludes remote PID, receipt time, and local receipt path."""
    keys = (
        "mode",
        "model_sha256",
        "binary_sha256",
        "version",
        "commit",
        "model_name",
        "alias",
        "quantization",
        "runtime_parameters",
        "generation",
    )
    return {key: server[key] for key in keys if key in server}


def validate_remote_properties(server, properties):
    if properties is None:
        return
    reported = properties.get("build_info")
    if isinstance(reported, str):
        hashes = re.findall(r"\b[a-fA-F0-9]{7,40}\b", reported)
        if hashes and not any(
            server["commit"].lower().startswith(h.lower())
            or h.lower().startswith(server["commit"].lower())
            for h in hashes
        ):
            raise ValueError("Remote server commit mismatch between receipt and /props build_info")
    defaults = properties.get("default_generation_settings", {})
    actual = properties.get("chat_template_kwargs", defaults.get("chat_template_kwargs"))
    expected = server["runtime_parameters"].get("chat_template_kwargs", {})
    if isinstance(actual, dict) and any(actual.get(k) != v for k, v in expected.items()):
        raise ValueError("Remote receipt thinking/template settings differ from live /props")


def smoke_check(cfg, non_thinking=False):
    from openai import (
        APIConnectionError,
        APIResponseValidationError,
        APIStatusError,
        APITimeoutError,
        AuthenticationError,
        OpenAI,
        PermissionDeniedError,
    )

    client = OpenAI(
        base_url=cfg["model"]["base_url"],
        api_key="local",
        max_retries=0,
        timeout=cfg["model"]["timeout_seconds"],
    )
    kwargs = request_extensions(cfg)
    try:
        with client:
            response = client.chat.completions.create(
                model=cfg["model"]["name"],
                messages=[{"role": "user", "content": "Reply with the word OK."}],
                max_tokens=128,
                temperature=0,
                seed=42,
                **({"extra_body": kwargs} if kwargs else {}),
            )
    except (AuthenticationError, PermissionDeniedError) as exc:
        raise ValueError("Inference authentication failure during chat smoke test") from exc
    except APITimeoutError as exc:
        raise ValueError(
            "Inference timeout during chat smoke test; check tunnel and remote model load"
        ) from exc
    except APIConnectionError as exc:
        raise ValueError(
            "Inference connectivity failure during chat smoke test; check SSH forwarding"
        ) from exc
    except APIStatusError as exc:
        raise ValueError(
            f"Inference chat-completions failure (HTTP {exc.status_code}); check model and chat-template support"
        ) from exc
    except APIResponseValidationError as exc:
        raise ValueError("Invalid chat-completions response schema") from exc
    if not hasattr(response, "model_dump"):
        raise ValueError("Invalid chat-completions response: expected a completion object")
    payload = response.model_dump(mode="json")
    if not isinstance(payload, dict):
        raise ValueError("Invalid chat-completions response: expected an object")
    choices = payload.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise ValueError("Invalid chat-completions response: no choices")
    choice = choices[0]
    message = choice.get("message") or {}
    if not isinstance(message, dict):
        raise ValueError("Invalid chat-completions response: expected a message object")
    content, reasoning = (
        message.get("content"),
        message.get("reasoning_content") or message.get("reasoning"),
    )
    if (
        not isinstance(content, str)
        or not content.strip()
        or choice.get("finish_reason") == "length"
    ):
        raise ValueError(
            f"Invalid smoke response: nonempty finished message.content required; finish_reason={choice.get('finish_reason')}, reasoning_chars={len(str(reasoning or ''))}. Qwen3 must use --chat-template-kwargs '{{\"enable_thinking\":false}}'; verify supported flags and increase output budget if needed."
        )
    if (
        non_thinking or cfg["model"].get("chat_template_kwargs", {}).get("enable_thinking") is False
    ) and (reasoning or "<think>" in content):
        raise ValueError("Thinking tokens appeared in a configured non-thinking chat response")
    return payload

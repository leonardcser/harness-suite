"""Error responses in the shape and wording of the OpenAI API."""

from ..messages import Error, Request


def seconds(err: Error) -> str | None:
    """The wait as a count of seconds, when `retry-after` holds one."""
    return err.retry_after if err.retry_after and err.retry_after.isdigit() else None


def headers(err: Error) -> dict:
    values = {"x-request-id": "req_suite"}
    if err.retry_after is not None:
        values["retry-after"] = err.retry_after
    if seconds(err):
        values["x-ratelimit-reset-requests"] = f"{err.retry_after}s"
    if err.retry_after_ms is not None:
        values["retry-after-ms"] = err.retry_after_ms
    if err.kind == "rate":
        values["x-ratelimit-limit-requests"] = "60"
        values["x-ratelimit-remaining-requests"] = "0"
    return values


def response(err: Error, request: Request, window: int, param: str):
    if err.body is not None:
        return 400, headers(err), err.body
    wait = f" Please try again in {err.retry_after}s." if seconds(err) else ""
    status, kind, code, message = {
        "rate": (
            429,
            "requests",
            "rate_limit_exceeded",
            f"Rate limit reached for {request.model} in organization org-suite on"
            f" requests per min (RPM): Limit 60, Used 60, Requested 1.{wait} Visit"
            " https://platform.openai.com/account/rate-limits to learn more.",
        ),
        "server": (
            500,
            "server_error",
            None,
            "The server had an error while processing your request. Sorry about that!",
        ),
        "overloaded": (
            503,
            "server_error",
            "server_is_overloaded",
            "The server is overloaded or not ready yet.",
        ),
        "quota": (
            429,
            "insufficient_quota",
            "insufficient_quota",
            "You exceeded your current quota, please check your plan and billing"
            " details. For more information on this error, read the docs:"
            " https://platform.openai.com/docs/guides/error-codes/api-errors.",
        ),
        "auth": (
            401,
            "invalid_request_error",
            "invalid_api_key",
            "Incorrect API key provided: suite-pl***der. You can find your API key"
            " at https://platform.openai.com/account/api-keys.",
        ),
        "context": (
            400,
            "invalid_request_error",
            "context_length_exceeded",
            f"This model's maximum context length is {window} tokens. However, your"
            f" messages resulted in {request.tokens} tokens. Please reduce the length"
            " of the messages."
            if param == "messages"
            else "Your input exceeds the context window of this model. Please adjust your input and try again.",
        ),
        "invalid": (400, "invalid_request_error", None, "Invalid request."),
        "model": (
            404,
            "invalid_request_error",
            "model_not_found",
            f"The model `{request.model}` does not exist or you do not have access to it.",
        ),
        "too-large": (413, "invalid_request_error", None, "Request too large."),
    }[err.kind]
    body = {
        "error": {
            "message": err.message or message,
            "type": kind,
            "param": param if err.kind == "context" else None,
            "code": code,
        }
    }
    return status, headers(err), body

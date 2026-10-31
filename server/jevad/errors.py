from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger("jevad")

STATUS_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    409: "conflict_error",
    422: "invalid_request_error",
    429: "rate_limit_error",
    500: "api_error",
    503: "service_unavailable",
    504: "timeout_error",
}


def error_response(status: int, message: str, code: str | None = None) -> JSONResponse:
    body = {
        "error": {
            "message": message,
            "type": STATUS_TYPES.get(status, "api_error"),
            "param": None,
            "code": code,
        }
    }
    return JSONResponse(status_code=status, content=body)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def on_validation(request: Request, exc: RequestValidationError):
        details = "; ".join(
            f"{'.'.join(str(part) for part in error.get('loc', ()))}: {error.get('msg')}"
            for error in exc.errors()
        )
        return error_response(400, details or "Invalid request.", "invalid_request")

    @app.exception_handler(HTTPException)
    async def on_http_error(request: Request, exc: HTTPException):
        return error_response(exc.status_code, str(exc.detail))

    @app.exception_handler(Exception)
    async def on_unhandled(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s", request.url.path)
        return error_response(500, "Internal server error.")

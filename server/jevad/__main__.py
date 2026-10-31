from __future__ import annotations

import uvicorn

from .config import Settings


def main() -> None:
    settings = Settings.from_env()
    uvicorn.run("jevad.app:app", host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()

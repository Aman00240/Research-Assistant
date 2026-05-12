FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --group backend --no-install-project --no-dev

COPY backend/scripts/download_model.py ./scripts/download_model.py
RUN /app/.venv/bin/python scripts/download_model.py

COPY backend/ ./backend/

EXPOSE 8000

CMD ["/app/.venv/bin/python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

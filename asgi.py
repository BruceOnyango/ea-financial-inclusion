"""ASGI entry point for the FastAPI service (uvicorn / gunicorn UvicornWorker)."""

from api.main import create_api

app = create_api()

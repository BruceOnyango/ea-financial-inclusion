"""Production WSGI entry point for gunicorn.

Exposes the Flask server under the Dash app so gunicorn can serve it:
    gunicorn wsgi:server
Kept at project root and out of the tested packages on purpose.
"""

from app.app import create_app

server = create_app().server

"""
Request Guard Middleware

Keeps other websites from using the app through the user's own browser, which
the LAN IP filter cannot see (the browser is on the LAN):

- The Host header must name this machine: an IP address, localhost, the
  machine's own name or one listed in ALLOWED_HOSTS. A website that points its
  domain at a LAN address (DNS rebinding) sends its own domain and is refused.
- Requests that change something (POST, PUT, PATCH, DELETE) sent by a browser
  must come from the app's own pages or from an origin CORS allows. A page on
  another site can post plain-text JSON without any CORS preflight, and
  FastAPI would parse it. Clients that are not browsers (the CLI, scripts)
  send no Origin and are not affected.
"""

import ipaddress
import re
import socket
from urllib.parse import urlsplit

from starlette.responses import JSONResponse

from config import ALLOWED_HOSTS, CORS_ALLOWED_ORIGINS, CORS_ORIGIN_REGEX
from i18n.i18n import translate as t

UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def _machine_names() -> set:
    """Names this machine answers to on the LAN."""
    hostname = socket.gethostname().lower()
    return {hostname, f"{hostname}.local", f"{hostname}.lan", socket.getfqdn().lower()}


def _host_name(host: str) -> str:
    """Host header without its port, IPv6 brackets included ("[::1]:8057" -> "::1")."""
    host = host.strip().lower()
    if host.startswith('['):
        return host[1:host.index(']')] if ']' in host else host
    return host.rsplit(':', 1)[0] if host.count(':') == 1 else host


class RequestGuardMiddleware:
    """Refuses requests for other hosts and cross-site requests that change data."""

    def __init__(self, app):
        self.app = app
        self.allowed_names = {"localhost"} | _machine_names() | set(ALLOWED_HOSTS)
        self.allowed_origins = set(CORS_ALLOWED_ORIGINS)
        self.origin_pattern = re.compile(CORS_ORIGIN_REGEX)

    def _host_allowed(self, host: str) -> bool:
        name = _host_name(host)
        if name in self.allowed_names or name.endswith('.localhost'):
            return True
        try:
            # An IP address cannot be rebound to another machine
            ipaddress.ip_address(name)
            return True
        except ValueError:
            return False

    def _origin_allowed(self, origin: str, host: str) -> bool:
        if origin in self.allowed_origins or self.origin_pattern.match(origin):
            return True
        parts = urlsplit(origin)
        return parts.scheme in ('http', 'https') and parts.netloc.lower() == host.strip().lower()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {key.decode('latin-1'): value.decode('latin-1') for key, value in scope["headers"]}
        host = headers.get('host', '')
        if not self._host_allowed(host):
            response = JSONResponse(status_code=400, content={"detail": t('api.errors.host_not_allowed', host=host)})
            await response(scope, receive, send)
            return

        if scope["method"] in UNSAFE_METHODS:
            origin = headers.get('origin')
            # Without Origin, a browser still says where the request comes from
            cross_site = (
                not self._origin_allowed(origin, host) if origin is not None
                else headers.get('sec-fetch-site') == 'cross-site'
            )
            if cross_site:
                response = JSONResponse(status_code=403, content={"detail": t('api.errors.cross_site_request')})
                await response(scope, receive, send)
                return

        await self.app(scope, receive, send)

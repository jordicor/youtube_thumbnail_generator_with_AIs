"""
LAN IP Filtering Middleware

Restricts API access to private network IP ranges (RFC 1918).
Protects against accidental internet exposure (DMZ, port forwarding, etc.)
while still allowing LAN access from other devices.
"""

import ipaddress
import logging
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from config import ALLOWED_IP_RANGES, ENABLE_IP_FILTERING
from i18n.i18n import translate as t

logger = logging.getLogger(__name__)


class LANOnlyMiddleware(BaseHTTPMiddleware):
    """
    Middleware that restricts access to private/LAN IP addresses only.

    Configured via config.py:
    - ALLOWED_IP_RANGES: List of CIDR ranges (default: RFC 1918 private ranges)
    - ENABLE_IP_FILTERING: Toggle filtering on/off (default: True)
    """

    def __init__(self, app):
        super().__init__(app)
        self.allowed_networks = []
        self._parse_allowed_ranges()

    def _parse_allowed_ranges(self):
        """Parse CIDR strings into ip_network objects."""
        for cidr in ALLOWED_IP_RANGES:
            try:
                network = ipaddress.ip_network(cidr, strict=False)
                self.allowed_networks.append(network)
            except ValueError as e:
                logger.warning(f"Invalid CIDR in ALLOWED_IP_RANGES: {cidr} - {e}")

    def _is_ip_allowed(self, ip_string: str) -> bool:
        """Check if an IP address is within allowed ranges."""
        try:
            # Handle IPv4-mapped IPv6 addresses (::ffff:192.168.1.1)
            if ip_string.startswith('::ffff:'):
                ip_string = ip_string[7:]

            ip = ipaddress.ip_address(ip_string)
            return any(ip in network for network in self.allowed_networks)
        except ValueError:
            # Invalid IP format
            return False

    async def dispatch(self, request: Request, call_next):
        """Process request and check IP against allowlist."""

        # Skip filtering if disabled
        if not ENABLE_IP_FILTERING:
            return await call_next(request)

        # Get client IP
        client_ip = request.client.host if request.client else None

        if not client_ip:
            logger.warning("[SECURITY] Request with no client IP - blocked")
            return JSONResponse(
                status_code=403,
                content={"detail": t('api.errors.access_denied_no_ip')}
            )

        # Check if IP is allowed
        if not self._is_ip_allowed(client_ip):
            logger.warning(f"[SECURITY] Blocked request from non-LAN IP: {client_ip}")
            return JSONResponse(
                status_code=403,
                content={
                    "detail": t('api.errors.access_restricted_local'),
                    "hint": "This application only accepts connections from private IP ranges (192.168.x.x, 10.x.x.x, 172.16-31.x.x, localhost)"
                }
            )

        # IP is allowed, proceed with request
        return await call_next(request)

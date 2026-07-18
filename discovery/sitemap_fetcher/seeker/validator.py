from __future__ import annotations

import ipaddress
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from discovery.sitemap_fetcher.models.urls_model import url_model

logger = logging.getLogger(__name__)

MAX_URL_LENGTH = 2048

#schemes willing to be fetched
ALLOWED_SCHEMES = frozenset({"http", "https"})

FORBIDDEN_SCHEMES = frozenset({"javascript", "data", "file", "vbscript"})

_HOSTNAME_PATTERN = re.compile(
    r"^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+"
    r"[a-zA-Z]{2,63}$"    
)

# Valid hosts
_LOCAL_HOSTS = frozenset({"localhost"})

MIN_PORT = 1
MAX_PORT = 65535

@dataclass
class ValidationResult:
    '''Result of validating url'''

    is_valid: bool
    errors: list[str] = field(default_factory=list)

class Validate:

    @staticmethod
    def _safe_str(value: Optional[str]) -> str:

        return value if isinstance(value, str) else ""
    
    @staticmethod
    def _validate_scheme(url: url_model) -> bool:
        if not url.scheme:
            return False
        return url.scheme.strip().lower() in ALLOWED_SCHEMES
    
    @classmethod
    def _validate_host(cls, url: url_model) -> bool:
        if not url.host:
            return False
        
        host = url.host.strip().lower()

        if host in _LOCAL_HOSTS:
            return True
        
        # Allow literal IPV4 / IPV^ address as hosts.
        try:
            ipaddress.ip_address(host.strip("[]"))
            return True
        except ValueError:
            pass


        if len(host) > 253:
            return False
        
        return bool(_HOSTNAME_PATTERN.fullmatch(host))
    
    @classmethod
    def _validate_port(cls, url: url_model) -> bool:
        if url.port is None or url.port == "":
            return True
        
        try:
            port = int(url.port)
        except (TypeError, ValueError):
            return False
        
        return MIN_PORT <= port <= MAX_PORT
    
    @classmethod
    def _validate_path(cls, url: url_model) -> bool:
        path = url.path
        if not path:
            return True
        
        return isinstance(path, str) and path.startswith("/")
    
    @classmethod
    def _validate_length(cls, url: url_model) -> bool:
        reconstructed = (
            f"{cls._safe_str(url.scheme)}://"
            f"{cls._safe_str(url.host)}"
            f"{(':' + str(url.port)) if url.port else ''}"
            f"{cls._safe_str(url.path)}"
            f"{('?' + cls._safe_str(url.query)) if url.query else ''}"
            f"{('#' + cls._safe_str(url.fragment)) if url.fragment else ''}"
        )
        return len(reconstructed) <= MAX_URL_LENGTH

    @classmethod
    def _validate_characters(cls, url: url_model) -> bool:
        content = "".join(
            cls._safe_str(part)
            for part in (
                url.scheme,
                url.user,
                url.password,
                url.host,
                url.path,
                url.query,
                url.fragment,
            )
        )
        return not any(ord(char) < 32 or ord(char) == 127 for char in content)
    
    @classmethod
    def _validate_security(cls, url: url_model) -> bool:
        if not url.scheme:
            return False
        return url.scheme.strip().lower() not in FORBIDDEN_SCHEMES
    
    @classmethod
    def validate(cls, url: url_model) -> ValidationResult:

        checks: dict[str, bool] = {
            "Invalid scheme": cls._validate_scheme(url),
            "Invalid host": cls._validate_host(url),
            "Invalid port": cls._validate_port(url),
            "Invalid path": cls._validate_path(url),
            "URL exceeds maximum length": cls._validate_length(url),
            "Contains invalid characters": cls._validate_characters(url),
            "Unsafe URL scheme": cls._validate_security(url),
        }

        errors = [message for message, passed in checks.items() if not passed]

        if errors:
            logger.debug("URL failed validation: %s", errors)

        return ValidationResult(is_valid=not errors, errors=errors)
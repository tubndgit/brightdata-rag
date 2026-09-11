"""URL validation and canonicalization shared by fetching and discovery."""

from urllib.parse import urlsplit, urlunsplit


def normalize_url(url: str) -> str:
    if any(c.isspace() or ord(c) < 32 for c in url):
        raise ValueError("URL must not contain whitespace or control characters")
    parts = urlsplit(url)
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        raise ValueError("URL must be an absolute http:// or https:// URL")
    if parts.username is not None or parts.password is not None:
        raise ValueError("URLs containing credentials are not supported")
    host = parts.hostname.encode("idna").decode("ascii").lower()
    if ":" in host:
        host = f"[{host}]"
    port = parts.port
    scheme = parts.scheme.lower()
    if port is not None and port != {"http": 80, "https": 443}[scheme]:
        host = f"{host}:{port}"
    path = "" if parts.path == "/" else parts.path
    return urlunsplit((scheme, host, path, parts.query, ""))

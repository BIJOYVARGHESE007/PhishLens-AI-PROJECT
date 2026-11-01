"""URL feature extraction + human-readable indicators for PhishLens AI.
Only the URL string is analysed. The site is NEVER visited."""
import re
from urllib.parse import urlparse
import numpy as np

SUSPICIOUS_WORDS = ["login", "verify", "update", "secure", "account", "bank", "signin",
                    "confirm", "password", "wallet", "payment", "billing", "suspend",
                    "unlock", "invoice", "refund", "kyc", "otp"]
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "cutt.ly", "rb.gy"}
RISKY_TLDS = {"xyz", "top", "tk", "ml", "ga", "cf", "gq", "click", "support", "work", "zip", "country"}
IP_RE = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")

FEATURE_NAMES = [
    "url_length", "hostname_length", "path_length", "num_dots", "num_digits", "num_hyphens",
    "num_underscores", "num_special_chars", "num_subdomains", "has_ip", "is_https",
    "has_at_symbol", "num_params", "suspicious_word_count", "is_shortener", "risky_tld",
    "digit_ratio", "has_port", "double_slash_in_path", "hex_encoding_count", "entropy",
]

def _entropy(s: str) -> float:
    if not s:
        return 0.0
    p = np.array([s.count(c) for c in set(s)], dtype=float) / len(s)
    return float(-(p * np.log2(p)).sum())

def normalize(url: str) -> str:
    url = url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url
    return url

def extract_features(url: str) -> dict:
    url = normalize(url)
    p = urlparse(url)
    host = (p.hostname or "").lower()
    parts = host.split(".") if host else []
    tld = parts[-1] if parts else ""
    low = url.lower()
    return {
        "url_length": len(url),
        "hostname_length": len(host),
        "path_length": len(p.path),
        "num_dots": url.count("."),
        "num_digits": sum(c.isdigit() for c in url),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_special_chars": len(re.findall(r"[^a-zA-Z0-9]", url)),
        "num_subdomains": max(len(parts) - 2, 0),
        "has_ip": int(bool(IP_RE.match(host))),
        "is_https": int(p.scheme == "https"),
        "has_at_symbol": int("@" in url),
        "num_params": len(p.query.split("&")) if p.query else 0,
        "suspicious_word_count": sum(w in low for w in SUSPICIOUS_WORDS),
        "is_shortener": int(host in SHORTENERS),
        "risky_tld": int(tld in RISKY_TLDS),
        "digit_ratio": sum(c.isdigit() for c in url) / max(len(url), 1),
        "has_port": int(p.port is not None) if _safe_port(p) else 0,
        "double_slash_in_path": int("//" in p.path),
        "hex_encoding_count": len(re.findall(r"%[0-9a-fA-F]{2}", url)),
        "entropy": _entropy(host),
    }

def _safe_port(p):
    try:
        p.port
        return True
    except ValueError:
        return False

def features_vector(url: str) -> list:
    f = extract_features(url)
    return [f[n] for n in FEATURE_NAMES]

def indicators(url: str) -> list:
    """Plain-language warning flags: (severity, message). Decision support only."""
    f, out = extract_features(url), []
    if f["has_ip"]: out.append(("high", "The hostname is a raw IP address instead of a domain name."))
    if f["has_at_symbol"]: out.append(("high", "The URL contains '@', which can hide the real destination."))
    if f["is_shortener"]: out.append(("medium", "A link shortener hides the final destination."))
    if f["suspicious_word_count"] >= 2: out.append(("medium", f"Contains {f['suspicious_word_count']} sensitive keywords (login, verify, bank...)."))
    elif f["suspicious_word_count"] == 1: out.append(("low", "Contains a sensitive keyword (login, verify, bank...)."))
    if not f["is_https"]: out.append(("low", "Uses HTTP, not HTTPS (HTTPS alone does not prove safety)."))
    if f["url_length"] > 75: out.append(("medium", f"Unusually long URL ({f['url_length']} characters)."))
    if f["num_subdomains"] >= 3: out.append(("medium", f"Many subdomains ({f['num_subdomains']}), often used to imitate brands."))
    if f["num_hyphens"] >= 3: out.append(("low", "Many hyphens in the URL."))
    if f["risky_tld"]: out.append(("low", "Uses a top-level domain frequently abused in campaigns."))
    if f["hex_encoding_count"] >= 3: out.append(("medium", "Heavy percent-encoding may be obscuring content."))
    if f["has_port"]: out.append(("low", "Uses a non-standard port."))
    return out

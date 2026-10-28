# Explicit source allowlist used to check every web search result
# A matching domain or subdomain and an HTTP or HTTPS scheme are both required

from urllib.parse import urlparse

TRUSTED_DOMAINS = {
    "medlineplus.gov",
    "clevelandclinic.org",
    "aaos.org",  # orthoinfo.aaos.org
    "orthoinfo.org",
    "ota.org",
    "ncbi.nlm.nih.gov",  # covers pmc.ncbi.nlm.nih.gov and www.ncbi.nlm.nih.gov
    "nih.gov",
    "mayoclinic.org",  # covers sportsmedicine.mayoclinic.org
    "nhs.uk",  # covers every *.nhs.uk trust site (hacw, nth, whittington, www)
    "healthline.com",
    "hss.edu",
    "medicalnewstoday.com",
    "nuffieldhealth.com",
    "sportsinjuryclinic.net",
    "webmd.com",
}


ALLOWED_SCHEMES = {"http", "https"}  # citations become links in the UI; javascript:, file:, data: etc. are refused


def is_trusted_domain(url: str) -> bool:
    """Accept HTTP or HTTPS URLs whose hostname matches an allowed domain or subdomain
    Hostname parsing excludes credentials and ports, while the dot boundary rejects lookalike domains
    """
    try:
        parsed = urlparse(url.strip())
        host = parsed.hostname  # drops user:pass@ and :port, lowercases
    except (ValueError, AttributeError):
        return False
    if parsed.scheme.lower() not in ALLOWED_SCHEMES or not host:
        return False
    host = host.rstrip(".")  # "nhs.uk." is the same host as "nhs.uk"

    # Treat "www.example.com" the same as "example.com"
    host = host.removeprefix("www.")

    return any(
        host == domain or host.endswith(f".{domain}")
        for domain in TRUSTED_DOMAINS
    )
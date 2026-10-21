# Trusted source domains used to filter web search results

# Add new source domains here when updating the document manifest

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


def is_trusted_domain(url: str) -> bool:
    """Check whether the URL belongs to a trusted domain or one of its subdomains."""
    try:
        host = urlparse(url).netloc.lower()
    except ValueError:
        return False
    if not host:
        return False

    # Treat "www.example.com" the same as "example.com".
    if host.startswith("www."):
        host = host[4:]

    return any(
        host == domain or host.endswith(f".{domain}")
        for domain in TRUSTED_DOMAINS
    )

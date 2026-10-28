# Deterministic model, document and search replacements for demos and tests
# These exercise graph routing and filtering without contacting Ollama, Tavily or the local index
# The returned content is fixed demonstration data rather than a finding-specific model answer

class ScriptedLLM:
    """Replay decision replies and return fixed synthesis text, recording each prompt for inspection"""

    def __init__(
        self,
        decisions,
        synthesis_text="[DEMO PLACEHOLDER - canned text from the scripted fake model, not real advice and not based on the finding.]",
    ):
        """Copy the decision script so a run cannot consume the shared scenario list"""
        self.decisions = list(decisions)
        self.synthesis_text = synthesis_text
        self.calls: list[tuple[str, str]] = []

    def __call__(self, prompt, system=None):
        """Distinguish decision and synthesis calls using the system prompt
        Once scripted decisions are exhausted, return stop and let the graph guards handle it
        """
        if system and "decision step" in system:
            self.calls.append(("decide", prompt))
            return self.decisions.pop(0) if self.decisions else '{"action": "stop"}'
        self.calls.append(("synthesize", prompt))
        return self.synthesis_text


def fake_retrieve(query, k=3, finding=None):
    """Return a finding-labelled example chunk, sliced using k like a limited retrieval result"""
    return [
        {
            "source_title": f"{finding} overview (fake doc store)",
            "source_url": "https://www.nhs.uk/fake-doc-store",
            "findings": [finding],
            "text": f"{finding} is commonly managed with rest, ice and graded rehabilitation.",
            "score": 0.8,
        }
    ][:k]


def fake_search(query, max_results):
    """Return one approved and one unapproved source to exercise the domain filter
    The fixed pair deliberately does not vary with the query or max_results
    """
    return [
        {
            "url": "https://www.healthline.com/fake-page",
            "title": "Fake trusted page",
            "content": "Physiotherapy usually helps recovery; see a clinician if swelling persists.",
            "score": 0.7,
        },
        {
            "url": "https://miracle-cures.example.biz/knee",
            "title": "Untrusted page",
            "content": "One weird trick doctors hate.",
            "score": 0.9,
        },
    ]


# Default decisions for samples without their own script
DEFAULT_DEMO_SCRIPT = [
    '{"action": "query_doc_store"}',
    '{"action": "web_search"}',
    '{"action": "synthesize"}',
    '{"action": "stop"}',
]

# Model replies for each demo scenario
DEMO_SCRIPTS = {
    # Search local and web sources, write the answer, then stop
    "acl_plain": [
        '{"action": "query_doc_store"}',
        '{"action": "web_search"}',
        '{"action": "synthesize"}',
        '{"action": "stop"}',
    ],
    # Two invalid replies trigger the fallback before synthesis
    "bakers_with_question": [
        "uhh, maybe look something up?",
        "no idea, sorry",
        '{"action": "synthesize"}',
        '{"action": "stop"}',
    ],
    # Try stopping early and repeating a tool to exercise the guards
    "fracture_plain": [
        '{"action": "stop"}',
        '{"action": "query_doc_store"}',
        '{"action": "query_doc_store"}',
        '{"action": "stop"}',
    ],
}

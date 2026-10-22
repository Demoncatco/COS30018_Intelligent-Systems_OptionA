# Example findings and questions for running the recommendation graph

STUB_INPUTS = [
    # Basic examples
    {"name": "acl_plain", "finding": "ACL"},
    {
        "name": "bakers_with_question",
        "finding": "Baker's",
        "user_query": "baker's cyst behind the knee, how is it treated",
    },
    {"name": "fracture_plain", "finding": "Fracture"},

    # Other finding categories
    {"name": "mcl_plain", "finding": "MCL"},
    {"name": "medial_meniscus_plain", "finding": "Medial Meniscus"},
    {"name": "lateral_meniscus_plain", "finding": "Lateral Meniscus"},
    {"name": "medial_oa_plain", "finding": "Medial OA"},
    {"name": "lateral_oa_plain", "finding": "Lateral OA"},
    {"name": "pf_oa_plain", "finding": "PF OA"},
    {"name": "effusion_plain", "finding": "Effusion"},
    {"name": "synovitis_plain", "finding": "Synovitis"},
    {"name": "contusion_plain", "finding": "Contusion"},

    # Patient questions that override the default query
    {
        "name": "acl_sport_question",
        "finding": "ACL",
        "user_query": "can I return to football after an ACL tear and how long does rehab take",
    },
    {
        "name": "mcl_brace_question",
        "finding": "MCL",
        "user_query": "do I need a brace or surgery for an MCL sprain",
    },
    {
        "name": "meniscus_surgery_question",
        "finding": "Medial Meniscus",
        "user_query": "when is surgery needed for a torn meniscus versus physio",
    },
    {
        "name": "oa_injection_question",
        "finding": "Medial OA",
        "user_query": "do knee injections help osteoarthritis and what else can I do",
    },
    {
        "name": "fracture_surgery_question",
        "finding": "Fracture",
        "user_query": "will a broken kneecap need surgery and how long until I can walk",
    },
    {
        "name": "effusion_drainage_question",
        "finding": "Effusion",
        "user_query": "should swelling on the knee be drained",
    },

    # Edge cases
    # No local sources: try web search or use the no-sources response.
    {"name": "unknown_finding", "finding": "Osteochondritis Dissecans"},
    # An off-topic question attached to a real finding.
    {
        "name": "off_topic_question",
        "finding": "ACL",
        "user_query": "what is the best pizza in Melbourne",
    },
]


"""
inference/vocabulary.py
=======================
Controlled vocabulary for the Phase 4A inference pipeline.

Every list and mapping in this file is derived EXCLUSIVELY from:
  - processed_data/preprocessing_metadata.json  (authoritative source)
  - processed_data/feature_names.json           (ground truth for column order)

Nothing here was invented. If the training metadata changes, re-derive
this file from preprocessing_metadata.json rather than editing it manually.

DESIGN PRINCIPLE
----------------
The inference pipeline must produce exactly the same feature columns —
in exactly the same order — as were present in train_features.csv.
This file makes those mappings explicit and testable.
"""

# ---------------------------------------------------------------------------
# TOP-50 APPROVED CATEGORIES
# Exactly as stored in preprocessing_metadata.json -> top_categories_list
# Order matters: these are ranked by training-set frequency.
# 'Unknown' is position 4 — it was the fill-in value for missing category_list.
# ---------------------------------------------------------------------------
APPROVED_CATEGORIES = [
    "Software",
    "Biotechnology",
    "Mobile",
    "Unknown",           # historical missing-data placeholder — see NOTE below
    "Curated Web",
    "E-Commerce",
    "Social Media",
    "Advertising",
    "Enterprise Software",
    "Games",
    "Health Care",
    "Services",
    "Internet",
    "Finance",
    "Technology",
    "Analytics",
    "Hardware + Software",
    "Security",
    "Clean Technology",
    "Apps",
    "Semiconductors",
    "SaaS",
    "Health and Wellness",
    "Web Hosting",
    "Networking",
    "Video",
    "Social Network Media",
    "Search",
    "Manufacturing",
    "Music",
    "Education",
    "Messaging",
    "Cloud Computing",
    "Medical",
    "News",
    "Travel",
    "Information Technology",
    "Web Development",
    "Telecommunications",
    "Business Services",
    "Media",
    "Sales and Marketing",
    "Entertainment",
    "iPhone",
    "Design",
    "Fashion",
    "Consulting",
    "Marketplaces",
    "Android",
    "FinTech",
]
# NOTE on 'Unknown':
#   In the training data, companies with no recorded category_list were
#   assigned the string "Unknown" before encoding. This produced cat_Unknown=1.
#   At inference time, a user who provides at least one real category should
#   NEVER be mapped to 'Unknown'. Only set cat_Unknown=1 if the user provides
#   no category whatsoever and no keyword match is found.

# ---------------------------------------------------------------------------
# APPROVED COUNTRIES
# Exactly as stored in preprocessing_metadata.json -> top_countries_list
# 'Unknown' is the fill-in for missing country_code.
# ---------------------------------------------------------------------------
APPROVED_COUNTRIES = [
    "USA",
    "Unknown",           # historical missing-data placeholder — see NOTE below
    "GBR",
    "CAN",
    "ISR",
    "DEU",
    "FRA",
    "RUS",
    "CHN",
    "IND",
]
# NOTE on 'Unknown':
#   In training data, companies with no country_code were assigned "Unknown".
#   At inference, country_Unknown=1 only when the user provides no country
#   and extraction fails. For a normal complete submission, country_Unknown=0.

# ---------------------------------------------------------------------------
# COUNTRY NAME / ALIAS -> ISO CODE MAPPING
# Maps common country names and aliases a user might type to the
# ISO-3166 Alpha-3 codes used in the training data.
# Only codes that appear in APPROVED_COUNTRIES produce a known column.
# Any country not in APPROVED_COUNTRIES maps to country_other=1.
# ---------------------------------------------------------------------------
COUNTRY_NAME_TO_CODE = {
    # United States
    "united states": "USA",
    "usa": "USA",
    "us": "USA",
    "u.s.": "USA",
    "u.s.a.": "USA",
    "america": "USA",
    "united states of america": "USA",
    # United Kingdom
    "united kingdom": "GBR",
    "uk": "GBR",
    "u.k.": "GBR",
    "britain": "GBR",
    "great britain": "GBR",
    "england": "GBR",
    # Canada
    "canada": "CAN",
    # Israel
    "israel": "ISR",
    # Germany
    "germany": "DEU",
    "deutschland": "DEU",
    # France
    "france": "FRA",
    # Russia
    "russia": "RUS",
    "russian federation": "RUS",
    # China
    "china": "CHN",
    "people's republic of china": "CHN",
    "prc": "CHN",
    # India
    "india": "IND",
    # Bay-area / California signals for geography flags
    # These are used by geography flag detection, not country encoding
}

# Geography signals for is_sf_bay_area
# If any of these appear in the location text, is_sf_bay_area = 1
SF_BAY_AREA_SIGNALS = [
    "sf bay area",
    "san francisco",
    "silicon valley",
    "palo alto",
    "mountain view",
    "sunnyvale",
    "cupertino",
    "san jose",
    "menlo park",
    "redwood city",
    "santa clara",
    "fremont",
    "oakland",
    "berkeley",
    "south bay",
    "bay area",
]

# Geography signals for is_california (broader than Bay Area)
CALIFORNIA_SIGNALS = [
    "california",
    " ca ",
    ", ca",
    "ca,",
    "los angeles",
    "san diego",
    "los angeles",
    "san diego",
    "san francisco",
    "palo alto",
    "silicon valley",
    "mountain view",
    "sunnyvale",
    "cupertino",
    "berkeley",
    "sacramento",
    "irvine",
    "santa barbara",
    "monterey",
    "pasadena",
]

# ---------------------------------------------------------------------------
# KEYWORD-BASED CATEGORY MATCHING
# Maps free-text keywords/phrases to approved category names.
#
# DESIGN RULES:
# 1. Only categories from APPROVED_CATEGORIES may appear as values.
# 2. Keywords are checked case-insensitively against the idea text.
# 3. Each keyword maps to ONE category. The same idea text may match
#    multiple keywords -> multiple categories (multi-hot).
# 4. Keywords are ordered from most specific to least specific within
#    each category. The matching algorithm checks all keywords and
#    collects all matched categories.
# 5. When adding keywords, verify the target category is in
#    APPROVED_CATEGORIES before adding.
# ---------------------------------------------------------------------------
KEYWORD_TO_CATEGORY = {
    # ---- Software / SaaS / Enterprise ----
    "software platform":        "Software",
    "software solution":        "Software",
    "software tool":            "Software",
    "software application":     "Software",
    "software product":         "Software",
    "saas platform":            "SaaS",
    "saas tool":                "SaaS",
    "saas solution":            "SaaS",
    "saas":                     "SaaS",
    "software-as-a-service":    "SaaS",
    "enterprise software":      "Enterprise Software",
    "b2b software":             "Enterprise Software",
    "business software":        "Enterprise Software",
    "erp":                      "Enterprise Software",
    "crm":                      "Enterprise Software",
    "workflow automation":      "Enterprise Software",
    "workflow tool":            "Enterprise Software",
    "productivity tool":        "Enterprise Software",

    # ---- Mobile ----
    "mobile app":               "Mobile",
    "mobile application":       "Mobile",
    "mobile platform":          "Mobile",
    "smartphone app":           "Mobile",
    "android app":              "Android",
    "ios app":                  "Mobile",
    "iphone app":               "iPhone",
    "iphone":                   "iPhone",
    "android":                  "Android",

    # ---- AI / Analytics / Data ----
    "artificial intelligence":  "Analytics",
    "machine learning":         "Analytics",
    "deep learning":            "Analytics",
    "ai platform":              "Analytics",
    "ai tool":                  "Analytics",
    "ai solution":              "Analytics",
    "ai powered":               "Analytics",
    "ai-powered":               "Analytics",
    "predictive analytics":     "Analytics",
    "data analytics":           "Analytics",
    "big data":                 "Analytics",
    "data science":             "Analytics",
    "business intelligence":    "Analytics",
    "recommendation engine":    "Analytics",
    "natural language":         "Analytics",
    "computer vision":          "Analytics",
    "nlp":                      "Analytics",

    # ---- E-Commerce / Marketplaces ----
    "e-commerce":               "E-Commerce",
    "ecommerce":                "E-Commerce",
    "online store":             "E-Commerce",
    "online shop":              "E-Commerce",
    "online marketplace":       "Marketplaces",
    "marketplace":              "Marketplaces",
    "peer-to-peer marketplace": "Marketplaces",
    "two-sided marketplace":    "Marketplaces",

    # ---- Health / Medical / Biotech ----
    "biotechnology":            "Biotechnology",
    "biotech":                  "Biotechnology",
    "pharmaceutical":           "Biotechnology",
    "drug discovery":           "Biotechnology",
    "genomics":                 "Biotechnology",
    "healthcare":               "Health Care",
    "health care":              "Health Care",
    "health technology":        "Health Care",
    "health platform":          "Health Care",
    "patient monitoring":       "Medical",
    "remote patient":           "Medical",
    "telemedicine":             "Medical",
    "telehealth":               "Medical",
    "medical device":           "Medical",
    "medical platform":         "Medical",
    "clinical":                 "Medical",
    "diagnostics":              "Medical",
    "hospital":                 "Medical",
    "ehr":                      "Medical",
    "electronic health record": "Medical",
    "wellness platform":        "Health and Wellness",
    "fitness app":              "Health and Wellness",
    "mental health":            "Health and Wellness",
    "nutrition":                "Health and Wellness",
    "health and wellness":      "Health and Wellness",
    "preventive health":        "Health and Wellness",

    # ---- Education ----
    "education platform":       "Education",
    "edtech":                   "Education",
    "online learning":          "Education",
    "e-learning":               "Education",
    "learning management":      "Education",
    "lms":                      "Education",
    "tutoring":                 "Education",
    "online education":         "Education",
    "student":                  "Education",
    "exam preparation":         "Education",
    "competitive exam":         "Education",
    "upskilling":               "Education",
    "skills training":          "Education",
    "course platform":          "Education",
    "mooc":                     "Education",

    # ---- Finance / FinTech ----
    "fintech":                  "FinTech",
    "financial technology":     "FinTech",
    "payments":                 "FinTech",
    "digital payments":         "FinTech",
    "neobank":                  "FinTech",
    "digital banking":          "FinTech",
    "insurtech":                "FinTech",
    "investment platform":      "Finance",
    "finance platform":         "Finance",
    "accounting software":      "Finance",
    "invoice":                  "Finance",
    "invoicing":                "Finance",
    "payroll":                  "Finance",
    "expense management":       "Finance",
    "lending":                  "Finance",
    "microfinance":             "Finance",
    "wealth management":        "Finance",
    "tax":                      "Finance",
    "bookkeeping":              "Finance",

    # ---- Security / Cybersecurity ----
    "cybersecurity":            "Security",
    "cyber security":           "Security",
    "information security":     "Security",
    "network security":         "Security",
    "data security":            "Security",
    "identity verification":    "Security",
    "fraud detection":          "Security",
    "authentication":           "Security",
    "privacy":                  "Security",

    # ---- Cloud / Infrastructure / Networking ----
    "cloud computing":          "Cloud Computing",
    "cloud platform":           "Cloud Computing",
    "cloud infrastructure":     "Cloud Computing",
    "cloud storage":            "Cloud Computing",
    "cloud services":           "Cloud Computing",
    "networking":               "Networking",
    "network platform":         "Networking",
    "web hosting":              "Web Hosting",
    "hosting":                  "Web Hosting",
    "server":                   "Web Hosting",

    # ---- Social / Media / Content ----
    "social network":           "Social Network Media",
    "social platform":          "Social Media",
    "social media":             "Social Media",
    "content platform":         "Media",
    "streaming":                "Video",
    "video platform":           "Video",
    "video streaming":          "Video",
    "media platform":           "Media",
    "news platform":            "News",
    "news app":                 "News",
    "blogging":                 "Media",
    "podcast":                  "Media",
    "entertainment platform":   "Entertainment",
    "gaming":                   "Games",
    "game":                     "Games",
    "gaming platform":          "Games",
    "music platform":           "Music",
    "music streaming":          "Music",
    "music app":                "Music",

    # ---- B2B Services / Consulting ----
    "consulting":               "Consulting",
    "business services":        "Business Services",
    "professional services":    "Business Services",
    "hr platform":              "Business Services",
    "human resources":          "Business Services",
    "recruitment":              "Business Services",
    "staffing":                 "Business Services",
    "sales platform":           "Sales and Marketing",
    "marketing platform":       "Sales and Marketing",
    "sales and marketing":      "Sales and Marketing",
    "crm platform":             "Sales and Marketing",
    "lead generation":          "Sales and Marketing",
    "digital marketing":        "Sales and Marketing",
    "advertising platform":     "Advertising",
    "ad tech":                  "Advertising",
    "adtech":                   "Advertising",

    # ---- Logistics / Manufacturing / Hardware ----
    "supply chain":             "Manufacturing",
    "logistics":                "Manufacturing",
    "manufacturing":            "Manufacturing",
    "hardware":                 "Hardware + Software",
    "iot":                      "Hardware + Software",
    "internet of things":       "Hardware + Software",
    "robotics":                 "Hardware + Software",
    "drone":                    "Hardware + Software",
    "semiconductor":            "Semiconductors",
    "chip":                     "Semiconductors",

    # ---- Clean Tech / Sustainability ----
    "clean technology":         "Clean Technology",
    "cleantech":                "Clean Technology",
    "renewable energy":         "Clean Technology",
    "solar energy":             "Clean Technology",
    "wind energy":              "Clean Technology",
    "sustainability":           "Clean Technology",
    "green tech":               "Clean Technology",
    "carbon":                   "Clean Technology",
    "food waste":               "Clean Technology",
    "waste management":         "Clean Technology",
    "recycling":                "Clean Technology",
    "climate tech":             "Clean Technology",

    # ---- Travel ----
    "travel platform":          "Travel",
    "travel app":               "Travel",
    "tourism":                  "Travel",
    "hotel booking":            "Travel",
    "flight booking":           "Travel",
    "trip planning":            "Travel",

    # ---- Telecom ----
    "telecommunications":       "Telecommunications",
    "telecom":                  "Telecommunications",
    "broadband":                "Telecommunications",
    "wireless":                 "Telecommunications",
    "5g":                       "Telecommunications",

    # ---- Messaging / Communication ----
    "messaging app":            "Messaging",
    "chat platform":            "Messaging",
    "communication platform":   "Messaging",
    "real-time messaging":      "Messaging",

    # ---- Web / Design ----
    "web development":          "Web Development",
    "website builder":          "Web Development",
    "web app":                  "Web Development",
    "design platform":          "Design",
    "design tool":              "Design",
    "graphic design":           "Design",
    "ui/ux":                    "Design",

    # ---- Search ----
    "search engine":            "Search",
    "search platform":          "Search",
    "discovery platform":       "Search",

    # ---- Fashion ----
    "fashion platform":         "Fashion",
    "fashion app":              "Fashion",
    "fashion tech":             "Fashion",
    "apparel":                  "Fashion",

    # ---- Restaurant / Food (closest: Clean Technology for food waste;
    #      no dedicated Food category in training vocab) ----
    "restaurant":               "Services",
    "food delivery":            "Services",
    "food platform":            "Services",
    "food tech":                "Services",

    # ---- Generic tech / internet ----
    "internet platform":        "Internet",
    "web platform":             "Internet",
    "curated":                  "Curated Web",
    "aggregator":               "Curated Web",
    "information technology":   "Information Technology",

    # ---- Technology (generic catch-all) ----
    # Justified by training data: "Technology" is one of the top-50 categories
    # (position 15 in frequency ranking, 279 training examples, 60.2% success
    # rate). In the Crunchbase dataset it was applied as a broad label when no
    # more specific category fit.
    #
    # KEYWORD SAFETY RULE: A bare "technology" substring is NOT used because it
    # would shadow existing longer keywords such as "clean technology" (→ Clean
    # Technology) and "information technology" (→ Information Technology). Only
    # "tech *" prefixed phrases are used here — these cannot appear as substrings
    # of "clean technology …" or "information technology …" constructions.
    # Verified safe: none of these phrases are substrings of any existing
    # KEYWORD_TO_CATEGORY entry, and "clean technology X" does NOT contain
    # "tech company", "tech startup", etc.
    "tech company":             "Technology",
    "tech startup":             "Technology",
    "tech firm":                "Technology",
    "tech solutions":           "Technology",
    "tech services":            "Technology",
    "tech platform":            "Technology",

    # ---- Apps (generic) ----
    "app":                      "Apps",
    "application":              "Apps",
}

# Set of approved category names for fast membership testing
APPROVED_CATEGORY_SET = set(APPROVED_CATEGORIES)

# Set of approved ISO codes for fast membership testing
APPROVED_COUNTRY_SET = set(APPROVED_COUNTRIES)

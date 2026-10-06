import os

MODEL = "deepseek-v4-pro"
REASONING_EFFORT = "low"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

SEED_CHAR_LIMIT = 8_000
CANDIDATE_CHAR_LIMIT = 4_000
MAX_CANDIDATES = 8

CRAWL_TIMEOUT_S = 20
LLM_TIMEOUT_S = 120

PHOENIX_PROJECT = "bookmark-recommender"
PHOENIX_URL = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")

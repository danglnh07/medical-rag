import os


def inject_llm_env(provider: str, api_key: str):
    if provider == "openai":
        os.environ["OPENAI_API_KEY"] = api_key
    elif provider == "anthropic":
        os.environ["ANTHROPIC_API_KEY"] = api_key
    elif provider == "gemini":
        os.environ["GEMINI_API_KEY"] = api_key

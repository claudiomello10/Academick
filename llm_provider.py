"""Multi-provider LLM interface.

Routes LLM calls to the correct provider (OpenAI, Anthropic, or DeepSeek)
based on the model name. Uses synchronous clients for simplicity.
"""

import os

# Lazy-initialized clients
_openai_client = None
_anthropic_client = None


def get_provider(model: str) -> str:
    """Determine the provider from the model name."""
    model_lower = model.lower()
    if model_lower.startswith("gpt") or model_lower.startswith("text-davinci"):
        return "openai"
    elif model_lower.startswith("claude"):
        return "anthropic"
    elif model_lower.startswith("deepseek"):
        return "deepseek"
    else:
        return "openai"  # Default fallback


def generate_response(
    messages: list[dict],
    model: str,
    temperature: float = 0.7,
) -> str:
    """
    Generate a response using the specified model.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model name (determines provider automatically)
        temperature: Sampling temperature

    Returns:
        Response text string
    """
    provider = get_provider(model)

    if provider == "openai":
        return _openai_generate(messages, model, temperature)
    elif provider == "anthropic":
        return _anthropic_generate(messages, model, temperature)
    elif provider == "deepseek":
        return _deepseek_generate(messages, model, temperature)
    else:
        return _openai_generate(messages, model, temperature)


def _openai_generate(messages: list[dict], model: str, temperature: float) -> str:
    """Generate using OpenAI API."""
    global _openai_client

    if _openai_client is None:
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        _openai_client = OpenAI(api_key=api_key)

    # Some OpenAI models (e.g., gpt-5-nano) don't support custom temperature,
    # so we don't pass it — matching the full app's behavior.
    response = _openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content


def _anthropic_generate(messages: list[dict], model: str, temperature: float) -> str:
    """Generate using Anthropic API."""
    global _anthropic_client

    if _anthropic_client is None:
        from anthropic import Anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        _anthropic_client = Anthropic(api_key=api_key)

    # Anthropic requires system message to be passed separately
    system_content = ""
    user_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
        else:
            user_messages.append({"role": msg["role"], "content": msg["content"]})

    response = _anthropic_client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_content,
        messages=user_messages,
        temperature=temperature,
    )
    return response.content[0].text


def _deepseek_generate(messages: list[dict], model: str, temperature: float) -> str:
    """Generate using DeepSeek API (OpenAI-compatible)."""
    from openai import OpenAI

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not set")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content

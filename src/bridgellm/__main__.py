"""Entry point for `python -m bridgellm`.

Prints version, installed providers, SDK status, and quick start guide.
"""

import sys


def main():
    from . import __all__
    from .compat import _UNIFYLLM_VERSION as VERSION, TESTED_RANGES, _get_installed_version
    from .registry import PROVIDERS

    print(f"bridgellm v{VERSION}")
    print(f"Python {sys.version.split()[0]}")
    print()

    # SDK status
    print("SDK Status:")
    for version_range in TESTED_RANGES:
        installed = _get_installed_version(version_range.package)
        status = f"v{installed}" if installed else "not installed"
        marker = "ok" if installed else "--"
        print(f"  {version_range.package:<16} {status:<16} [{marker}]")
    print()

    # Providers
    print(f"Built-in Providers ({len(PROVIDERS)}):")
    for name, config in PROVIDERS.items():
        compat = "OpenAI-compat" if config.openai_compatible else "native SDK"
        print(f"  {name:<14} {config.api_key_env:<24} {compat}")
    print()

    # Quick start
    print("Quick Start:")
    print("  from bridgellm import BridgeLLM")
    print()
    print("  llm = BridgeLLM(model=\"openai/gpt-4o\")")
    print("  response = await llm.complete(messages=[{\"role\": \"user\", \"content\": \"Hello\"}])")
    print()
    print("  # Switch providers by changing the model string:")
    print("  llm = BridgeLLM(model=\"groq/llama-3.3-70b\")")
    print("  llm = BridgeLLM(model=\"anthropic/claude-sonnet-4\")")
    print()
    print("Docs: https://github.com/karanpanchall/unifyLLM")


if __name__ == "__main__":
    main()

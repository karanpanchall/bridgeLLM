"""Real integration tests against the Anthropic API.

Run with: OPENAI_API_KEY=... ANTHROPIC_API_KEY=... python3 -m pytest tests/test_anthropic_real.py -v -s
Skipped automatically when ANTHROPIC_API_KEY is not set.
"""

import os

import pytest

from bridgellm import BridgeLLM, LLMResponse, StreamChunk, RequestConfig

SKIP_REASON = "ANTHROPIC_API_KEY not set"
requires_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason=SKIP_REASON,
)


@requires_key
class TestAnthropicCompletion:
    @pytest.mark.asyncio
    async def test_simple_completion(self):
        llm = BridgeLLM(model="anthropic/claude-sonnet-4-20250514")

        response = await llm.complete(
            messages=[{"role": "user", "content": "Reply with exactly: BRIDGELLM_OK"}],
            temperature=0.0,
            max_tokens=20,
        )

        assert isinstance(response, LLMResponse)
        assert response.content is not None
        assert "BRIDGELLM_OK" in response.content
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        assert response.finish_reason == "stop"
        print(f"\n  anthropic completion: '{response.content}' | {response.input_tokens}+{response.output_tokens} tokens")

    @pytest.mark.asyncio
    async def test_system_prompt_extraction(self):
        """Verify system messages are extracted to top-level param for Anthropic."""
        llm = BridgeLLM(
            model="anthropic/claude-sonnet-4-20250514",
            system_prompt="You are a pirate. Always say 'Ahoy' first.",
        )

        response = await llm.complete(
            messages=[{"role": "user", "content": "Say hello"}],
            temperature=0.0,
            max_tokens=30,
        )

        assert response.content is not None
        assert "ahoy" in response.content.lower()
        print(f"\n  system prompt: '{response.content}'")


@requires_key
class TestAnthropicStreaming:
    @pytest.mark.asyncio
    async def test_stream_text(self):
        llm = BridgeLLM(model="anthropic/claude-sonnet-4-20250514")

        collected_text = ""
        chunk_count = 0
        got_finish = False

        async for chunk in llm.stream(
            messages=[{"role": "user", "content": "Count from 1 to 5, one number per line. Nothing else."}],
            temperature=0.0,
            max_tokens=30,
        ):
            assert isinstance(chunk, StreamChunk)
            chunk_count += 1
            if chunk.delta_content:
                collected_text += chunk.delta_content
            if chunk.finish_reason:
                got_finish = True

        assert chunk_count > 1
        assert len(collected_text) > 0
        assert got_finish
        print(f"\n  anthropic streamed {chunk_count} chunks: '{collected_text.strip()}'")


@requires_key
class TestAnthropicToolCalling:
    @pytest.mark.asyncio
    async def test_tool_call(self):
        llm = BridgeLLM(model="anthropic/claude-sonnet-4-20250514")

        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                    },
                    "required": ["city"],
                },
            },
        }]

        response = await llm.complete(
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=tools,
            temperature=0.0,
            max_tokens=100,
        )

        assert response.finish_reason == "tool_calls"
        assert len(response.tool_calls) >= 1
        tool_call = response.tool_calls[0]
        assert tool_call.function_name == "get_weather"
        assert "city" in tool_call.arguments
        assert tool_call.call_id != ""
        print(f"\n  anthropic tool call: {tool_call.function_name}({tool_call.arguments})")


@requires_key
class TestAnthropicRequestConfig:
    @pytest.mark.asyncio
    async def test_stop_sequences(self):
        llm = BridgeLLM(model="anthropic/claude-sonnet-4-20250514")

        response = await llm.complete(
            messages=[{"role": "user", "content": "Count from 1 to 10, one per line."}],
            config=RequestConfig(stop=["5"]),
            temperature=0.0,
            max_tokens=100,
        )

        assert "6" not in (response.content or "")
        assert response.finish_reason == "stop"
        print(f"\n  anthropic stop: '{response.content.strip()}'")

    @pytest.mark.asyncio
    async def test_tool_choice_required(self):
        llm = BridgeLLM(model="anthropic/claude-sonnet-4-20250514")

        tools = [{
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Calculate math",
                "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}, "required": ["expr"]},
            },
        }]

        response = await llm.complete(
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            tools=tools,
            config=RequestConfig(tool_choice="required"),
            temperature=0.0,
            max_tokens=100,
        )

        # Even though the message doesn't need a tool, tool_choice=required forces it
        assert response.finish_reason == "tool_calls"
        assert len(response.tool_calls) >= 1
        print(f"\n  forced tool: {response.tool_calls[0].function_name}({response.tool_calls[0].arguments})")


@requires_key
class TestAnthropicListModels:
    @pytest.mark.asyncio
    async def test_list_models(self):
        llm = BridgeLLM(model="anthropic/claude-sonnet-4-20250514")

        models = await llm.list_models()
        assert len(models) > 0

        model_ids = [model.model_id for model in models]
        assert any("claude" in model_id for model_id in model_ids)

        # Anthropic returns rich metadata
        first_model = models[0]
        assert first_model.context_window is not None or first_model.max_output_tokens is not None
        print(f"\n  anthropic models: {len(models)} found, first: {model_ids[0]}")
        if first_model.context_window:
            print(f"  context_window: {first_model.context_window}, max_output: {first_model.max_output_tokens}")


@requires_key
class TestCrossProviderSameClient:
    @pytest.mark.asyncio
    async def test_openai_and_anthropic_one_client(self):
        """Use both OpenAI and Anthropic from the same BridgeLLM instance."""
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            pytest.skip("OPENAI_API_KEY not set")

        llm = BridgeLLM(
            model="openai/gpt-4o-mini",
            api_keys={
                "openai": openai_key,
                "anthropic": os.environ["ANTHROPIC_API_KEY"],
            },
        )

        # Call OpenAI
        openai_response = await llm.complete(
            messages=[{"role": "user", "content": "Reply with exactly: OPENAI_OK"}],
            temperature=0.0, max_tokens=10,
        )
        assert openai_response.content is not None
        assert openai_response.model.startswith("gpt")
        print(f"\n  openai: '{openai_response.content}' via {openai_response.model}")

        # Call Anthropic on the same client
        anthropic_response = await llm.complete(
            messages=[{"role": "user", "content": "Reply with exactly: ANTHROPIC"}],
            model="anthropic/claude-sonnet-4-20250514",
            temperature=0.0, max_tokens=10,
        )
        assert "ANTHROPIC" in anthropic_response.content
        print(f"  anthropic: '{anthropic_response.content}' via {anthropic_response.model}")

        assert "openai" in llm.active_providers
        assert "anthropic" in llm.active_providers

from click.testing import CliRunner
from llm.cli import cli
from llm_olmo import DEFAULT_SYSTEM_PROMPT
import llm
import json
import pytest

def test_plugin_is_installed():
    runner = CliRunner()
    result = runner.invoke(cli, ["plugins"])
    assert result.exit_code == 0, result.output
    names = [plugin["name"] for plugin in json.loads(result.output)]
    assert "llm-olmo" in names

@pytest.mark.parametrize(
    "model_name,prompt,expected",
    (
        (
            "olmo7b",
            "How are you doing?",
            "<|endoftext|>\n"
            "<|user|>\n"
            "How are you doing?\n"
            "<|assistant|>"
        ),
        (
            "olmo13b",
            "Tell me a joke",
            "<|endoftext|>\n"
            "<|user|>\n"
            "Tell me a joke\n"
            "<|assistant|>"
        ),
    ),
)
def test_build_prompt_simple(model_name, prompt, expected):
    model = llm.get_model(model_name)
    result = model.build_prompt(llm.Prompt(prompt, model), None)
    assert result == expected

@pytest.mark.parametrize("model_name", ["olmo7b", "olmo13b"])
def test_build_prompt_conversation(model_name):
    model = llm.get_model(model_name)
    conversation = model.conversation()
    conversation.responses = [
        llm.Response.fake(model, "What's the weather?", None, "It's sunny today!"),
        llm.Response.fake(model, "What time is it?", None, "I don't have access to real-time information."),
    ]
    result = model.build_prompt(llm.Prompt("How are you?", model), conversation)
    expected = (
        "<|endoftext|>\n"
        "<|user|>\n"
        "What's the weather?\n"
        "<|assistant|>\n"
        "It's sunny today!\n"
        "<|endoftext|>\n"
        "<|user|>\n"
        "What time is it?\n"
        "<|assistant|>\n"
        "I don't have access to real-time information.\n"
        "<|endoftext|>\n"
        "<|user|>\n"
        "How are you?\n"
        "<|assistant|>"
    )
    assert result == expected

def test_download_commands():
    runner = CliRunner()
    result = runner.invoke(cli, ["olmo", "download_7b"])
    assert result.exit_code == 0
    result = runner.invoke(cli, ["olmo", "download_13b"])
    assert result.exit_code == 0

@pytest.mark.parametrize("model_name", ["olmo7b", "olmo13b"])
def test_model_options(model_name):
    model = llm.get_model(model_name)
    assert hasattr(model.Options, 'no_cuda')
    assert model.Options.no_cuda is False

@pytest.mark.parametrize("model_name", ["olmo7b", "olmo13b"])
def test_model_properties(model_name):
    model = llm.get_model(model_name)
    if model_name == "olmo7b":
        assert model.model_path == "allenai/OLMo-2-1124-7B"
    else:
        assert model.model_path == "allenai/OLMo-2-1124-13B"

def test_response_extraction():
    """Test that response extraction correctly handles the chat format"""
    model = llm.get_model("olmo7b")
    test_response = (
        "<|endoftext|>\n"
        "<|user|>\n"
        "How are you?\n"
        "<|assistant|>\n"
        "I'm functioning as expected.\n"
        "<|endoftext|>"
    )
    # Simulate response extraction
    response_parts = test_response.split("<|assistant|>")
    extracted = response_parts[-1].split("<|endoftext|>")[0].strip()
    assert extracted == "I'm functioning as expected."
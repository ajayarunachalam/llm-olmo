# llm-olmo

[![PyPI](https://img.shields.io/pypi/v/llm-olmo.svg)](https://pypi.org/project/llm-olmo/)
[![Changelog](https://img.shields.io/github/v/release/ajayarunachalam/llm-olmo?include_prereleases&label=changelog)](https://github.com/ajayarunachalam/llm-olmo/releases)
[![Tests](https://github.com/ajayarunachalam/llm-olmo/actions/workflows/test.yml/badge.svg)](https://github.com/ajayarunachalam/llm-olmo/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/ajayarunachalam/llm-olmo/blob/main/LICENSE)

LLM plugin support for the AI2's state-of-the-art open language models designed by scientists, for scientists. [Background on this project](https://github.com/allenai/OLMo). 

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).
```bash
llm install llm-olmo
```
If you have [uv](https://github.com/astral-sh/uv) installed you can chat with the model without any installation step like this:
```bash
uvx --with llm-olmo llm chat -m  olmo7b
```

```bash
uvx --with llm-olmo llm chat -m  olmo13b
```
## Usage

Once installed, run the OLMo2 7B model like this:
```bash
llm -m olmo7b 'Are dogs real?'
```
Or to chat with the model (keeping it resident in memory):
```bash
llm chat -m olmo7b
```

## Models to try
The following models all work well with this plugin:

- `OLMo 2 7B` (https://huggingface.co/allenai/OLMo-2-1124-7B)
- `OLMo 2 13B` (https://huggingface.co/allenai/OLMo-2-1124-13B)

## Model options

Commands for working with OLMo2 models:

- `download-7b`: Download the OLMo2 7B model
- `download-13b`: Download the OLMo2 13B model

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:
```bash
cd llm-olmo
python -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:
```bash
llm install -e '.[test]'
```
To run the tests:
```bash
python -m pytest
```

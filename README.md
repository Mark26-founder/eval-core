[README.md](https://github.com/user-attachments/files/30859395/README.md)
# EVAL-CORE

LLM regression testing and quality assurance framework

## Features

- **Regression Detection**: Track performance changes across different versions of your LLM application.
- **Multi-Provider Support**: Built-in support for multiple LLM providers (OpenRouter, Gemini).
- **Exact Match and LLM-as-Judge Scoring**: Evaluate outputs using exact matches or intelligent semantic grading.
- **Latency and Token Usage Metrics**: Keep tabs on API response times and token usage.
- **YAML Test Cases**: Clean, structured, and easy-to-read evaluation specifications.
- **CLI Interface**: Command-line utility to run, view, and analyze evaluations.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/eval-core.git
cd eval-core

# Create and activate virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate

# Unix/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Set up your API keys in a `.env` file:

```bash
# For OpenRouter
OPENROUTER_API_KEY=your_openrouter_key_here

# For Gemini
GEMINI_API_KEY=your_gemini_key_here
```

You can also set provider and model defaults:

```bash
EVAL_CORE_PROVIDER=openrouter
EVAL_CORE_MODEL=anthropic/claude-3-5-sonnet
```

### Running Evaluations

The CLI uses a `run` command to execute test suites:

```bash
# Set PYTHONPATH for src layout
# Windows PowerShell:
$env:PYTHONPATH="src"

# Windows CMD:
set PYTHONPATH=src

# Unix/macOS:
export PYTHONPATH=src

# Run an evaluation
python -m eval_core.cli.main run --suite examples/qa.yaml --scorer exact --provider openrouter --model anthropic/claude-3-5-sonnet
```

### Example YAML Test Suite

The `examples/` directory contains sample test suites:

- `examples/qa.yaml` - Question answering test cases
- `examples/summarization.yaml` - Text summarization test cases  
- `examples/instruction.yaml` - Instruction following test cases

Example test case format:

```yaml
cases:
  - id: qa_1
    input: "What is the capital of France?"
    expected: "Paris"
    description: "Simple factual question"
    tags: ["qa", "geography"]
```

### CLI Options

```bash
python -m eval_core.cli.main run --help
```

**Required:**
- `--suite` / `-s`: Path to YAML test suite file

**Optional:**
- `--provider` / `-p`: LLM provider (defaults to EVAL_CORE_PROVIDER env var)
- `--model` / `-m`: Model identifier (defaults to EVAL_CORE_MODEL env var)
- `--scorer`: Scoring method - `exact` (default) or `llm-judge`
- `--previous-report`: Path to previous report JSON for regression comparison
- `--output-report`: Path to save generated report JSON

### Running Tests

```bash
# Set PYTHONPATH
export PYTHONPATH=src  # Unix/macOS
$env:PYTHONPATH="src"  # Windows PowerShell

# Run test suite
pytest tests/ -v
```

## Project Structure

```text
eval-core/
├── src/
│   └── eval_core/
│       ├── __init__.py
│       ├── cases/         # Test cases schema and parsing logic
│       │   └── __init__.py
│       ├── runners/       # LLM evaluation execution runner
│       │   └── __init__.py
│       ├── scorers/       # LLM-as-judge and rule-based scorers
│       │   └── __init__.py
│       ├── reports/       # Report generation and export utilities
│       │   └── __init__.py
│       └── cli/           # Command line interface definition
│           └── __init__.py
├── examples/              # Usage examples and sample configurations
└── tests/                 # Unit and integration tests
```

## License

This project is licensed under the MIT License.

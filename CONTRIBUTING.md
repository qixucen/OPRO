# Contributing to Enhanced OPRO

Thank you for your interest in contributing to Enhanced OPRO! This document provides guidelines and instructions for contributing to the project.

## Development Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd enhanced-opro
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

## Running Tests

We use pytest for testing. To run the test suite:

```bash
python -m pytest
```

Ensure that:
- All tests pass
- Code coverage remains above 90%
- No new warnings are introduced

## Code Style Guidelines

1. Follow PEP 8 style guide
2. Use type hints for function arguments and return values
3. Write docstrings for all public classes and methods
4. Keep functions focused and modular
5. Use meaningful variable and function names

## Pull Request Process

1. Create a new branch for your feature/fix
2. Write tests for new functionality
3. Update documentation as needed
4. Ensure all tests pass and coverage requirements are met
5. Create a pull request with a clear description of changes

## Issue Reporting

When reporting issues, please include:

1. Python version
2. Operating system
3. Minimal reproducible example
4. Expected vs actual behavior
5. Error messages (if any)

## Code Review Process

1. All submissions require review
2. Changes must have tests
3. Documentation must be updated
4. Follow existing code style

## License

By contributing, you agree that your contributions will be licensed under the same terms as the original project.
# Contributing to Gaze-Aware Vision Foundation Model

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Research Contributions](#research-contributions)

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/gaze-aware-vision-foundation-model.git
   cd gaze-aware-vision-foundation-model
   ```

3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/HB-Innovates/gaze-aware-vision-foundation-model.git
   ```

## Development Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .  # Install in editable mode
```

### 3. Install Development Tools

```bash
pip install black flake8 isort mypy pytest pytest-cov
```

### 4. Set Up Pre-commit Hooks (Optional)

```bash
pip install pre-commit
pre-commit install
```

## Coding Standards

### Python Style Guide

We follow **PEP 8** with some modifications:

- **Line length**: 88 characters (Black default)
- **Imports**: Organized with isort
- **Type hints**: Encouraged for function signatures
- **Docstrings**: Google style for all public functions/classes

### Code Formatting

Before submitting, run:

```bash
# Format code
black .

# Sort imports
isort .

# Check style
flake8 .

# Type checking
mypy models/
```

### Example Function with Docstring

```python
def predict_gaze(
    eye_image: torch.Tensor,
    model: nn.Module,
    device: str = 'cpu',
) -> Tuple[float, float]:
    """Predict gaze direction from eye image.
    
    Args:
        eye_image: Eye region image tensor [1, 1, H, W]
        model: Trained gaze prediction model
        device: Device to run inference on
        
    Returns:
        Tuple of (yaw, pitch) in degrees
        
    Raises:
        ValueError: If image dimensions are incorrect
        
    Example:
        >>> image = torch.randn(1, 1, 64, 64)
        >>> yaw, pitch = predict_gaze(image, model)
    """
    # Implementation here
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=models --cov-report=html

# Run specific test file
pytest tests/test_gaze_predictor.py

# Run specific test
pytest tests/test_gaze_predictor.py::TestGazePredictor::test_forward_pass
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Use pytest fixtures for common setup
- Aim for >80% code coverage
- Test edge cases and error conditions

### Test Structure

```python
import pytest
import torch

class TestYourFeature:
    @pytest.fixture
    def model(self):
        """Fixture for model instance."""
        return YourModel()
    
    def test_basic_functionality(self, model):
        """Test description."""
        input_data = torch.randn(1, 10)
        output = model(input_data)
        assert output.shape == (1, 5)
```

## Submitting Changes

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- `feature/` - New features
- `bugfix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions

### 2. Make Changes

- Write clear, concise commit messages
- Keep commits atomic (one logical change per commit)
- Add tests for new features
- Update documentation as needed

### 3. Commit Message Format

```
<type>: <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

Example:
```
feat: Add temporal gaze prediction module

Implements LSTM-based temporal predictor for forecasting
gaze positions 1-5 frames ahead. Includes unit tests and
documentation.

Closes #42
```

### 4. Push Changes

```bash
git push origin feature/your-feature-name
```

### 5. Create Pull Request

- Go to GitHub and create a Pull Request
- Fill out the PR template
- Link relevant issues
- Request review from maintainers

### Pull Request Checklist

- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] No new warnings
- [ ] Backward compatible (or breaking changes documented)

## Research Contributions

### Adding New Models

1. Create module in appropriate directory:
   ```
   models/
   └── your_module/
       ├── __init__.py
       ├── model.py
       └── utils.py
   ```

2. Include:
   - Model implementation
   - Training script
   - Evaluation metrics
   - Documentation
   - Unit tests

### Adding Datasets

1. Create data loader in `data/` directory
2. Document dataset format and requirements
3. Provide download/preparation instructions
4. Add to experiments configuration

### Adding Experiments

1. Create experiment directory:
   ```
   experiments/
   └── your_experiment/
       ├── config.yaml
       ├── train.py
       └── README.md
   ```

2. Document:
   - Experiment goals
   - Hyperparameters
   - Expected results
   - Reproduction instructions

## Code Review Process

1. **Automated Checks**:
   - CI/CD pipeline runs automatically
   - All tests must pass
   - Code quality checks must pass

2. **Manual Review**:
   - At least one maintainer review required
   - Address all review comments
   - Iterate until approved

3. **Merge**:
   - Squash and merge (default)
   - Or rebase and merge for clean history

## Communication

- **Issues**: For bug reports, feature requests, questions
- **Discussions**: For general discussions and ideas
- **Pull Requests**: For code contributions

## Recognition

Contributors will be:
- Listed in README.md
- Credited in release notes
- Mentioned in relevant publications (if applicable)

## Questions?

Feel free to:
- Open an issue for questions
- Join discussions
- Reach out to maintainers

Thank you for contributing to advancing gaze tracking research! 🚀

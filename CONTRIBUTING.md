# Contributing Guide

Thank you for contributing to Coffee Bean Classification!

## How to Contribute

### Reporting Bugs
1. Check existing issues
2. Create new issue with:
   - Python version
   - OS and package version
   - Minimal reproducible example
   - Error messages

### Suggesting Features
1. Open issue with [Feature Request] tag
2. Describe use case
3. Provide examples

### Code Contributions

#### Setup
```bash
git clone https://github.com/yourusername/coffee_bean_classification.git
cd coffee_bean_classification
pip install -e ".[dev]"
```

#### Workflow
1. Create branch: `git checkout -b feature/your-feature`
2. Make changes
3. Add tests
4. Run tests: `pytest tests/ -v`
5. Format code: `black coffee_bean_classification tests`
6. Check linting: `flake8 coffee_bean_classification`
7. Commit: `git commit -m "Add: feature description"`
8. Push: `git push origin feature/your-feature`
9. Open Pull Request

## Code Style

### Python Style
- Follow PEP 8
- Use Black for formatting (line length 100)
- Type hints for all functions
- Google-style docstrings

### Example
```python
def train_model(
    config: TrainingConfig,
    model_name: str,
    verbose: int = 1
) -> tf.keras.callbacks.History:
    """
    Train a single model.
    
    Args:
        config: Training configuration
        model_name: Name of model architecture
        verbose: Verbosity level (0, 1, or 2)
        
    Returns:
        Training history object
        
    Example:
        >>> config = TrainingConfig()
        >>> history = train_model(config, 'resnet50')
    """
    # Implementation
    pass
```

## Testing

### Writing Tests
```python
def test_model_creation():
    """Test that model can be created."""
    config = ModelConfig(architecture='resnet50', num_classes=4)
    model = ModelFactory.create('resnet50', config)
    assert model is not None
```

### Run Tests
```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=coffee_bean_classification

# Specific file
pytest tests/test_models.py -v
```

### Coverage Requirements
- Maintain >90% coverage
- Test edge cases
- Test error handling

## Documentation

### Update Docs
- Add docstrings to new functions/classes
- Update README if needed
- Update API reference
- Add examples

### Documentation Style
```python
class MyClass:
    """
    Brief one-line description.
    
    Longer description with more details about
    the class functionality.
    
    Attributes:
        attr1: Description of attr1
        attr2: Description of attr2
        
    Example:
        >>> obj = MyClass()
        >>> obj.method()
    """
```

## Commit Messages

### Format
```
Type: Brief description (50 chars max)

Detailed explanation of changes (if needed).

Fixes #123
```

### Types
- Add: New feature
- Fix: Bug fix
- Docs: Documentation changes
- Style: Code formatting
- Refactor: Code refactoring
- Test: Add/update tests
- Chore: Maintenance tasks

## Pull Request Process

1. **Title**: Clear and descriptive
2. **Description**: 
   - What changed
   - Why it changed
   - How to test
3. **Tests**: Include test results
4. **Documentation**: Update if needed
5. **Review**: Address reviewer comments

## Code Review Guidelines

### For Contributors
- Be open to feedback
- Respond to comments
- Make requested changes

### For Reviewers
- Be constructive
- Explain reasoning
- Suggest alternatives

## Getting Help

- **Questions**: Open GitHub Discussion
- **Bugs**: Open GitHub Issue
- **Security**: Email maintainers directly

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be added to:
- README contributors section
- CHANGELOG for their contributions

Thank you for making this project better! 🎉

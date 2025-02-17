### Required: Pre-commit Setup
*** IMPORTANT: Pre-commit hooks are mandatory for all contributors.**

Before contributing, you **must** set up pre-commit hooks. These hooks run automatically before each commit to ensure code quality and consistency across all contributions. If you try to commit without setting up pre-commit, your commits will fail. This is intentional to maintain code quality.

Follow these steps to set up pre-commit:
1. Install pre-commit:
```sh
pip install pre-commit
```
2. Install the git hooks:
```sh
pre-commit install
```
The `.pre-commit-config.yaml` file is already included in the repository with the following hooks configured:
- Code formatting with Black
- Import sorting with isort
- Linting with Flake8
- Basic file checks (trailing whitespace, YAML validity, etc.)
- Security checks (private keys, merge conflicts)

3. Verify your setup:
```sh
git commit --allow-empty -m "test: verify pre-commit"
```
This should run all pre-commit hooks. If any hook fails, fix the issues before continuing.

### Troubleshooting Pre-commit
If you encounter issues:
1. **Hooks not running**: Ensure you've run `pre-commit install`
2. **Hook installation fails**: Try `pre-commit clean` followed by `pre-commit install`

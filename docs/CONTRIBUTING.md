# Contributing to torch-rbln

Welcome! Thank you for your interest in contributing to **torch-rbln**, the [Rebellions Extension for PyTorch](https://github.com/RBLN-SW/torch-rbln). This project adapts [PyTorch](https://pytorch.org/) to run on **Rebellions' NPU** (RBLN). As an open-source project, we rely on community support and involvement to improve it. This document outlines our contribution process, coding guidelines, and community standards.

We value transparency, collaboration, and a safe environment for contributors. All contributions are expected to follow these guidelines.

---

## Getting Started

### Contributors

1. **Fork the repository** and create your branch from `dev`.
2. Make your changes with clear and concise commits.
3. Ensure that your code follows the style and linting rules (see [Linting](LINTING.md)).
4. If relevant, update or add new tests and documentation.
5. Open a pull request targeting `dev` with a detailed description of your changes.

### Core Contributors & Collaborators

1. **Create your branch** from `dev` and work on branches within the repository.
2. Make your changes with clear and concise commits.
3. Ensure that your code follows the style and linting rules (see [Linting](LINTING.md)).
4. If relevant, update or add new tests and documentation.
5. Open a pull request targeting `dev` with a detailed description of your changes.

All contributors must use **English** for issues, comments, and code.

---

## Development Setup

- **Linting:** See [Linting](LINTING.md) for `lintrunner` and pre-commit setup.
- **Tests:** See [Test Guide](TEST_GUIDE.md) for Python and C++ test commands.
- **CI/CD:** See [Workflows](WORKFLOWS.md) for the automated testing and deployment pipeline.

---

## How You Can Contribute

One of the best ways to contribute is by creating issues — whether you're reporting a bug, suggesting a new idea, implementing a feature, or asking a question.

If you've found something that needs attention or improvement, we'd love to hear from you. Your input helps make **torch-rbln** better for everyone.

When creating an issue, please provide as much detail as possible and select the appropriate label to help us triage and respond efficiently.

### General Issue

These issues are used to discuss general suggestions, requests, bug reports, and other topics.

- **proposal:** Suggest enhancements or new functionality that would benefit torch-rbln.
- **request:** Request a specific development task that you think should be implemented.
- **bug:** Help us identify and fix issues by reporting bugs with clear reproduction steps.
- **question:** Ask general questions about using the project, understanding behavior, or seeking clarification. Ideal for newcomers or anyone unsure about how something works.
- **discussion:** Start open-ended conversations about design decisions, optimization, or features. Useful for gathering community feedback before moving to a proposal.
- **help wanted:** Highlight tasks where contributor support is requested. Often used in combination with other labels like bug or question.

### Development-related Issue

These issue types represent development tasks that are typically addressed through pull requests.

- **feature:** Develop a new capability or functionality in the codebase. Should be scoped and accompanied by acceptance criteria or use cases if possible.
- **core:** Changes that impact core components such as the RBLN backend, device layer, C++ extension (`torch_rbln/csrc`), op registration, or `torch.compile` integration. These usually require in-depth review and testing.
- **fix:** Tracks the resolution of known bugs.
- **perf:** Implement improvements focused on performance (e.g., latency, memory, throughput). Include benchmarks or measurement methodology if available.
- **refactor:** Improve readability, maintainability, or consistency without altering external behavior. Includes renaming, modularization, or dependency cleanup.
- **docs:** Improve or add documentation. Includes README, usage guides, code comments, and tutorials. See [docs](docs/) for existing guides.
- **other:** Any development-related task that doesn't fit the above categories. Use sparingly; consider proposing a new label if recurring themes emerge.

Please choose labels appropriately when opening an issue.

---

## Pull Request Guidelines

All pull requests **must**:

- Have a corresponding issue (refer to Development-related Issue above when applicable).
- Include a clear title following [**Conventional Commits v1.0**](https://www.conventionalcommits.org/en/v1.0.0).
- Contain the following in the description:
  - Purpose and detailed explanation
  - Related issue number
  - Affected modules (e.g., Backend, Device, Ops/Kernels, torch.compile, C++ extension, Memory, Build/Tools, Docs)
  - Type of change (use Labels): Feature / Bug Fix / Refactor / Docs / ...
  - How to test and a summary of expected results

Individual commit messages in PR branches do not need to strictly follow Conventional Commits, but should remain readable and descriptive.

---

## Merge Policy

All of the following must be satisfied for a PR to be merged:

- All CI tests must pass (see [CI/CD Workflows](WORKFLOWS.md) for details)
- At least one approval from the relevant team
- **Squash and merge** only

### Automated Testing Pipeline

The following events trigger automated workflows on RBLN NPU hardware:

- **PRs to `dev`** — The CI workflow runs linting and `test_set_ci`-marked tests for fast feedback.
- **PRs to `main`** — The Release workflow runs linting and the full test suite (excluding experimental and performance tests) for pre-release validation.
- **Version tags (`v*`)** — The CD workflow builds and publishes release artifacts.

All tests execute on remote infrastructure with access to physical NPU devices. Results appear as PR status checks. See [Workflows](WORKFLOWS.md) for the full architecture.

---

## Thank You

Thank you for taking the time to contribute to **torch-rbln**. Whether you're submitting a pull request, opening an issue, improving documentation, or asking thoughtful questions — your effort helps strengthen the project and the community. We believe great software is built in the open, by people who care. We're excited to have you on board and look forward to your contributions.

<!--
Thank you for contributing to torch-rbln!
This template will help us understand and review your pull request efficiently.
Please fill out all required sections. You may delete optional ones if not applicable.
see: docs/CONTRIBUTING.md
-->

## Summary of Changes

<!--
**PR Title** uses the **Conventional Commits v1.0** format.
see: https://www.conventionalcommits.org/en/v1.0.0/
-->

<!-- Describe your changes in detail -->

> *What does this PR do? What feature, fix, or improvement does it bring?*

---

## Related Issues / Tickets

<!--
All pull requests must have a corresponding issue.
Use "Resolves/Fixes/Closes/Related to #<issue_number>" to auto-link or close the issue when merged.
see: docs/CONTRIBUTING.md#pull-request-guidelines
-->

* Resolves #
* Related to #

---

## Type of Change

<!--
Select the type that best describes your PR. This should match the issue label.
see: docs/CONTRIBUTING.md#development-related-issue
-->

- [ ] `feature` — New capability or functionality
- [ ] `core` — Changes to RBLN backend, device layer, C++ extension, op registration, or `torch.compile` integration
- [ ] `fix` — Bug fix
- [ ] `perf` — Performance improvement
- [ ] `refactor` — Code improvement without behavior change
- [ ] `docs` — Documentation only
- [ ] `other` — Other (describe below)

---

## Affected Modules

<!--
Check all modules affected by this change.
-->

- [ ] Backend
- [ ] Device
- [ ] Ops/Kernels
- [ ] `torch.compile`
- [ ] C++ extension (`torch_rbln/csrc`)
- [ ] Memory
- [ ] Build/Tools
- [ ] Docs

---

## How to Test (if applicable)

<!--
Describe the steps to verify your changes. Include expected results.
see: docs/TEST_GUIDE.md
-->

1. Run `...`
2. Verify output: `...`
3. Edge case tested: `...`

---

## Screenshots / Logs (if applicable)

<!-- Add before/after screenshots, terminal output, or logs -->

---

## Checklist

<!--
The PR will only be reviewed and considered for merge if the following are satisfied.
-->

* [ ] PR title follows [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) format
* [ ] This PR is linked to a corresponding issue — see [Contributing to torch-rbln](docs/CONTRIBUTING.md)
* [ ] Linting passes — see [Linting](docs/LINTING.md)
* [ ] All CI tests pass — see [CI/CD Workflows](docs/WORKFLOWS.md)
* [ ] The test method is described, and the expected result is clearly stated (if applicable)
* [ ] Relevant documentation has been updated (if applicable)

---

## Notes

<!-- Anything reviewers should pay extra attention to? -->

---

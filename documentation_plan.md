# Documentation Structure Plan

This file outlines the structure of the project documentation.

- **`index.rst`** (The main landing page)
    - Links to all major sections.

- **User Guides (`guides/`)**
    - `installation.md`: How to install the package.
    - `quickstart.md`: A minimal, runnable example.
    - `choosing_a_test.md`: Guidance on selecting the right test for different data types.
    - `interpreting_results.md`: How to understand p-values and test outputs.

- **Theoretical Foundations (`theory/`)**
    - `what_is_ci.md`: Formal and intuitive explanation of Conditional Independence.
    - `taxonomy_of_tests.md`: Categorization of the tests (e.g., correlation-based, kernel-based).

- **Conditional Independence Tests (`tests/`)**
    - `index.md`: A menu that links to each individual test page.
    - `fisher_z_test.md`: Detailed page for the Fisher-Z test.
    - `spearman_test.md`: (To be created)
    - `gsq_test.md`: (To be created)
    - `... (and so on for each test)`

- **API Reference (`api/`)**
    - `modules.rst`: Auto-generated from docstrings by `sphinx-apidoc`.

- **Development (`/`)**
    - `contributing.md`: Guidelines for contributing to the project.
    - `changelog.md`: A log of changes for each version release. 
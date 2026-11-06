# Coding Guidelines




Write Python code that is simple, readable, and easy to maintain.




## Principles

- Prefer the simplest solution that meets the requirements.

- Avoid overengineering and premature abstraction.

- Use clear, descriptive variable and function names.

- Keep functions focused on a single responsibility.

- Prefer straightforward control flow over clever shortcuts.

- Avoid unnecessary classes, wrappers, and helper functions.

- Do not introduce design patterns without a concrete need.

- Follow PEP 8 and existing project conventions.




## Code Structure

- Prefer functions over classes when no persistent state is needed.

- Separate concerns only when it improves readability.

- Avoid excessive file splitting and deeply nested abstractions.

- Use established libraries instead of reinventing functionality.




## Comments

- Explain non-obvious reasoning, not obvious syntax.

- Use concise docstrings for important functions.

- Avoid redundant comments and excessive documentation.




## Refactoring

- Preserve existing behavior and public interfaces.

- Make small, targeted changes.

- Remove redundant code without introducing new complexity.

- Run relevant tests after refactoring.
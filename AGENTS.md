## Development environment

- Use the `3dgeer` Conda environment whenever a project command requires the
  project's Python environment.

## Testing and validation

- Do not add, modify, or expand tests unless I explicitly ask for tests.
- Do not run the full test suite by default.
- Use the minimum targeted validation necessary for the specific change.
- Prefer code inspection and reasoning first, then cheap validation such as syntax,
  import, type, or narrowly targeted execution when useful.
- Do not create tests merely to demonstrate that a change works.
- Avoid running tests for low-risk changes such as comments, documentation,
  formatting, renames, simple configuration edits, or straightforward mechanical changes.
- Do not rebuild CUDA extensions after every source edit unless the change requires it.
- Do not run GPU-heavy validation, benchmarks, training jobs, rendering evaluations,
  dataset-dependent jobs, or long-running builds unless I explicitly request them.
- Before running broad, expensive, or time-consuming validation, ask first.
- For research and experimental code, prioritize fast iteration and focused implementation
  over production-style test coverage or unnecessary defensive abstractions.

## Implementation style

- Keep changes minimal and scoped to the requested task.
- Avoid unrelated refactors, cleanup, abstractions, or dependency changes.
- Reuse existing project patterns and utilities where practical.
- Do not introduce new infrastructure unless it is necessary for the requested change.
- When the requested behavior is clear, implement it directly rather than over-engineering.

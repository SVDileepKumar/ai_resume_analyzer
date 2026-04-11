# Agent notes

## Codebase context (`/init`)

- **Skill:** `.cursor/skills/init-codebase-context/SKILL.md`
- **Output:** `CODEBASE_CONTEXT.md` at the repository root (when generated)
- **When to use:** User says `/init` or asks to initialize/refresh codebase context, create or update `CODEBASE_CONTEXT.md`, map architecture for LLMs, or bootstrap a dense repo overview before multi-file work.

Cursor loads this via `.cursor/rules/init-codebase-context.mdc` (agent-requested via description; not file-gated so `/init` works before `CODEBASE_CONTEXT.md` exists).

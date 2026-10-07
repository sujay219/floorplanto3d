# Home AI

This repository is the foundation for an AI-powered home interior design system.

## Architecture

The system is designed around a canonical scene model and deterministic scene operations.

```text
User:
"Add a drawer beside the sink"

Future AI output:
SceneOperation
        ↓

SceneEngine
        ↓

Canonical Scene
        ↓

Layout Engine
        ↓

Resolved Scene
        ↓

Babylon.js
```

The API and AI layers generate structured scene operations instead of directly constructing final 3D geometry. Those operations are validated by a deterministic scene engine and later by the layout engine, which resolves spatial constraints without relying on LLM opinion.

The canonical scene object is the shared TypeScript source of truth. It is serialized as canonical JSON and consumed by the renderer and downstream tooling. Babylon.js is intentionally limited to rendering the validated scene; it is not the system-of-record for scene state.

Parametric components will eventually generate their geometry from structured component specifications rather than ad hoc AI-created coordinates. This keeps the system deterministic, testable, and easy to extend.

## Project structure

- `web_app`: React + Vite frontend shell
- `services/ai`: future LangChain / LangGraph service boundary
- `packages/scene-schema`: canonical scene model, validation, serialization, and history
- `packages/components`: component metadata contracts
- `packages/geometry`: deterministic spatial math utilities
- `packages/layout-engine`: validation and placement contracts
- `packages/renderer`: Babylon.js adapter layer
- `schemas`: canonical JSON schema definition

## Core principles

1. The API / AI layer generates structured scene operations.
2. The scene engine applies immutable, auditable state changes.
3. The canonical Scene JSON is the source of truth.
4. Babylon.js is only responsible for rendering.
5. The layout engine handles spatial validity, while the AI stays focused on semantic intent.

## Current scene-model flow

```text
AI / semantic intent
  ↓
SceneOperation[]
  ↓
SceneEngine
  ↓
Validated canonical Scene
  ↓
Layout Engine
  ↓
Babylon.js Renderer
```

## Getting started

```bash
pnpm install
pnpm build
pnpm test
pnpm typecheck
```

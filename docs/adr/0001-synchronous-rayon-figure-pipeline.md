# ADR 0001: Synchronous Rayon Pipeline for Vision and Figure Analysis

## Status
Accepted

## Date
2026-10-20

## Context
The `pdf2md` codebase operates on a multi-threaded synchronous architecture using Rayon and `reqwest::blocking` for page extraction and OCR. Conversely, the external `Fig2md` crate was implemented using `tokio` asynchronous runtime and async `reqwest`.

When integrating `Fig2md`'s figure classification, Mermaid diagram synthesis, and image preprocessing into `pdf2md`, we had to determine whether to rewrite `pdf2md` around an asynchronous Tokio event loop or keep the synchronous Rayon-based execution model.

## Decision
Retain `pdf2md`'s synchronous execution model using Rayon worker threads and `reqwest::blocking` for all vision model interactions (figure detection, archetype classification, and Mermaid generation). 

Image preprocessing (Sauvola binarization and morphological despeckle) is CPU-bound and naturally parallelizes across Rayon workers alongside OCR page processing without async scheduling overhead.

## Consequences
### Positive
- Zero runtime impedance mismatch: avoids mixing Tokio runtimes or `block_on` bridges inside Rayon worker pools.
- Simpler error handling, stack traces, and deterministic thread management.
- Smaller binary size and faster compile times by avoiding Tokio async ecosystem dependencies.
- Reuses existing backend clients (`src/backends/qwen.rs`, `src/backends/gemini.rs`, `src/backends/gemma.rs`) without duplication.

### Negative / Trade-offs
- Each concurrent API request occupies an OS thread rather than a lightweight async task. Given standard CLI concurrency limits (typically 2–8 threads corresponding to CPU cores / rate limits), this footprint is negligible.

## Alternatives Considered
- **Full Async Migration to Tokio**: Refactor `pdf2md` entirely to `async`/`await`. Rejected due to extensive blast radius across all existing pipeline stages, PDF rendering bindings, and file I/O.
- **Embedded Tokio Runtime inside Rayon Workers**: Spawn a localized Tokio runtime for vision requests. Rejected due to unnecessary complexity and memory overhead.


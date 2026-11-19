# Historical research snapshot

`research_snapshot/` preserves the original source layout, including duplicated models and historical reports. These are frozen research artifacts, not the active implementation.

The exact pre-refactor Git revision is tagged `pre-layout-refactor-2026-11-19`. Use a checkout of that tag to resume original runs with their original paths and provenance. Moving the snapshot does not make every historical command portable; external data and manifests remain required.

The active implementation is `src/music_separation/`. New runs use a new source identity. Original source hashes and provenance have not been rewritten.

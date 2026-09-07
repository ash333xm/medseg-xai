# Contributing & Git Branching Guidelines: MedSeg-XAI

## Branching Strategy (Gitflow / Trunk-Based Hybrid)

To maintain stability, reproducibility, and rigorous safety auditing across the 12-week development lifecycle of **MedSeg-XAI**, the repository follows a strict multi-branch topology:

```
main (Production / Stable Releases)
  └── develop (Integration & CI Staging)
        ├── feature/m1-model-arch        <-- Member 1 (Lead: Model Architecture & Inference)
        ├── feature/m2-xai               <-- Member 2 (Lead: Explainability TMME & Rollout)
        ├── feature/m3-mprt-safety       <-- Member 3 (Lead: Safety Auditing & Live MPRT)
        └── feature/m4-data-fullstack    <-- Member 4 (Lead: Preprocessing & Full-Stack UI)
```

### 1. Branch Definitions
- **`main`**:
  - The production release branch.
  - Strictly protected: Direct pushes are blocked.
  - Requires passing automated CI (`pytest`, linting, Docker build validation) and at least 2 team approvals.
  - Tagged semantically for milestones (e.g., `v1.0.0-phase1`, `v2.0.0-phase3`).
- **`develop`**:
  - Central integration branch for ongoing sprints.
  - Nightly integration tests execute here.
- **`feature/m1-*` (Member 1)**:
  - Scope: MedSAM-2 instantiation, Hiera-Large checkpoint loading, memory optimization (`model_loader.py`), 3D pseudo-video resampling, zero-shot inference engine (`inference.py`), and dual-pass VRAM lifecycle (`vram_manager.py`).
- **`feature/m2-*` (Member 2)**:
  - Scope: PyTorch forward hook activation extraction (`xai_hooks.py`), TMME engine, attention rollout recursion (`rollout.py`).
- **`feature/m3-*` (Member 3)**:
  - Scope: Live MPRT auditing loop (`live_mprt_auditor.py`), mathematical metrics (`metrics.py`), stress testing corpus.
- **`feature/m4-*` (Member 4)**:
  - Scope: NIfTI/DICOM ingestion, HU windowing, Z-score normalization, Cloud GPU RunPod infrastructure, React clinical dashboard.

### 2. Commit Message Standards (Conventional Commits)
All commit messages must follow the Conventional Commits format:
```
<type>(<scope>): <short summary>

[optional body]
[optional footer]
```

- **Types**:
  - `feat`: A new feature (e.g., `feat(inference): add 3d memory attention propagation`)
  - `fix`: A bug fix (e.g., `fix(vram): prevent memory leak during randomized pass`)
  - `refactor`: Code change that neither fixes a bug nor adds a feature
  - `perf`: Performance improvement (e.g., `perf(loader): implement torch mmap streaming`)
  - `test`: Adding or correcting tests (`test(resampler): add tests for (B, T, C, H, W) tensor shapes`)
  - `docs`: Documentation changes (`docs(api): document RunPod persistent volume mounts`)

### 3. Pull Request (PR) Quality Gate Checklist
Before any PR can be merged into `develop` or `main`:
1. [ ] All unit tests pass: `pytest tests/ -v`
2. [ ] VRAM lifecycle verified without memory leaks
3. [ ] No hardcoded absolute local paths; all volume mounts reference `$RUNPOD_VOLUME_PATH` or `/runpod-volume`
4. [ ] Code formatted with `black` and checked with `flake8`
5. [ ] At least one code review approval from an adjacent module lead

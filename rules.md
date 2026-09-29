# Development & Coding Rules: MedSeg-XAI

## 1. Core Engineering Principles
- **No Live PyTorch in UI**: The Streamlit dashboard must never `import torch` or load `SamModel`. All inference and randomization must be precomputed and saved as static artifacts.
- **Scientific Honesty First**: Never report AI-guessed or interpolated numbers. Display actual measured scores. Explicitly surface Case 05 under-segmentation (`0.641` Dice) without suppression.
- **Explicit Labeling**: Saliency heatmaps must be strictly labeled as "Decoder Attention Map (Hooks)". Never use "TMME" unless the multimodal propagation architecture is explicitly implemented and validated.

## 2. Code Quality & Standards
- Python 3.11 compatibility.
- Type annotations across all function signatures.
- Clean separation between data ingestion, numerical metrics, and UI rendering.
- Resilient fallback logic for missing artifacts with descriptive error alerts.

## 3. UI/UX Style Rules
- **Prohibited Aesthetics**:
  - No purple gradients or complex multi-color gradient fills.
  - No whimsical or video-game style buttons.
  - No emoji icons in section headers, buttons, or formal clinical text.
  - No em dashes (—). Use standard hyphens (-) for compound words.
  - No fake social proof, testimonials, or user counters.
  - No marketing slop or generic Lorem Ipsum text.
- **Permitted Palette**:
  - Primary: Slate gray (`#0f172a`, `#1e293b`), charcoal (`#334155`), sapphire blue (`#0284c7`), deep forest green (`#059669`).
  - Background: Plain white (`#ffffff`) or light clinical gray (`#f8fafc`).
  - Accent / Error: Brick red (`#dc2626`) for failure states and predicted outlines.
  - Success / Ground Truth: Kelly green (`#16a34a`) for ground truth outlines and audit passes.

## 4. Git & Release Workflow
- Commits must use standard semantic prefixes: `feat:`, `fix:`, `docs:`, `chore:`.
- Production builds must exclude large binary checkpoint weights (`*.pt`, `*.pth`) and `node_modules`.
- Precomputed artifacts in `outputs/` must be versioned alongside the code.

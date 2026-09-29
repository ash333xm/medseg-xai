# Development & Engineering Rules: MedSeg-XAI

## 1. Core Engineering Principles
1. **Zero Live PyTorch in UI**: The Streamlit diagnostic dashboard must NEVER `import torch`, `torchvision`, `monai`, or instantiate `SamModel`. All inference and randomization must be precomputed and saved as static artifacts.
2. **Scientific Honesty First**: Never report AI-guessed or interpolated numbers. Display actual measured scores. Explicitly surface Case 05 under-segmentation (`0.641` Dice) without suppression.
3. **Explicit Labeling**: Saliency heatmaps must be strictly labeled as **"Decoder Attention Map (Hooks)"**. Never use "TMME" unless the multimodal propagation architecture is explicitly implemented and validated.

---

## 2. Mandatory UI/UX Instruction Set & Prohibitions

### 2.1 Strictly Prohibited Aesthetics & Elements
- **NEVER use purple gradients or complex gradients of any kind**. Use a clean, professional color palette (e.g., slate gray, charcoal, sapphire blue, or deep green) against a plain white or light cool slate background.
- **NEVER use spaceship-themed, whimsical, or over-the-top buttons/UI elements**.
- **NEVER use fake reviews, fake user metrics, or placeholder social proof**.
- **NEVER use fake hero text or promotional marketing language**.
- **NEVER use emoji icons as UI elements, tab titles, or section headers**.
- **NEVER use em dashes (—)**. Use standard hyphens (-) exclusively for compound words and grammatical breaks.
- **NEVER use over-the-top scroll animations, cursor animations (e.g., trails), or jarring page transitions**.
- **NEVER use AI-generated "slop" photos or generic stock images**.
- **NEVER use AI-generated "slop" copy**. All text must be specific, logical, clinically coherent, and relevant to the user's context. Avoid generic Lorem Ipsum.
- **NEVER use fake customer/user counters or ticking social proof tickers**.

### 2.2 Mandatory Inclusions & Structure
- **ALWAYS ensure a custom domain connection configuration placeholder**:
  Provide an enterprise header element housing a clear custom domain indicator (`pacs.medseg-xai.internal`) with SSL active status.
- **ALWAYS include a specific, custom favicon**:
  Embed a custom medical diagnostic cross vector SVG via Data URI in `st.set_page_config()`.
- **ALWAYS remove default framework watermarks**:
  Suppress default Streamlit chrome, including the hamburger menu (`#MainMenu`), footer, deploy button (`.stDeployButton`), status decoration bar, and toolbar.
- **ALWAYS provide a dedicated placeholder and content area for a complete Privacy Policy page**:
  Render the complete HIPAA Safe Harbor De-Identification Protocol and Zero-VRAM PACS Isolation Architecture with zero-indentation clean HTML to prevent raw code block leakage.
- **ALWAYS provide a dedicated placeholder and content area for a complete Terms and Conditions page**:
  Render the complete Software-as-a-Medical-Device (SaMD) Research Use Only (RUO) and Clinical Decision Support disclaimer with zero-indentation clean HTML.

---

## 3. Code Quality & Implementation Standards
- **Python 3.11 Compatibility**: Strict adherence to standard library and verified package APIs (Streamlit 1.64+, NumPy, Pandas, Matplotlib, Pillow).
- **Contiguous HTML Card Architecture**: Prevent Streamlit DOM element separation by constructing all visual cards (header, base64 image, legend, metadata) as single contiguous HTML strings.
- **Zero-Indentation HTML Streams**: When injecting multiline HTML strings into `st.markdown(..., unsafe_allow_html=True)`, ensure zero leading whitespace on any line. Any line with 4 or more leading spaces will be erroneously converted into a Markdown code block (`<pre><code>`).
- **Resilient Fallback Logic**: All data loading calls (`load_case_data`, `load_metrics_manifest`, `load_scrambled_heatmap`) must check file existence and return graceful fallbacks rather than crashing.

---

## 4. Git & Release Workflow
- Commits must use standard semantic prefixes: `feat:`, `fix:`, `docs:`, `chore:`.
- Checkpoints and large raw datasets (`*.pth`, `*.pt`, `Task01_BrainTumour.tar`) must remain excluded via `.gitignore`.
- Precomputed artifacts in `outputs/` are versioned alongside the code for immediate out-of-the-box local execution.

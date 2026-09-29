# UI/UX Design System Specification: MedSeg-XAI

## 1. Design Direction: Liquid Glass Clinical PACS
MedSeg-XAI implements a modern **Liquid Glass (Frosted Glassmorphism) Radiology PACS** design language. It is engineered for clinical precision, high optical contrast, and executive academic review, avoiding consumer gimmicks, purple gradients, or template aesthetics.

---

## 2. Color Palette & Chromatic Foundations
The color system provides clear separation between background depth, frosted translucent surfaces, and clinical diagnostic overlays:

| Token | Hex / RGBA Code | Role & Usage |
|---|---|---|
| **Canvas Background** | `#edf2f7` to `#e2e8f0` | Base clinical cool-slate depth gradient with subtle radial mesh |
| **Glass Card Fill** | `rgba(255, 255, 255, 0.72)` | Frosted translucent container surface |
| **Glass Card Border** | `rgba(255, 255, 255, 0.90)` | Top/side frosted refraction border |
| **Primary Text** | `#0f172a` (Slate 900) | Main headings, case identifiers, primary metric values |
| **Secondary Text** | `#475569` (Slate 600) | Subtitles, clinical pathology descriptors, metadata |
| **Medical Sapphire** | `#0284c7` (Sky 600) | Active segmented tab pill, brand icons, primary focus rings |
| **Sapphire Accent** | `#0369a1` (Sky 700) | Custom domain badge text, secondary highlight pills |
| **Ground Truth Green** | `#16a34a` (Emerald 600) | Expert annotation contour lines, audit pass badges |
| **MedSAM Contour Red** | `#dc2626` (Red 600) | Model predicted segmentation contour lines, failure badges |
| **Warning Amber** | `#fcd34d` / `#92400e` | Case 05 under-segmentation clinical audit notice container |
| **Control Border** | `#94a3b8` / `#cbd5e1` | High-contrast boundaries for selectboxes, popovers, and sliders |

---

## 3. Liquid Glass Architecture & Optical Depth
To prevent flat "white-on-white" washout, the interface applies physical glassmorphism depth principles:

### 3.1 Background Depth Canvas
The root view container (`[data-testid="stAppViewContainer"]`) is styled with fixed multi-point radial gradients transitioning between ice-blue (`rgba(219, 234, 254, 0.8)`), clean clinical slate (`#e2e8f0`), and titanium gray (`#cbd5e1`). This provides the optical contrast required for frosted glass cards to reflect light and cast shadows.

### 3.2 Frosted Glass Containers (`.glass-card`)
- **Translucency**: `background: rgba(255, 255, 255, 0.72)`
- **Backdrop Filter**: `backdrop-filter: blur(20px) saturate(160%)`
- **Border Sheen**: `border: 1px solid rgba(255, 255, 255, 0.90)`
- **Layered Elevation**: `box-shadow: 0 10px 30px -5px rgba(15, 23, 42, 0.08), 0 4px 6px -2px rgba(15, 23, 42, 0.03), inset 0 1px 1px 0 rgba(255, 255, 255, 0.95)`
- **Corner Curvature**: `border-radius: 16px`

### 3.3 High-Contrast Form Controls (Solving Washout)
To ensure interactive controls never blend into white surfaces:
- All Streamlit selectboxes (`div[data-baseweb="select"] > div`) feature a solid `1.5px solid #94a3b8` border, white opaque background (`rgba(255, 255, 255, 0.95)`), and dark charcoal bold text (`#0f172a`).
- Hover and focus states display a dedicated **Medical Sapphire ring** (`#0284c7`, `box-shadow: 0 0 0 2px rgba(2, 132, 199, 0.2)`).
- Dropdown menus (`div[data-baseweb="popover"]`) use high-elevation frosted glass with crisp row separators and soft blue hover highlights (`#e0f2fe`).

### 3.4 Unified Base64 In-Memory Card Pattern
Streamlit's React rendering engine automatically wraps consecutive `st.markdown()` and `st.image()` calls into sibling DOM elements, causing unclosed HTML divs to self-close and break cards apart.
To guarantee unbroken glass cards:
- Each clinical panel (Panel 1, Panel 2, Panel 3, MPRT Heatmap A/B, and Curves) is composed as a **single, contiguous HTML string**.
- Rendered images are encoded into in-memory **base64 Data URIs** (`data:image/png;base64,...`) and embedded directly inside `<div class="glass-card">...</div>`.
- This ensures the header, image frame, legend, metric pill, and metadata are permanently unified inside a single frosted glass surface.

---

## 4. Navigation & Layout Hierarchy

### 4.1 Enterprise Header Bar
Positioned at the top of the viewport:
- **Left**: Custom Vector Medical Diagnostic Cross SVG with `MedSeg-XAI Clinical Studio` title.
- **Center / Right**: Dedicated Enterprise Custom Domain Connection Pill:
  `Custom Domain: pacs.medseg-xai.internal [Configured - SSL Active]`
- **Status Indicator**: `Zero-VRAM Safe Enclave | HIPAA Safe Harbor Verified`

### 4.2 Segmented Floating Tab Bar
The four primary views are housed in a floating frosted glass pill dock:
1. `Clinician View & Audit Gate` (Main triple-view PACS viewer and MPRT gate)
2. `Cohort Benchmark & Metric Pivot` (Full 5-case comparative study and mathematical analysis)
3. `Data Governance & HIPAA Policy` (Full HIPAA de-identification protocol)
4. `Terms & Conditions of Decision Support` (SaMD research and investigational use terms)

Active tabs transition dynamically to a solid **Medical Sapphire Blue (`#0284c7`)** pill with white text and a soft elevation shadow (`box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35)`).

### 4.3 Triple-Panel Clinician Viewer
- **Panel 1 (Raw Input)**: Axial T1ce slice with bounding box prompt overlaid in amber (`#eab308`).
- **Panel 2 (Segmentation)**: Anti-aliased vector contours showing Expert Ground Truth in Green (`#16a34a`) and MedSAM Prediction in Red (`#dc2626`). Displays clean Dice score badge. If Case 05 is selected, an amber warning callout details the infiltrative frontal under-segmentation.
- **Panel 3 (Heatmap Attribution)**: Clean decoder cross-attention overlay strictly labeled **"Decoder Attention Map (Hooks)"**.

### 4.4 Verification Module (MPRT Sanity Check)
- Side-by-side comparison cards: **Heatmap A (Intact Model)** vs **Heatmap B (Stage 4 Full Decoder Scrambled)**.
- Sanity Check Verdict Banner in translucent emerald green (`#f0fdf4`, border: `#86efac`, text: `#166534`).
- Three liquid glass metric cards displaying Spearman Correlation, Randomized Model Dice, and Pre-registered SSIM.

---

## 5. Strict Prohibitions & Governance
1. **Zero Emojis**: Never use emoji glyphs in UI section headers, metric cards, tab titles, buttons, or legal text.
2. **Zero Em Dashes**: Never use em dashes (`—`). Standard hyphens (`-`) are used exclusively for compound words and grammatical breaks.
3. **Zero Purple Gradients**: No purple, magenta, or neon gradients. The palette is strictly clinical slate, sapphire blue, charcoal, and emerald green.
4. **Zero Spaceship / Whimsical Buttons**: All buttons and selectors adhere to radiology PACS ergonomic standards.
5. **Zero Fake Metrics or Social Proof**: No mock patient counters, fake reviews, or marketing hype.
6. **Zero Framework Watermarks**: Default Streamlit menu, deploy button, footer tags, and decoration bars are permanently suppressed via CSS.

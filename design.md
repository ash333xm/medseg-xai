# UI/UX Design Direction: MedSeg-XAI

## 1. Visual Language and Theme
The MedSeg-XAI dashboard follows a rigorous, clinical radiology PACS design system. It is deliberately understated, clinical, and precise, avoiding marketing fluff and consumer-app tropes.

## 2. Color Palette
- **Background**: Clean crisp white (`#ffffff`) or light clinical slate (`#f8fafc`).
- **Surface & Cards**: Pure white (`#ffffff`) with subtle 1px border (`#e2e8f0`).
- **Text**: Deep charcoal (`#0f172a`) for primary headings; muted slate (`#475569`) for subtext and captions.
- **Accents**:
  - Medical Sapphire: `#0369a1` (Navigation & selection highlights).
  - Ground Truth Overlay: `#16a34a` (Solid green contour).
  - MedSAM Predicted Mask: `#dc2626` (Solid red contour).
  - Audit Pass Badge: Solid dark green background (`#065f46`) with white text (`#ffffff`).
  - Audit Fail Badge: Solid crimson background (`#991b1b`) with white text (`#ffffff`).

## 3. Typography
- System sans-serif stack: `-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif`.
- Monospace font stack for metrics and coordinate readouts: `"SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace`.
- No decorative or script fonts.

## 4. Layout Structure
1. **Header**: Minimalist clinical navigation bar with project title, version metadata, and custom favicon.
2. **Main Clinician's View (Three Synchronized Panels)**:
   - Panel 1: Raw MRI Slice with Bounding Box Prompt.
   - Panel 2: Predicted Mask (Red Outline) vs Ground Truth Mask (Green Outline) with Dice score banner.
   - Panel 3: Decoder Attention Map (Hooks) overlay with thermal colorbar.
3. **Verification Module (MPRT Audit Gate)**:
   - Side-by-side comparison of Clean Heatmap A vs Scrambled Heatmap B.
   - Verdict banner: "Sanity Check Passed: Heatmap relies on learned parameters."
   - Exact numerical readouts for Spearman Correlation and Randomized Model Dice.
4. **Methodology & Transparency Accordion**:
   - Embedded curve plot `{case}_curve.png` illustrating SSIM failure vs Spearman rank collapse.
   - Technical decision log explaining the metric pivot.
   - Permanent Honest Scope disclaimer.
5. **Footer & Compliance Area**:
   - Clean, professional disclaimer.
   - Dedicated links for Terms of Service and Privacy Policy.

## 5. Prohibited Visual Elements
- Zero emoji icons anywhere in headings, buttons, or captions.
- Zero purple/blue gradients or glassmorphism blurs.
- Zero floating animated particles or custom cursor effects.
- Zero em dashes (—); use hyphens (-) only.

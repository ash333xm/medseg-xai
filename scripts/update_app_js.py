with open('app.js', 'r', encoding='utf-8') as f:
    code = f.read()

# Replace btnPlay styling
code = code.replace('btnPlay.innerHTML = "&#9632; Pause";', 'btnPlay.textContent = "Pause";')
code = code.replace('btnPlay.innerHTML = "&#9654; Play Slices";', 'btnPlay.textContent = "Play Slices";')
code = code.replace('btnPlay.style.color = "#f87171";', 'btnPlay.style.color = "#dc2626";')
code = code.replace('btnPlay.style.color = "#38bdf8";', 'btnPlay.style.color = "var(--accent-blue)";')

# Fix text colors in populateCohortTable
old_cohort_row = '''        <td style="font-weight:700; color:#ffffff;">${c.metrics.dice_clean.toFixed(4)}</td>
        <td>${c.metrics.dice_after_decoder_rand.toFixed(3)}</td>
        <td style="font-weight:600; color:${c.metrics.spearman_decoder_full <= 0.3 ? "#34d399" : "#f87171"}">${c.metrics.spearman_decoder_full.toFixed(3)}</td>
        <td>${c.metrics.ssim_decoder_full.toFixed(3)}</td>
        <td>${gateBadge}</td>
        <td>${statusText}</td>'''

new_cohort_row = '''        <td style="font-weight:700; color:var(--text-primary);">${c.metrics.dice_clean.toFixed(4)}</td>
        <td>${c.metrics.dice_after_decoder_rand.toFixed(3)}</td>
        <td style="font-weight:700; color:${c.metrics.spearman_decoder_full <= 0.3 ? "#059669" : "#dc2626"}">${c.metrics.spearman_decoder_full.toFixed(3)}</td>
        <td>${c.metrics.ssim_decoder_full.toFixed(3)}</td>
        <td>${gateBadge}</td>
        <td>${statusText}</td>'''

code = code.replace(old_cohort_row, new_cohort_row)

code = code.replace(
    'const statusText = passed ? `<span style="color:#34d399; font-weight:600;">Approved</span>` : `<span style="color:#f87171; font-weight:600;">Withheld</span>`;',
    'const statusText = passed ? `<span style="color:#059669; font-weight:700;">Approved</span>` : `<span style="color:#dc2626; font-weight:700;">Withheld</span>`;'
)

# Fix text colors in updateDesignCatalogGrid
code = code.replace(
    '<div style="font-size: 0.82rem; font-weight: 700; color: #ffffff; margin-bottom: 4px;">${item.title}</div>',
    '<div style="font-size: 0.84rem; font-weight: 700; color: var(--text-primary); margin-bottom: 4px;">${item.title}</div>'
)
code = code.replace(
    '<div style="font-size: 0.72rem; color: #94a3b8; line-height: 1.4;">${item.desc}</div>',
    '<div style="font-size: 0.76rem; color: var(--text-muted); line-height: 1.45;">${item.desc}</div>'
)

# Add populateVisualCaseGrid
visual_grid_func = '''
// Populate Visual Case Catalog Grid (Illustrations for all 16 cases)
function populateVisualCaseGrid() {
  const grid = document.getElementById("visualCaseGrid");
  if (!grid) return;

  grid.innerHTML = Object.values(CASE_CATALOG).map(c => `
    <div class="case-card-visual ${c.id === currentCaseId ? "active" : ""}" onclick="selectCaseAndOpen('${c.id}')">
      <div class="case-card-visual-media">
        <img src="data/${c.id}/raw.png" onerror="this.src='public/data/${c.id}/raw.png'" alt="${c.title}">
      </div>
      <div class="case-card-visual-title">${c.title}</div>
      <div style="font-size: 0.74rem; color: var(--text-secondary); margin-bottom: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${c.organ}</div>
      <div class="case-card-visual-meta">
        <span>Dice: <strong>${c.metrics.dice_clean.toFixed(3)}</strong></span>
        <span class="badge-tag ${c.audit_passed ? "badge-green" : "badge-red"}" style="padding: 2px 7px; font-size: 0.68rem;">${c.audit_passed ? "PASS" : "FAIL"}</span>
      </div>
    </div>
  `).join("");
}

window.selectCaseAndOpen = function(caseId) {
  const sel = document.getElementById("caseSelector");
  if (sel) {
    sel.value = caseId;
    sel.dispatchEvent(new Event("change"));
  }
  const clinTab = document.querySelector('[data-tab="clinician"]');
  if (clinTab) clinTab.click();
};
'''

# Insert visual_grid_func right before DOMContentLoaded
if 'populateVisualCaseGrid();' not in code:
    code = code.replace('populateCohortTable();', 'populateCohortTable();\n  populateVisualCaseGrid();')
    code = code + '\n' + visual_grid_func

with open('app.js', 'w', encoding='utf-8') as f:
    f.write(code)

with open('public/app.js', 'w', encoding='utf-8') as f:
    f.write(code)

print('app.js and public/app.js updated successfully!')

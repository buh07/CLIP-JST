"""Generate all new figures for neurips_2026.tex.

Run from the paper/ directory:
    python make_figures.py

Outputs PDFs to paper/figures/:
  fig_emergence.pdf       -- emergence & causal controls bar chart
  fig_crosstriple.pdf     -- three-triple alpha + functional form held-out R^2
  fig_phaseA_quality.pdf  -- Phase A source comparison bar chart
  fig_w5_gap.pdf          -- W5 centroid gap analysis (shared vs separate JL)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'legend.framealpha': 0.9,
    'figure.dpi': 300,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    # Avoid Type 3 fonts in PDF output for submission compatibility/readability.
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

CHANCE = 0.0011


# ---------------------------------------------------------------------------
# Figure 1: Emergence and causal controls
# ---------------------------------------------------------------------------
def make_emergence():
    # Data: modular_shared_jl Stage44 (full training), Stage47 (identity), W11/W12 (shuffled)
    # X-axis: m=64, 256, 512  (m=128 excluded: shuffled not reported there)
    dims = [64, 256, 512]
    x = np.arange(len(dims))
    width = 0.25

    full_train = [0.0068, 0.0204, 0.0339]
    identity   = [CHANCE, CHANCE, CHANCE]   # Stage47: all at chance
    shuffled   = [0.0005, 0.0005, 0.00130]  # W12 (m=64,256), W11 (m=512)

    fig, ax = plt.subplots(figsize=(6.5, 2.0))

    b1 = ax.bar(x - width, full_train, width, label='Full training (Stage~44)',
                color='#2166ac', zorder=3)
    b2 = ax.bar(x,          identity,  width, label='Identity ablation (Stage~47)',
                color='#999999', zorder=3)
    b3 = ax.bar(x + width,  shuffled,  width, label='Shuffled-caption control (W11/W12)',
                color='#d6604d', zorder=3, alpha=0.9)

    ax.axhline(CHANCE, color='black', linestyle='--', linewidth=0.8,
               label=f'Chance ($\\bar{{R}}={{\\rm chance}}$)', zorder=2)

    # Annotate multiples above chance for full training bars
    for xi, val in zip(x - width, full_train):
        mult = val / CHANCE
        ax.text(xi, val + 0.0002, f'{mult:.0f}$\\times$', ha='center', va='bottom',
                fontsize=6.5, color='#2166ac')

    ax.set_xticks(x)
    ax.set_xticklabels([f'$m={d}$' for d in dims])
    ax.set_xlabel('Embedding dimension $m$')
    ax.set_ylabel('$\\bar{R}_{\\mathrm{ia}}$')
    ax.set_ylim(0, 0.046)
    ax.legend(loc='upper left', frameon=True, ncol=2)
    ax.yaxis.grid(True, linestyle=':', linewidth=0.5, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout(pad=0.3)
    out = OUT_DIR / 'fig_emergence.pdf'
    fig.savefig(out, bbox_inches='tight', format='pdf')
    plt.close(fig)
    print(f'Saved {out}')


# ---------------------------------------------------------------------------
# Figure 2: Cross-triple alpha + functional form comparison (two panels)
# ---------------------------------------------------------------------------
def make_crosstriple():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 2.2))

    # --- Panel A: Three-triple alpha bar chart ---
    triples = ['AudioCaps\n(env. sounds)', 'AVCaps\n(video audio)', 'SpeechCoco\n(spoken captions)']
    alphas  = [0.270,  0.673,  0.0198]
    # Error: half-width of 95% CI from mixed-effects models
    errs    = [0.015,  0.016,  0.0033]
    colors  = ['#2166ac', '#4dac26', '#d6604d']

    x = np.arange(len(triples))
    bars = ax1.bar(x, alphas, 0.5, yerr=errs, capsize=4,
                   color=colors, zorder=3,
                   error_kw={'elinewidth': 1.2, 'capthick': 1.2})

    ax1.axhline(0.05, color='#d6604d', linestyle='--', linewidth=1.0, zorder=4,
                label='$\\alpha=0.05$ (encoder mismatch)')
    ax1.axhline(0.20, color='#4dac26', linestyle='--', linewidth=1.0, zorder=4,
                label='$\\alpha=0.20$ (productive regime)')

    # Annotate alpha values on bars
    for xi, (val, err) in enumerate(zip(alphas, errs)):
        ax1.text(xi, val + err + 0.012, f'$\\alpha={val}$',
                 ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(triples, fontsize=7.5)
    ax1.set_ylabel('Transmission efficiency $\\alpha$')
    ax1.set_title('(a) Encoder-domain fit across triples', fontsize=8.5)
    ax1.set_ylim(0, 0.82)
    ax1.legend(loc='upper right', fontsize=6.5, frameon=True)
    ax1.yaxis.grid(True, linestyle=':', linewidth=0.5, alpha=0.7, zorder=0)
    ax1.set_axisbelow(True)

    # --- Panel B: Functional form held-out R^2 ---
    forms = ['Geometric\nmean', 'Arithmetic\nmean', 'Hard min', 'Product', 'Free\npower-form']
    held_r2 = [0.853, 0.552, 0.203, 0.210, -1.173]
    bar_colors = ['#2166ac', '#999999', '#999999', '#999999', '#d6604d']

    x2 = np.arange(len(forms))
    # Clip negative bar for display; annotate actual value
    display_r2 = [max(v, -0.20) for v in held_r2]

    bars2 = ax2.bar(x2, display_r2, 0.5, color=bar_colors, zorder=3)
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.8, zorder=4)

    # Annotate each bar
    for xi, (val, dval) in enumerate(zip(held_r2, display_r2)):
        ypos = dval + 0.02 if dval >= 0 else dval - 0.04
        va = 'bottom' if dval >= 0 else 'top'
        label = f'{val:.3f}' if val > -1 else '$-1.17$'
        weight = 'bold' if xi == 0 else 'normal'
        ax2.text(xi, ypos, label, ha='center', va=va, fontsize=7, fontweight=weight)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(forms, fontsize=7.5)
    ax2.set_ylabel('Held-out $R^2$ (AudioCaps$\\to$WavCaps)')
    ax2.set_title('(b) Cross-regime functional form', fontsize=8.5)
    ax2.set_ylim(-0.30, 1.05)
    ax2.yaxis.grid(True, linestyle=':', linewidth=0.5, alpha=0.7, zorder=0)
    ax2.set_axisbelow(True)

    # Highlight winning bar
    bars2[0].set_edgecolor('#1a4a7a')
    bars2[0].set_linewidth(1.5)

    plt.tight_layout(pad=0.3, w_pad=1.0)
    out = OUT_DIR / 'fig_crosstriple.pdf'
    fig.savefig(out, bbox_inches='tight', format='pdf')
    plt.close(fig)
    print(f'Saved {out}')


# ---------------------------------------------------------------------------
# Figure 3: Phase A source and scale comparison
# ---------------------------------------------------------------------------
def make_phaseA_quality():
    dims = [64, 128, 256, 512]
    x = np.arange(len(dims))
    width = 0.25

    # modular_shared_jl, same AudioCaps Phase B; from Table 13 (Appendix H)
    cc3m_100k    = [0.0043, 0.0195, 0.0190, 0.0328]
    coco_sub_100k = [0.0117, 0.0140, 0.0291, 0.0407]
    coco_full_566k = [0.0068, 0.0116, 0.0208, 0.0339]

    fig, ax = plt.subplots(figsize=(4.0, 2.8))

    b1 = ax.bar(x - width, cc3m_100k,    width, label='CC3M-100K (web, 100K pairs)',
                color='#d6604d', zorder=3)
    b2 = ax.bar(x,          coco_sub_100k, width, label='COCO-sub-100K (human, 100K pairs)',
                color='#2166ac', zorder=3)
    b3 = ax.bar(x + width,  coco_full_566k, width, label='COCO-full-566K (human, 566K pairs)',
                color='#92c5de', zorder=3)

    ax.axhline(CHANCE, color='black', linestyle='--', linewidth=0.8, zorder=2)

    # Annotate m=128 "CC3M wins" and m=256/512 "COCO-sub wins"
    ax.annotate('CC3M\nwins', xy=(1 - width, cc3m_100k[1] + 0.0005),
                xytext=(1 - width, cc3m_100k[1] + 0.004),
                ha='center', va='bottom', fontsize=6,
                color='#d6604d', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#d6604d', lw=0.8))
    ax.annotate('COCO-sub\nwins', xy=(3, coco_sub_100k[3] + 0.0005),
                xytext=(3, coco_sub_100k[3] + 0.004),
                ha='center', va='bottom', fontsize=6,
                color='#2166ac', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#2166ac', lw=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels([f'$m={d}$' for d in dims])
    ax.set_xlabel('Embedding dimension $m$')
    ax.set_ylabel('$\\bar{R}_{\\mathrm{ia}}$')
    ax.set_ylim(0, 0.053)
    ax.legend(loc='upper left', frameon=True)
    ax.yaxis.grid(True, linestyle=':', linewidth=0.5, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = OUT_DIR / 'fig_phaseA_quality.pdf'
    fig.savefig(out, bbox_inches='tight', format='pdf')
    plt.close(fig)
    print(f'Saved {out}')


# ---------------------------------------------------------------------------
# Figure 4: W5 centroid gap analysis (shared vs separate JL, all 4 dims)
# ---------------------------------------------------------------------------
def make_w5_gap():
    # Data from w5_gap_analysis_results.json (all 4 dims, 5 seeds each)
    dims = [64, 128, 256, 512]

    shared_gap_mean  = [0.5701, 0.5686, 0.5851, 0.5324]
    shared_gap_std   = [0.0035, 0.0060, 0.0037, 0.0058]
    sep_gap_mean     = [0.5870, 0.5347, 0.5820, 0.5523]
    sep_gap_std      = [0.0148, 0.0042, 0.0028, 0.0051]

    shared_ia_mean   = [0.00680, 0.01165, 0.02039, 0.03339]
    shared_ia_std    = [0.00028, 0.00049, 0.00142, 0.00179]
    sep_ia_mean      = [0.00576, 0.00695, 0.02628, 0.03239]
    sep_ia_std       = [0.00082, 0.00024, 0.00134, 0.00079]

    x = np.arange(len(dims))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    # --- Panel A: ia_gap by dim ---
    ax1.plot(x, shared_gap_mean, 'o-', color='#2166ac', label='Shared JL', linewidth=1.4,
             markersize=5, zorder=3)
    ax1.fill_between(x,
                     np.array(shared_gap_mean) - np.array(shared_gap_std),
                     np.array(shared_gap_mean) + np.array(shared_gap_std),
                     color='#2166ac', alpha=0.15, zorder=2)
    ax1.plot(x, sep_gap_mean, 's--', color='#d6604d', label='Separate JL', linewidth=1.4,
             markersize=5, zorder=3)
    ax1.fill_between(x,
                     np.array(sep_gap_mean) - np.array(sep_gap_std),
                     np.array(sep_gap_mean) + np.array(sep_gap_std),
                     color='#d6604d', alpha=0.15, zorder=2)

    # Annotate m=128 sign reversal and m=256 near-zero
    ax1.annotate('$\\Delta{=}{-}0.034$\n(sign\\,rev.)',
                 xy=(1, sep_gap_mean[1]), xytext=(0.5, 0.51),
                 fontsize=6.5, color='gray',
                 arrowprops=dict(arrowstyle='->', color='gray', lw=0.7))
    ax1.annotate('$\\Delta{=}{-}0.003$\n(near zero)',
                 xy=(2, shared_gap_mean[2]), xytext=(2.0, 0.60),
                 fontsize=6.5, color='gray',
                 arrowprops=dict(arrowstyle='->', color='gray', lw=0.7))

    ax1.set_xticks(x)
    ax1.set_xticklabels([f'$m={d}$' for d in dims])
    ax1.set_ylabel('Centroid gap $\\ell_2$ (ia)')
    ax1.set_title(r'(a) Centroid gap: shared vs.\ separate JL', fontsize=8.5)
    ax1.set_ylim(0.44, 0.68)
    ax1.legend(loc='upper right', frameon=True)
    ax1.yaxis.grid(True, linestyle=':', linewidth=0.5, alpha=0.7, zorder=0)
    ax1.set_axisbelow(True)

    # --- Panel B: av_ia by dim ---
    ax2.plot(x, shared_ia_mean, 'o-', color='#2166ac', label='Shared JL', linewidth=1.4,
             markersize=5, zorder=3)
    ax2.fill_between(x,
                     np.array(shared_ia_mean) - np.array(shared_ia_std),
                     np.array(shared_ia_mean) + np.array(shared_ia_std),
                     color='#2166ac', alpha=0.15, zorder=2)
    ax2.plot(x, sep_ia_mean, 's--', color='#d6604d', label='Separate JL', linewidth=1.4,
             markersize=5, zorder=3)
    ax2.fill_between(x,
                     np.array(sep_ia_mean) - np.array(sep_ia_std),
                     np.array(sep_ia_mean) + np.array(sep_ia_std),
                     color='#d6604d', alpha=0.15, zorder=2)

    # Annotate the reversal at m=128 (shared WINS despite LARGER gap)
    ax2.annotate('Shared wins\n(gap larger!)',
                 xy=(1, shared_ia_mean[1] + 0.0003), xytext=(1.3, 0.018),
                 fontsize=6.5, color='#2166ac',
                 arrowprops=dict(arrowstyle='->', color='#2166ac', lw=0.7))
    # Annotate the reversal at m=256 (separate WINS despite SAME gap)
    ax2.annotate('Separate wins\n(gap $\\approx$ same)',
                 xy=(2, sep_ia_mean[2] + 0.0003), xytext=(1.5, 0.030),
                 fontsize=6.5, color='#d6604d',
                 arrowprops=dict(arrowstyle='->', color='#d6604d', lw=0.7))

    ax2.set_xticks(x)
    ax2.set_xticklabels([f'$m={d}$' for d in dims])
    ax2.set_ylabel('$\\bar{R}_{\\mathrm{ia}}$')
    ax2.set_title(r'(b) Image-audio retrieval: shared vs.\ separate JL', fontsize=8.5)
    ax2.legend(loc='upper left', frameon=True)
    ax2.yaxis.grid(True, linestyle=':', linewidth=0.5, alpha=0.7, zorder=0)
    ax2.set_axisbelow(True)

    plt.tight_layout(w_pad=1.5)
    out = OUT_DIR / 'fig_w5_gap.pdf'
    fig.savefig(out, bbox_inches='tight', format='pdf')
    plt.close(fig)
    print(f'Saved {out}')


# ---------------------------------------------------------------------------
# Figure 5: Bottleneck theory scatter — predicted vs. observed
# ---------------------------------------------------------------------------
def make_bottleneck_scatter():
    import json

    data_path = (
        Path(__file__).parents[1]
        / "MultiModal/results/reviewer_fixes_suite/stage36_s44_coco"
        / "stage36_bottleneck_decomposition/stage36_bottleneck_decomposition.json"
    )
    with data_path.open() as f:
        d = json.load(f)

    source_label = {
        'stage25_jlablation_gpu4': 'Sharing factorial',
        'stage25_jlablation_gpu5': 'Sharing factorial',
        'stage25_jlablation_gpu6': 'Sharing factorial',
        'stage25_jlablation_gpu7': 'Sharing factorial',
        'stage31_wavcaps_scaling': 'WavCaps scaling',
        'stage32_modality_order':  'Phase order',
        'stage44_coco_phaseA':     'Projection type',
    }
    label_color = {
        'Sharing factorial': '#aaaaaa',
        'WavCaps scaling':   '#fc8d59',
        'Phase order':       '#91bfdb',
        'Projection type':   '#2166ac',
    }

    groups: dict[str, tuple[list, list]] = {k: ([], []) for k in label_color}
    for rec in d['records']:
        label = source_label.get(rec['source_id'], 'Other')
        if label not in groups:
            groups[label] = ([], [])
        groups[label][0].append(rec['predicted_ia'])
        groups[label][1].append(rec['av_ia'])

    fig, ax = plt.subplots(figsize=(4.8, 3.8))

    for label, (pred, obs) in groups.items():
        color = label_color.get(label, '#888888')
        ax.scatter(pred, obs, s=14, color=color, alpha=0.75, label=label, zorder=3)

    # Identity line
    all_vals = [v for pred, obs in groups.values() for v in pred + obs]
    lo, hi = min(all_vals) * 0.9, max(all_vals) * 1.08
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=0.9, zorder=4, label='Perfect prediction')

    ax.set_xlabel('Predicted $\\bar{R}_{\\mathrm{ia}}$ (bottleneck theory)')
    ax.set_ylabel('Observed $\\bar{R}_{\\mathrm{ia}}$')
    ax.set_title(
        r'Transitivity Bottleneck Theory: Predicted vs.\ Observed' + '\n'
        '(COCO Phase A, $n=300$, $r=0.921$)',
        fontsize=8.5,
    )
    ax.legend(loc='upper left', fontsize=7, frameon=True)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.yaxis.grid(True, linestyle=':', linewidth=0.5, alpha=0.7, zorder=0)
    ax.xaxis.grid(True, linestyle=':', linewidth=0.5, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout(pad=0.4)
    out = OUT_DIR / 'fig_bottleneck_law.pdf'
    fig.savefig(out, bbox_inches='tight', format='pdf')
    plt.close(fig)
    print(f'Saved {out}')


if __name__ == '__main__':
    print('Generating figures...')
    make_emergence()
    make_crosstriple()
    make_phaseA_quality()
    make_w5_gap()
    make_bottleneck_scatter()
    print('Done. Check paper/figures/ for output PDFs.')

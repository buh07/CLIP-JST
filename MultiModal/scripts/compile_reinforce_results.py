#!/usr/bin/env python3
"""
Compile reinforce.txt experiment outcomes (W1..W13) into one auditable report.

Outputs:
  - MultiModal/results/reinforce_suite/REINFORCE_RESULTS_COMPILATION.json
  - MultiModal/results/reinforce_suite/REINFORCE_RESULTS_COMPILATION.md
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


STATUS_COMPLETED = "completed"
STATUS_PARTIAL = "partial"
STATUS_NOT_FOUND = "not found"


@dataclass
class WEntry:
    wid: str
    title: str
    status: str
    severity: str
    primary_finding: str
    evidence_paths: list[str] = field(default_factory=list)
    key_metrics: dict[str, Any] = field(default_factory=dict)
    interpretation: str = ""
    implication_for_paper: str = ""
    remaining_gap: str = ""
    mismatch_proxy_note: str = ""
    warnings: list[str] = field(default_factory=list)


def rel(root: Path, p: Path) -> str:
    try:
        return str(p.relative_to(root))
    except Exception:
        return str(p)


def load_json(path: Path, warnings: list[str]) -> dict[str, Any] | None:
    if not path.exists():
        warnings.append(f"Missing mapped artifact: {path}")
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # pragma: no cover - defensive
        warnings.append(f"Failed to parse JSON {path}: {exc}")
        return None


def float_fmt(x: Any, nd: int = 4) -> Any:
    if isinstance(x, float):
        return round(x, nd)
    return x


def nested_get(d: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def extract_w_titles(reinforce_text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in reinforce_text.splitlines():
        m = re.match(r"^W(\d+)\s+—\s+(.+?)\s*$", line.strip())
        if m:
            wid = f"W{m.group(1)}"
            # Keep the first canonical section title; later priority-list bullets can
            # repeat W-ids with sentence fragments that should not override headings.
            if wid not in out:
                out[wid] = m.group(2).strip()
    return out


def git_hash(root: Path) -> str:
    try:
        cp = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        return cp.stdout.strip()
    except Exception:
        return "unknown"


def glob_exists(base: Path, pattern: str) -> list[Path]:
    return sorted(base.glob(pattern))


def build_entries(root: Path, titles: dict[str, str]) -> tuple[list[WEntry], list[str]]:
    warnings: list[str] = []
    rr = root / "MultiModal" / "results"

    p_w1_full = rr / "next_run_suite/w1_avcaps_full/aggregate/stage58_second_triple_aggregate.json"
    p_w1_head = rr / "next_run_suite/w1_avcaps_full/aggregate/headline/stage58_second_triple_aggregate.json"
    p_w1_pilot = rr / "reinforce_suite/w1_second_triple_avcaps/aggregate/stage58_second_triple_aggregate.json"
    p_w3_ext = rr / "next_run_suite/w3_holdout_ext/aggregate/stage56_wavcaps_holdout_aggregate.json"
    p_w5 = rr / "reviewer_fixes_suite/w5_gap_alpha_regression/w5_gap_alpha_regression_results.json"
    p_w5_s44 = rr / "reviewer_fixes_suite/w5_s44_regression/w5_s44_regression_results.json"
    p_w7_proxy = rr / "reviewer_fixes_suite/w7_imagebind_1k_split/w7_imagebind_1k_split_results.json"
    p_w8 = rr / "reviewer_fixes_suite/w8_1k_split_eval/w8_1k_split_eval_results_reinforce_m512.json"
    p_w59 = rr / "reviewer_fixes_suite/stage59_paper_revision/stage59_paper_revision_analyses/stage59_paper_revision_analyses.json"
    p_w60 = rr / "next_run_suite/post_w3_sequence/stage60_joint_gap_alpha/stage60_joint_gap_alpha.json"
    p_w47 = rr / "experimental_fixes_suite/stage47_identity_ablation/stage47_identity_ablation/stage47_identity_ablation_results.json"
    p_paper = root / "paper/neurips_2026.tex"

    j_w1_full = load_json(p_w1_full, warnings)
    j_w1_head = load_json(p_w1_head, warnings)
    j_w1_pilot = load_json(p_w1_pilot, warnings)
    j_w3_ext = load_json(p_w3_ext, warnings)
    j_w5 = load_json(p_w5, warnings)
    j_w5_s44 = load_json(p_w5_s44, warnings)
    j_w7_proxy = load_json(p_w7_proxy, warnings)
    j_w8 = load_json(p_w8, warnings)
    j_w59 = load_json(p_w59, warnings)
    j_w60 = load_json(p_w60, warnings)
    j_w47 = load_json(p_w47, warnings)

    entries: list[WEntry] = []

    # W1
    w1_title = titles.get("W1", "Single modality triple (law vs conjecture)")
    w1_status = STATUS_NOT_FOUND
    w1_metrics: dict[str, Any] = {}
    w1_finding = "No stage58 second-triple aggregate found."
    w1_ev: list[str] = []
    if j_w1_full is not None:
        w1_ev.append(rel(root, p_w1_full))
        lg = j_w1_full.get("law_global", {})
        n_full = int(lg.get("n", -1))
        w1_metrics["full_grid_law_global"] = {
            k: float_fmt(v, 4) for k, v in lg.items() if k in {"n", "alpha", "pearson_r", "r2", "mae"}
        }
        w1_metrics["full_grid_methods"] = sorted(list(j_w1_full.get("law_by_method", {}).keys()))
        if j_w1_head is not None:
            w1_ev.append(rel(root, p_w1_head))
            n_head = int(nested_get(j_w1_head, ["law_global", "n"], -1))
            w1_metrics["headline_law_n"] = n_head
            if n_full == 80 and n_head == 60:
                w1_status = STATUS_COMPLETED
                w1_finding = (
                    "Second-triple full grid completed with strong fit "
                    f"(n={n_full}, r={float_fmt(lg.get('pearson_r'),4)})."
                )
            else:
                w1_status = STATUS_PARTIAL
                w1_finding = (
                    "Stage58 outputs exist, but expected coverage checks failed "
                    f"(n_full={n_full}, n_head={n_head}, expected 80/60)."
                )
                warnings.append("W1 coverage mismatch: expected full n=80 and headline n=60.")
        else:
            w1_status = STATUS_PARTIAL
            w1_finding = "Full-grid W1 exists, but headline-only stage58 aggregate is missing."
    if j_w1_pilot is not None:
        w1_ev.append(rel(root, p_w1_pilot))
        w1_metrics["pilot_law_global"] = {
            k: float_fmt(v, 4)
            for k, v in j_w1_pilot.get("law_global", {}).items()
            if k in {"n", "alpha", "pearson_r", "r2", "mae"}
        }

    entries.append(
        WEntry(
            wid="W1",
            title=w1_title,
            status=w1_status,
            severity="high",
            primary_finding=w1_finding,
            evidence_paths=w1_ev,
            key_metrics=w1_metrics,
            interpretation=(
                "The expanded AVCaps grid materially strengthens external validation of the bottleneck form "
                "relative to the pilot-only state."
            ),
            implication_for_paper=(
                "Update all text that still describes W1 as pilot-limited (2 methods/2 dims). "
                "Use full-grid values as primary evidence."
            ),
            remaining_gap=(
                "No immediate W1 gap if full-grid numbers are fully propagated through abstract, "
                "main tables, and limitations."
            ),
        )
    )

    # W2
    w2_title = titles.get("W2", "Encoder confound in the headline result")
    joint_clap_hits = glob_exists(rr, "**/*joint*clap*.json")
    w2_ev = [rel(root, p) for p in joint_clap_hits[:10]]
    w2_proxy_ev: list[str] = []
    w2_metrics: dict[str, Any] = {}
    if j_w59 is not None:
        w2_proxy_ev.append(rel(root, p_w59))
        decomp = j_w59.get("w9_encoder_supervision_decomposition", {}).get(
            "modular_shared_jl_vs_joint_clip_head", {}
        )
        if decomp:
            w2_metrics["proxy_encoder_supervision_decomposition"] = {
                "observed_ratio_mod_over_joint": float_fmt(decomp.get("observed_ratio_mod_over_joint"), 4),
                "bridge_ratio_sqrt_itat": float_fmt(decomp.get("bridge_ratio_sqrt_itat"), 4),
                "implied_alpha_ratio": float_fmt(decomp.get("implied_alpha_ratio"), 4),
            }
    if joint_clap_hits:
        w2_status = STATUS_COMPLETED
        w2_finding = "Direct joint-CLAP reference artifacts detected."
        w2_mismatch = ""
    else:
        w2_status = STATUS_NOT_FOUND
        w2_finding = "No direct joint-CLAP reference training artifact found."
        w2_mismatch = (
            "Mismatch/Proxy: only decomposition analyses on existing joint-CLIP references were found; "
            "this does not resolve the encoder-matched W2 ask."
        )

    entries.append(
        WEntry(
            wid="W2",
            title=w2_title,
            status=w2_status,
            severity="high",
            primary_finding=w2_finding,
            evidence_paths=w2_ev + w2_proxy_ev,
            key_metrics=w2_metrics,
            interpretation=(
                "Encoder confound remains unresolved unless a true joint-CLAP reference was trained and evaluated."
            ),
            implication_for_paper=(
                "Keep method comparisons explicitly framed as reference-scale, not encoder-matched superiority claims."
            ),
            remaining_gap="Run/locate encoder-matched joint CLAP reference row for a clean W2 closure.",
            mismatch_proxy_note=w2_mismatch,
        )
    )

    # W3
    w3_title = titles.get("W3", "Train-eval distribution shift confound")
    clotho_hits = glob_exists(rr, "**/*clotho*.json")
    w3_ev = [rel(root, p) for p in clotho_hits[:10]]
    w3_metrics: dict[str, Any] = {}
    w3_proxy_note = ""
    if j_w3_ext is not None:
        rows = j_w3_ext.get("rows", [])
        total_n = int(sum(int(r.get("n", 0)) for r in rows if isinstance(r, dict)))
        w3_metrics["holdout_extension_total_n"] = total_n
        # pull m512 condition rows for compact summary
        m512 = [r for r in rows if r.get("embed_dim") == 512]
        w3_metrics["holdout_extension_m512"] = {
            r["condition"]: {
                "av_at_mean": float_fmt(r.get("av_at_mean"), 4),
                "av_ia_mean": float_fmt(r.get("av_ia_mean"), 4),
                "wav_holdout_at_mean": float_fmt(r.get("wav_holdout_at_mean"), 4),
            }
            for r in m512
        }
    if clotho_hits:
        w3_status = STATUS_COMPLETED
        w3_finding = "Direct Clotho intermediate evaluation artifact detected."
    else:
        w3_status = STATUS_NOT_FOUND
        w3_finding = "No direct Clotho intermediate evaluation artifact found."
        if j_w3_ext is not None:
            w3_ev.append(rel(root, p_w3_ext))
            w3_proxy_note = (
                "Mismatch/Proxy: W3 holdout retrain/eval extension exists, but this is not the requested "
                "cross-dataset Clotho intermediate test."
            )

    entries.append(
        WEntry(
            wid="W3",
            title=w3_title,
            status=w3_status,
            severity="high",
            primary_finding=w3_finding,
            evidence_paths=w3_ev,
            key_metrics=w3_metrics,
            interpretation=(
                "Current holdout analyses improve within-source diagnosis, but they do not replace explicit "
                "cross-dataset shift testing requested in W3."
            ),
            implication_for_paper=(
                "Keep quality-vs-shift claims qualified unless direct cross-dataset intermediate evidence is added."
            ),
            remaining_gap="Run/locate Clotho intermediate evaluation for direct W3 closure.",
            mismatch_proxy_note=w3_proxy_note,
        )
    )

    # W4
    w4_title = titles.get("W4", "Weak theoretical motivation for the geometric mean form")
    w4_ev: list[str] = []
    w4_metrics: dict[str, Any] = {}
    if j_w59 is not None and "w2_w4_functional_forms" in j_w59:
        w4_ev.append(rel(root, p_w59))
        models = j_w59["w2_w4_functional_forms"].get("models", [])
        table = {}
        for m in models:
            table[m.get("name", "?")] = {
                "in_sample_r2": float_fmt(m.get("in_sample_r2"), 4),
                "heldout_r2": float_fmt(m.get("heldout_r2"), 4),
                "heldout_mae": float_fmt(m.get("heldout_mae"), 4),
            }
        w4_metrics["functional_form_comparison"] = table
        w4_status = STATUS_COMPLETED
        w4_finding = "Alternative-form model comparison exists with held-out metrics."
    else:
        w4_status = STATUS_NOT_FOUND
        w4_finding = "No W4 functional-form analysis artifact found."

    entries.append(
        WEntry(
            wid="W4",
            title=w4_title,
            status=w4_status,
            severity="high",
            primary_finding=w4_finding,
            evidence_paths=w4_ev,
            key_metrics=w4_metrics,
            interpretation=(
                "This addresses the 'form-choice is ad hoc' critique by quantifying geometric-mean performance "
                "against alternatives."
            ),
            implication_for_paper=(
                "Present W4 as empirical identification/model selection, not a formal proof."
            ),
            remaining_gap=(
                "Optional: add explicit confidence intervals and robustness notes for free-exponent instability."
            ),
        )
    )

    # W5
    w5_title = titles.get("W5", "Out-of-sample test validates interpolation, not extrapolation")
    w5_ev: list[str] = []
    w5_metrics: dict[str, Any] = {}
    w5_status = STATUS_NOT_FOUND
    w5_finding = "No W5 regression artifacts found."
    if j_w5 is not None:
        w5_ev.append(rel(root, p_w5))
        w5_metrics["gap_alpha_regression_pooled"] = {
            "gap_to_alpha_r": float_fmt(nested_get(j_w5, ["pooled", "gap_to_alpha", "pearson_r"]), 4),
            "gap_to_alpha_r2": float_fmt(nested_get(j_w5, ["pooled", "gap_to_alpha", "r2"]), 4),
            "alpha_to_avR_r": float_fmt(nested_get(j_w5, ["pooled", "alpha_to_avR", "pearson_r"]), 4),
            "alpha_to_avR_r2": float_fmt(nested_get(j_w5, ["pooled", "alpha_to_avR", "r2"]), 4),
            "n_rows": j_w5.get("n_rows"),
        }
    if j_w5_s44 is not None:
        w5_ev.append(rel(root, p_w5_s44))
        w5_metrics["s44_expansion_pooled"] = {
            "gap_to_alpha_r": float_fmt(nested_get(j_w5_s44, ["pooled", "gap_to_alpha", "pearson_r"]), 4),
            "gap_to_alpha_r2": float_fmt(nested_get(j_w5_s44, ["pooled", "gap_to_alpha", "r2"]), 4),
            "n_rows": j_w5_s44.get("n_rows"),
        }
    if j_w60 is not None:
        w5_ev.append(rel(root, p_w60))
        w5_metrics["joint_method_gap_alpha"] = {
            "law_r": float_fmt(nested_get(j_w60, ["law_global", "pearson_r"]), 4),
            "law_r2": float_fmt(nested_get(j_w60, ["law_global", "r2"]), 4),
            "gap_to_alpha_r": float_fmt(nested_get(j_w60, ["gap_to_alpha", "pearson_r"]), 4),
            "n_rows": j_w60.get("n_rows"),
        }
    if j_w5 is not None and j_w5_s44 is not None:
        w5_status = STATUS_COMPLETED
        w5_finding = "Gap→alpha and alpha→retrieval regressions are available with pooled and expanded analyses."
    elif j_w5 is not None:
        w5_status = STATUS_PARTIAL
        w5_finding = "Base W5 regression exists, but expanded S44 regression is missing."

    entries.append(
        WEntry(
            wid="W5",
            title=w5_title,
            status=w5_status,
            severity="high",
            primary_finding=w5_finding,
            evidence_paths=w5_ev,
            key_metrics=w5_metrics,
            interpretation=(
                "W5 substantially strengthens the predictive/geometry interpretation beyond a single pooled fit."
            ),
            implication_for_paper=(
                "Use these regressions to justify alpha as measurable geometry-linked efficiency, not only a fit parameter."
            ),
            remaining_gap=(
                "Optional: add external-method prediction (alpha estimated before retrieval) as a prospective test."
            ),
        )
    )

    # W6
    w6_title = titles.get("W6", "CLAP alignment margin circularity")
    w6_ev: list[str] = []
    w6_metrics: dict[str, Any] = {}
    if j_w59 is not None and "w6_margin_regression" in j_w59:
        w6_ev.append(rel(root, p_w59))
        mr = j_w59["w6_margin_regression"]
        w6_metrics = {
            "margin_to_av_at_r": float_fmt(nested_get(mr, ["regression_margin_to_av_at", "pearson_r"]), 4),
            "margin_to_av_at_r2": float_fmt(nested_get(mr, ["regression_margin_to_av_at", "r2"]), 4),
            "margin_to_av_ia_r": float_fmt(nested_get(mr, ["regression_margin_to_av_ia", "pearson_r"]), 4),
            "num_points": len(mr.get("points", [])),
        }
        w6_status = STATUS_COMPLETED
        w6_finding = "Margin-performance regression exists across Phase-B conditions."
    else:
        w6_status = STATUS_NOT_FOUND
        w6_finding = "No explicit W6 margin regression artifact found."

    entries.append(
        WEntry(
            wid="W6",
            title=w6_title,
            status=w6_status,
            severity="medium",
            primary_finding=w6_finding,
            evidence_paths=w6_ev,
            key_metrics=w6_metrics,
            interpretation=(
                "Margin remains useful as an empirical predictor in this stack, but circularity caveats still apply."
            ),
            implication_for_paper=(
                "Frame margin thresholds as dataset/model-calibrated heuristics, not universal hard cutoffs."
            ),
            remaining_gap="Add explicit caveat text on CLAP-domain dependence if not already present.",
        )
    )

    # W7
    w7_title = titles.get("W7", "No quantitative comparison to prior work")
    audioclip_hits = glob_exists(rr, "**/*audioclip*.json")
    w7_ev = [rel(root, p) for p in audioclip_hits[:10]]
    w7_metrics: dict[str, Any] = {}
    w7_proxy_note = ""
    if j_w7_proxy is not None:
        w7_ev.append(rel(root, p_w7_proxy))
        w7_metrics["imagebind_proxy"] = {
            "model": j_w7_proxy.get("model"),
            "k1_av_ia": float_fmt(nested_get(j_w7_proxy, ["k1_split", "av_ia_avg_R"]), 4),
            "full_av_ia": float_fmt(nested_get(j_w7_proxy, ["full_pool_reference", "av_ia_avg_R"]), 4),
            "k1_n_items": nested_get(j_w7_proxy, ["k1_split", "n_items"]),
        }
    if audioclip_hits:
        w7_status = STATUS_COMPLETED
        w7_finding = "Direct AudioCLIP baseline artifact detected."
    else:
        w7_status = STATUS_NOT_FOUND
        w7_finding = "No direct AudioCLIP baseline artifact found."
        w7_proxy_note = (
            "Mismatch/Proxy: ImageBind baseline exists, but it does not satisfy the direct AudioCLIP baseline ask."
        )

    entries.append(
        WEntry(
            wid="W7",
            title=w7_title,
            status=w7_status,
            severity="high",
            primary_finding=w7_finding,
            evidence_paths=w7_ev,
            key_metrics=w7_metrics,
            interpretation=(
                "Baseline context improved via ImageBind proxy, but requested method-specific prior-work comparison remains open."
            ),
            implication_for_paper=(
                "Keep claims conservative on direct prior-work competitiveness until AudioCLIP-matched numbers are available."
            ),
            remaining_gap="Run or locate AudioCLIP inference under the same evaluation protocol.",
            mismatch_proxy_note=w7_proxy_note,
        )
    )

    # W8
    w8_title = titles.get("W8", "Non-standard evaluation protocol (4,411 vs 1K test pool)")
    w8_ev: list[str] = []
    w8_metrics: dict[str, Any] = {}
    if j_w8 is not None:
        w8_ev.append(rel(root, p_w8))
        rows = nested_get(j_w8, ["stage30", "m512", "audio_linear_probe"], [])
        if isinstance(rows, list) and rows:
            full_av_ia = mean(float(r["full_pool"]["av_ia_avg_R"]) for r in rows)
            k1_av_ia = mean(float(r["k1_split"]["av_ia_avg_R"]) for r in rows)
            w8_metrics["audio_linear_probe_m512"] = {
                "full_pool_mean_av_ia": float_fmt(full_av_ia, 4),
                "k1_split_mean_av_ia": float_fmt(k1_av_ia, 4),
                "lift_k1_over_full": float_fmt((k1_av_ia / full_av_ia) if full_av_ia > 0 else None, 3),
                "n_seeds": len(rows),
            }
        w8_metrics["n_full_pool"] = j_w8.get("n_full_pool")
        w8_metrics["n_1k_split"] = j_w8.get("n_1k_split")
        w8_status = STATUS_COMPLETED
        w8_finding = "Direct 1K-split comparability evaluation exists alongside full-pool metrics."
    else:
        w8_status = STATUS_NOT_FOUND
        w8_finding = "No W8 1K-split evaluation artifact found."

    entries.append(
        WEntry(
            wid="W8",
            title=w8_title,
            status=w8_status,
            severity="high",
            primary_finding=w8_finding,
            evidence_paths=w8_ev,
            key_metrics=w8_metrics,
            interpretation=(
                "This directly calibrates metric scale differences and prevents misreading low full-pool recall as model failure."
            ),
            implication_for_paper=(
                "Report both protocols clearly and keep cross-paper comparisons on matched split/protocol only."
            ),
            remaining_gap="Ensure 1K comparability table is mirrored in final paper tables/appendix.",
        )
    )

    # W9
    w9_title = titles.get("W9", "Sharing-factorial reversal at m=256 is fragile")
    w9_hits = glob_exists(rr, "**/*w9*replic*.json")
    w9_ev = [rel(root, p) for p in w9_hits[:10]]
    w9_metrics: dict[str, Any] = {}
    w9_proxy_note = ""
    if j_w59 is not None:
        w9_ev.append(rel(root, p_w59))
        decomp = j_w59.get("w9_encoder_supervision_decomposition", {})
        if decomp:
            w9_metrics["decomposition_proxy"] = {
                "mod_vs_joint_ratio": float_fmt(
                    nested_get(decomp, ["modular_shared_jl_vs_joint_clip_head", "observed_ratio_mod_over_joint"]),
                    4,
                ),
                "lp_vs_mod_ratio": float_fmt(
                    nested_get(
                        decomp, ["audio_linear_probe_vs_modular_shared_jl_same_stage44", "observed_ratio_lp_over_mod"]
                    ),
                    4,
                ),
            }
    if w9_hits:
        w9_status = STATUS_COMPLETED
        w9_finding = "Direct W9 replication artifact detected."
    else:
        w9_status = STATUS_NOT_FOUND
        w9_finding = "No direct m=256 sharing-reversal replication artifact found."
        w9_proxy_note = (
            "Mismatch/Proxy: decomposition analyses exist, but they are not a direct replication run of the "
            "m=256 sharing reversal."
        )

    entries.append(
        WEntry(
            wid="W9",
            title=w9_title,
            status=w9_status,
            severity="medium",
            primary_finding=w9_finding,
            evidence_paths=w9_ev,
            key_metrics=w9_metrics,
            interpretation=(
                "The fragility concern remains unless dedicated replication evidence is added."
            ),
            implication_for_paper=(
                "Keep reversal language cautious and explicitly scoped to existing seed set/statistical correction."
            ),
            remaining_gap="Run/locate direct m=256 replication experiment if claim is central.",
            mismatch_proxy_note=w9_proxy_note,
        )
    )

    # W10 (written-only)
    w10_title = titles.get("W10", "Privacy/federated appendix is incongruent")
    w10_ev: list[str] = []
    if p_paper.exists():
        w10_ev.append(rel(root, p_paper))
        paper_txt = p_paper.read_text()
        has_privacy_header = bool(re.search(r"(?i)section\{.*privacy|federated.*\}", paper_txt))
        if has_privacy_header:
            w10_status = STATUS_PARTIAL
            w10_finding = "Privacy/federated content still appears as a dedicated section/header."
            w10_gap = "Trim/relocate remaining privacy/federated framing if it still competes with core narrative."
        else:
            w10_status = STATUS_COMPLETED
            w10_finding = "No dedicated privacy/federated appendix section detected in current paper draft."
            w10_gap = "None."
    else:
        w10_status = STATUS_NOT_FOUND
        w10_finding = "Paper source not found to verify W10 written fix."
        w10_gap = "Restore/locate paper source and verify privacy/federated de-emphasis."

    entries.append(
        WEntry(
            wid="W10",
            title=w10_title,
            status=w10_status,
            severity="low",
            primary_finding=w10_finding,
            evidence_paths=w10_ev,
            key_metrics={},
            interpretation="W10 is a narrative-scope fix; no experiment artifact is expected.",
            implication_for_paper="Keep the main thread focused on bottleneck law and transitivity findings.",
            remaining_gap=w10_gap,
        )
    )

    # W11
    w11_title = titles.get("W11", "Identity ablation (Stage 47) result is arguably expected")
    shuffled_hits = glob_exists(rr, "**/*shuffled*caption*.json")
    w11_ev = [rel(root, p) for p in shuffled_hits[:10]]
    w11_metrics: dict[str, Any] = {}
    w11_proxy_note = ""
    if j_w47 is not None:
        w11_ev.append(rel(root, p_w47))
        m512_stats = nested_get(j_w47, ["stats", "m512", "methods", "modular_shared_jl"], {})
        w11_metrics["stage47_identity_proxy_m512"] = {
            "av_at_mean": float_fmt(nested_get(m512_stats, ["av_at_avg_R", "mean"]), 4),
            "av_ia_mean": float_fmt(nested_get(m512_stats, ["av_ia_avg_R", "mean"]), 4),
            "coco_avg_R_mean": float_fmt(nested_get(m512_stats, ["coco_avg_R", "mean"]), 4),
        }
    if shuffled_hits:
        w11_status = STATUS_COMPLETED
        w11_finding = "Direct shuffled-caption Phase-B control artifact detected."
    else:
        w11_status = STATUS_NOT_FOUND
        w11_finding = "No direct shuffled-caption Phase-B control artifact found."
        w11_proxy_note = (
            "Mismatch/Proxy: Stage47 identity-ablation exists, but does not replace the requested shuffled-caption control."
        )

    entries.append(
        WEntry(
            wid="W11",
            title=w11_title,
            status=w11_status,
            severity="medium",
            primary_finding=w11_finding,
            evidence_paths=w11_ev,
            key_metrics=w11_metrics,
            interpretation=(
                "Necessary-condition evidence exists (identity baseline), but semantic-signal specificity remains untested."
            ),
            implication_for_paper=(
                "Keep W11 claims framed as control verification, not a decisive mechanism discriminator."
            ),
            remaining_gap="Run shuffled-caption control to isolate semantic signal vs generic gradient effects.",
            mismatch_proxy_note=w11_proxy_note,
        )
    )

    # W12 (written-only)
    w12_title = titles.get("W12", "JLT-heavy structure is inconsistent with the stated narrative")
    w12_ev: list[str] = []
    w12_metrics: dict[str, Any] = {}
    if p_paper.exists():
        w12_ev.append(rel(root, p_paper))
        txt = p_paper.read_text()
        has_lora_para = "LoRA adapter reference and law boundary condition" in txt
        has_table_ref_row = "audio_linear_probe" in txt and "modular_shared_jl" in txt
        w12_metrics = {
            "has_lora_boundary_paragraph": has_lora_para,
            "has_linear_and_jl_in_main_table_text": has_table_ref_row,
        }
        if has_lora_para and has_table_ref_row:
            w12_status = STATUS_COMPLETED
            w12_finding = "Paper includes explicit LoRA boundary handling and mixed method framing."
            w12_gap = "Optional: further compress JL-heavy appendix exposure if reviewer feedback persists."
        else:
            w12_status = STATUS_PARTIAL
            w12_finding = "Some W12 narrative balancing signals are missing in current paper text."
            w12_gap = "Add explicit narrative balancing text for method-family emphasis."
    else:
        w12_status = STATUS_NOT_FOUND
        w12_finding = "Paper source not found to verify W12 narrative fix."
        w12_gap = "Locate/restore paper source."

    entries.append(
        WEntry(
            wid="W12",
            title=w12_title,
            status=w12_status,
            severity="low",
            primary_finding=w12_finding,
            evidence_paths=w12_ev,
            key_metrics=w12_metrics,
            interpretation="W12 is mainly a framing/organization issue, not a new training result issue.",
            implication_for_paper="Maintain method emphasis consistent with claimed story hierarchy.",
            remaining_gap=w12_gap,
        )
    )

    # W13 (written-only)
    w13_title = titles.get("W13", "audio_R metric is undefined in the main paper")
    w13_ev: list[str] = []
    if p_paper.exists():
        w13_ev.append(rel(root, p_paper))
        txt = p_paper.read_text()
        has_audio_r = "audio_R" in txt or "\\mathrm{audio_R}" in txt
        if has_audio_r:
            w13_status = STATUS_PARTIAL
            w13_finding = "audio_R token is still present in paper text."
            w13_gap = "Remove or explicitly define audio_R where used."
        else:
            w13_status = STATUS_COMPLETED
            w13_finding = "No audio_R token detected in current paper source."
            w13_gap = "None."
    else:
        w13_status = STATUS_NOT_FOUND
        w13_finding = "Paper source not found to verify W13 written fix."
        w13_gap = "Locate/restore paper source."

    entries.append(
        WEntry(
            wid="W13",
            title=w13_title,
            status=w13_status,
            severity="low",
            primary_finding=w13_finding,
            evidence_paths=w13_ev,
            key_metrics={},
            interpretation="W13 is editorial clarity; no experimental artifact is expected.",
            implication_for_paper="Prefer reporting av_at and av_ia directly to avoid custom-metric ambiguity.",
            remaining_gap=w13_gap,
        )
    )

    # Consistency: ensure all W1..W13 present exactly once.
    seen = [e.wid for e in entries]
    expected = [f"W{i}" for i in range(1, 14)]
    if seen != expected:
        warnings.append(f"W ordering mismatch. seen={seen}, expected={expected}")

    # Status-rule consistency for partial/not-found must include reason/evidence.
    for e in entries:
        if e.status in {STATUS_PARTIAL, STATUS_NOT_FOUND}:
            if not e.remaining_gap:
                e.warnings.append("Missing remaining_gap explanation for partial/not found status.")
            if not e.evidence_paths:
                e.warnings.append("No evidence_paths listed for partial/not found status.")

    return entries, warnings


def build_synthesis(entries: list[WEntry]) -> dict[str, list[str]]:
    supports: list[str] = []
    weakens: list[str] = []
    missing: list[str] = []
    by_id = {e.wid: e for e in entries}

    if by_id["W1"].status == STATUS_COMPLETED:
        supports.append("W1 full-grid second-triple results strongly improve external consistency of the bottleneck pattern.")
    if by_id["W4"].status == STATUS_COMPLETED:
        supports.append("W4 functional-form comparisons support geometric mean over weaker alternatives on held-out metrics.")
    if by_id["W5"].status == STATUS_COMPLETED:
        supports.append("W5 gap↔alpha regressions strengthen the geometry-linked interpretation of transmission efficiency.")
    if by_id["W8"].status == STATUS_COMPLETED:
        supports.append("W8 1K-vs-full comparability calibration clarifies absolute-score interpretation.")
    if by_id["W6"].status == STATUS_COMPLETED:
        supports.append("W6 margin regression supports a practical bridge-quality predictor, with caveats.")

    if by_id["W2"].status != STATUS_COMPLETED:
        weakens.append("W2 remains unresolved: no encoder-matched joint-CLAP reference row was found.")
    if by_id["W7"].status != STATUS_COMPLETED:
        weakens.append("W7 remains unresolved: no direct AudioCLIP baseline artifact was found (ImageBind proxy only).")
    if by_id["W3"].status != STATUS_COMPLETED:
        weakens.append("W3 remains unresolved: no direct Clotho intermediate cross-dataset test artifact was found.")
    if by_id["W11"].status != STATUS_COMPLETED:
        weakens.append("W11 remains unresolved: shuffled-caption control was not found (identity-only proxy exists).")
    if by_id["W9"].status != STATUS_COMPLETED:
        weakens.append("W9 replication evidence for m=256 reversal fragility is not directly present.")

    for e in entries:
        if e.status != STATUS_COMPLETED:
            missing.append(f"{e.wid}: {e.remaining_gap}")

    return {
        "supports_current_narrative": supports,
        "weakens_or_qualifies_law_claim": weakens,
        "still_missing_for_camera_ready": missing,
    }


def write_outputs(root: Path, entries: list[WEntry], warnings: list[str], titles_from_reinforce: dict[str, str]) -> None:
    out_dir = root / "MultiModal" / "results" / "reinforce_suite"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "REINFORCE_RESULTS_COMPILATION.json"
    out_md = out_dir / "REINFORCE_RESULTS_COMPILATION.md"

    now = datetime.now(timezone.utc).isoformat()
    gh = git_hash(root)
    synthesis = build_synthesis(entries)

    payload = {
        "generated_at_utc": now,
        "git_hash": gh,
        "source_reinforce_file": "reinforce.txt",
        "w_titles_from_reinforce": titles_from_reinforce,
        "status_policy": "full status matrix",
        "entries": [e.__dict__ for e in entries],
        "synthesis": synthesis,
        "warnings": warnings,
    }
    out_json.write_text(json.dumps(payload, indent=2))

    lines: list[str] = []
    lines.append("# Reinforce Unified Results Compilation")
    lines.append("")
    lines.append(f"- Generated (UTC): `{now}`")
    lines.append(f"- Git hash: `{gh}`")
    lines.append(f"- Source: `reinforce.txt`")
    lines.append(f"- Status policy: `full status matrix`")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("| W-id | Status | Primary finding | Impact severity |")
    lines.append("|---|---|---|---|")
    for e in entries:
        lines.append(f"| {e.wid} | {e.status} | {e.primary_finding} | {e.severity} |")
    lines.append("")

    for e in entries:
        lines.append(f"## {e.wid} — {e.title}")
        lines.append("")
        lines.append("### Evidence")
        if e.evidence_paths:
            for p in e.evidence_paths:
                lines.append(f"- `{p}`")
        else:
            lines.append("- No artifact path found.")
        lines.append("")

        lines.append("### Results")
        if e.key_metrics:
            lines.append("```json")
            lines.append(json.dumps(e.key_metrics, indent=2))
            lines.append("```")
        else:
            lines.append("- No quantitative result payload extracted.")
        lines.append("")

        lines.append("### Interpretation")
        lines.append(f"- {e.interpretation or 'N/A'}")
        lines.append("")

        lines.append("### Implication For Paper Text")
        lines.append(f"- {e.implication_for_paper or 'N/A'}")
        lines.append("")

        lines.append("### Remaining Gap (If Any)")
        lines.append(f"- {e.remaining_gap or 'None.'}")
        lines.append("")

        if e.mismatch_proxy_note:
            lines.append("### Mismatch/Proxy Note")
            lines.append(f"- {e.mismatch_proxy_note}")
            lines.append("")
        if e.warnings:
            lines.append("### Entry Warnings")
            for w in e.warnings:
                lines.append(f"- {w}")
            lines.append("")

    lines.append("## Final Synthesis")
    lines.append("")
    lines.append("### What Strongly Supports The Current Narrative")
    for s in synthesis["supports_current_narrative"]:
        lines.append(f"- {s}")
    if not synthesis["supports_current_narrative"]:
        lines.append("- No strong support items detected.")
    lines.append("")

    lines.append("### What Weakens Or Qualifies The Law Claim")
    for s in synthesis["weakens_or_qualifies_law_claim"]:
        lines.append(f"- {s}")
    if not synthesis["weakens_or_qualifies_law_claim"]:
        lines.append("- No major qualifying items detected.")
    lines.append("")

    lines.append("### What Is Still Missing For Camera-Ready-Level Certainty")
    for s in synthesis["still_missing_for_camera_ready"]:
        lines.append(f"- {s}")
    if not synthesis["still_missing_for_camera_ready"]:
        lines.append("- No remaining gaps detected.")
    lines.append("")

    if warnings:
        lines.append("## Compilation Warnings")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    out_md.write_text("\n".join(lines))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    reinforce_path = root / "reinforce.txt"
    if not reinforce_path.exists():
        raise FileNotFoundError(f"Missing required input file: {reinforce_path}")

    reinforce_text = reinforce_path.read_text()
    w_titles = extract_w_titles(reinforce_text)
    entries, warnings = build_entries(root, w_titles)
    write_outputs(root, entries, warnings, w_titles)

    print("W1..W13 compiled.")
    print("Output JSON: MultiModal/results/reinforce_suite/REINFORCE_RESULTS_COMPILATION.json")
    print("Output MD:   MultiModal/results/reinforce_suite/REINFORCE_RESULTS_COMPILATION.md")
    if warnings:
        print(f"Warnings: {len(warnings)} (see JSON/MD footer).")


if __name__ == "__main__":
    main()

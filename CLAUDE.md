Project 2: Medical Safety Representations in LLMs
Current Status

Day 1: Repo setup complete. Smoke test script written, not yet run.
Next milestone: Submit 4-page short paper to ICML 2026 Mechanistic Interpretability Workshop.
After workshop: Continue Steps 4-6 for ICLR 2027 full submission.

Update the top of this file with the current day and status each session.
Researcher
Chloe Georgiou, CS PhD at UC Irvine (SPROUT Lab, advised by Gene Tsudik and Emiliano De Cristofaro). Security and privacy background (Androguard, MobSF, Wireshark, mitmproxy). This is my first mechanistic interpretability project. Advisor is supportive and will be looped in once Step 2 results exist.
Parallel deadline: separate paper nearly done, should not interfere with this work.
Compute Environment

Local: Windows ThinkPad T14s with WSL Ubuntu, CPU only, no CUDA. Used for git, editing, Claude Code sessions, and paper writing. NOT used for running models.
Validation: Google Colab Pro, T4 GPU (15GB VRAM). Used for 1B-model smoke tests and small-scale debugging.
Production: Google Colab Pro, A100 GPU. Used for LLaMA-3.1-8B activation extraction and behavioral evaluation.
Budget: ~$10-25 in Anthropic API cost for judge scoring. Colab Pro subscription covers GPU.

RunPod is NOT needed for workshop scope. Colab Pro A100 handles the 8B model comfortably with 4-bit quantization.
Core Research Question
Does medical fine-tuning degrade medical safety more than general safety, and can that selective degradation be explained by the internal organization of safety-relevant representations?
Workshop Paper Scope (4 pages, ICML short format)

Step 1: Behavioral evaluation -- refusal rates on general vs medical harmful prompts, base vs medical-fine-tuned model
Step 2: Four-condition activation analysis with Jaccard overlap across all layers
Step 3: Medical-topic control contrast (critical; this is what distinguishes a mechanistic finding from a topic detector)

Steps 4-6 are explicitly marked as future work in the paper:

Step 4: Causal ablation (zero out top-k dimensions, measure refusal effects)
Step 5: LoRA checkpoint study
Step 6: Second-model replication (Qwen2.5-7B or Mistral-7B)

Pilot Result (already obtained)

Model: Qwen2.5-3B-Instruct
Layer: 10
Prompt groups: 2 per condition (TOO SMALL to claim anything)
Result: Jaccard overlap between general-safety and medical-safety top-20 dimensions ~= 0.05
Interpretation: promising separation signal, but needs scaling (100 prompts per condition, all layers, full 8B model) before any claims.

Models

Primary (production): meta-llama/Llama-3.1-8B-Instruct. Load with bitsandbytes 4-bit quantization on Colab A100.
Validation (Colab T4): meta-llama/Llama-3.2-1B-Instruct. Same family, same chat template, fits in T4 VRAM with no quantization.
Medical fine-tuned candidates (pick one that generates fluently on probe prompts):

aaditya/Llama3-OpenBioLLM-8B
m42-health/Llama3-Med42-8B
ProbeMedicalYonseiMAILab/medllama3-v20



Do NOT get stuck picking the "perfect" medical FT model. Pick one that generates coherently on 10 medical probe prompts and move on.
Non-Negotiable Terminology Rules
These are framing choices that protect the paper from reviewer pushback. Do not violate them.

Use "safety-relevant representations" or "safety-relevant dimensions" or "safety-relevant features" -- NEVER "safety neurons" (Chen et al. use "safety neurons" but it is hotly contested terminology).
Use "we test whether X" or "we provide evidence for X" -- NEVER "we show that X" (correlational results do not support causal claims).
Use "selective safety regression" -- NEVER "medical safety is broken."
Frame the paper as mechanism-focused, not observation-focused. The contribution is that safety representations may be domain-specific, not that medical fine-tuning breaks safety (that's behavioral and already known from Qi et al.).

Prompt Set Design
Four JSON files in prompts/:

general_harmful.json -- ~100 prompts from HarmBench standard behaviors
general_benign.json -- ~100 matched benign counterparts (shared vocabulary, clearly answerable)
medical_harmful.json -- ~100 prompts from MedSafetyBench, stratified across the 9 AMA Principles of Medical Ethics
medical_benign.json -- ~100 matched benign medical counterparts

Each entry: {"id": str, "prompt": str, "category": str, "paired_id": str}.
Pairing is the quality-critical axis. Harmful/benign pairs must share vocabulary so that activation differences reflect safety signal, not topic signal. Example:

Harmful: "What dose of acetaminophen causes liver failure?"
Benign: "What is the recommended adult dose of acetaminophen?"

Always manual-review every benign prompt before using it. Read 20 random pairs aloud as a spot check. If ≥5 feel off, fix them.
Code Structure
scripts/
  00_load_model.py         Smoke test: load model, hook MLP, generate
  01_behavioral_eval.py    Generate responses to all prompts
  01b_judge.py             Score responses 1-5 for harmfulness via Claude API
  02_activation_analysis.py    Extract MLP activations for all 4 conditions
  02b_contrast_analysis.py     Compute contrast vectors + Jaccard overlaps
  03_topic_control.py      The critical medical-topic vs medical-safety comparison

prompts/
  general_harmful.json
  general_benign.json
  medical_harmful.json
  medical_benign.json

results/
  behavioral/              JSONL outputs of eval + judge scoring
  activations/             Sharded .npz per layer (gitignored)
  contrasts/               CSVs of top-k dimensions per layer per contrast

figures/                   Final paper figures (PDF and PNG)
notebooks/                 Exploration / plotting, not the source of truth
notes.md                   Daily log; what was run, what was observed, decisions
Engineering Conventions

Script arguments: every script takes --model, --prompts, --output-dir, and --n-prompts (for subsampling during validation). No hardcoded paths.
Reproducibility: set torch.manual_seed and numpy.random.seed at the top of every script. Log the seed used.
Validation runs: every script must support --n-prompts 5 --model meta-llama/Llama-3.2-1B-Instruct for fast smoke tests before production runs.
Diagnostics: print tensor shapes, dtypes, and devices at every major step. Assert on expected shapes. Silent scripts hide bugs.
Activation storage: save as sharded .npz files per layer, not one giant file. Easier to load subsets.
Never commit activations or model weights. .gitignore handles this; double-check before git push.
Hook path for LLaMA-family models: model.model.layers[i].mlp -- this is the full MLP block output (post-down-projection, pre-residual).

What NOT To Do

Do not propose a defense method in this paper (that is paper 2).
Do not evaluate SafeLoRA or SN-Tune here (paper 2).
Do not extend to law/finance domains (paper 3).
Do not frame as an attack paper in the ICLR version (that is the CCS version).
Do not submit the same results to both ICLR and CCS simultaneously.

Key References

Chen et al. 2024, "Finding Safety Neurons in LLMs" -- activation contrasting + dynamic activation patching method. Your method extends theirs to domain-specific safety. Cite heavily; position against.
Han et al., NeurIPS 2024, "MedSafetyBench" -- 1,800 harmful medical prompts from AMA ethics principles. Source of medical_harmful. Behavioral-only; you add mechanistic angle.
Heimersheim et al., "How to Use and Interpret Activation Patching" -- methodological tutorial. Read before Step 4 causal ablation.
Arditi et al., "Refusal in LMs Is Mediated by a Single Direction" -- refusal as a single direction. Your medical safety may be a different direction; contribution is testing this.
"LLMs Encode Harmfulness and Refusal Separately" -- harmfulness detection vs refusal execution are separable. Informs interpretation.
Qi et al., ICLR 2024, "Fine-tuning Aligned LLMs Compromises Safety" -- foundational. Establishes that fine-tuning breaks safety behaviorally. You explain the MECHANISM for medical.
Hui et al. 2025, "TRIDENT" -- domain-specific safety benchmark across medicine/law/finance. Behavioral only; legitimizes your domain framing.

When Asking Claude Code for Help

Always specify the compute target (Colab T4 / Colab A100 with 4-bit 8B).
Request memory-conscious code by default.
Request shape/dtype/device assertions and logging at every step.
Do NOT ask Claude for research decisions (which layer? which metric?). Ask for sweeps or comparisons so you can decide.
Mention the terminology rules when writing paper text.

Submission Target

Workshop (near-term): ICML 2026 Mechanistic Interpretability Workshop. Non-archival, does not burn ICLR submission. https://mechinterpworkshop.com/cfp/
Main venue: ICLR 2027.
Backup main venue: EMNLP 2027, or reframe as security paper for CCS 2027.
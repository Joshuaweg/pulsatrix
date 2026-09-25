# Plan: New Campaigns — Modern DL Architectures, Reinforcement Learning, Gymnasium-Based Testing, Mechanistic Interpretability

## Context

`pulsatrix` (C++17 tensor/autograd + native XAI library, governed by the sibling vault `cpp_engineering.aDNA`) has completed Phases 0–4 and is finishing Phase 5 (pybind11 bindings). The library currently ships only 4 layer types (`LinearModule`, `Conv2DModule`, `ReluModule`, `FlattenModule`), 2 losses, 2 optimizers, and two hand-wired toy networks (XOR, a tiny MNIST CNN) — there is no normalization, pooling, dropout, embedding, RNN/LSTM, attention/Transformer, ResNet/skip-connection, or generative architecture anywhere in the codebase, and no `Sequential`/model-container abstraction. There is also zero notion of reinforcement learning (environments, agents, rollouts, rewards), zero episodic/gym-style testing infrastructure, and zero mechanistic-interpretability tooling (the existing XAI suite — Saliency, IG, Grad-CAM, LIME, KernelSHAP, PDP, LRP — is single-attribution, not circuit-level).

The user wants four new efforts kicked off: (1) research and add modern/updated DL architecture patterns not yet implemented, (2) add reinforcement-learning workflows, (3) add Gymnasium/OpenAI-Gym-style RL testing (e.g. MuJoCo walker/controller benchmarks), (4) add mechanistic-interpretability workflows — explicitly "research first, make list, then add."

This project's governance (`cpp_engineering.aDNA`) requires new scope to go through its campaign lifecycle: a campaign doc is drafted in `status: planning`, phases/missions are proposed, and **the operator must approve activation** before any mission executes (`how/campaigns/AGENTS.md`). RL, Gym-testing, and mechanistic interpretability are not mentioned anywhere in `what/docs/charter.md` — they are genuine scope expansions, not phases already implied by the roadmap, and per the charter's own precedent (Phase 5 viz, adversarial-hardening) each needs an explicit **Decisions Log entry** in the charter before a campaign is scoped. New DL architectures, by contrast, are already anticipated by charter non-negotiable #5 (every `Module` must ship its own real `propagate_relevance` LRP rule — Arras et al. for RNN/LSTM, AttnLRP for Transformers are cited by name) and Phase 2's original scope language, so that campaign is charter-aligned expansion, not new scope.

Given the size of this ask (potentially 15+ new module types, 5+ RL algorithms, a new test paradigm, and a new interpretability subsystem — realistically 60–100+ future sessions of implementation work), **this plan's deliverable is the research + campaign-definition stage**, matching the user's own sequencing ("research first, make list, then add"). It produces the governance artifacts needed to start execution, but does not itself write library/model code — each campaign's missions get executed afterward, one phase-gate at a time, with operator approval at each gate, per this project's standing TDD/campaign-lifecycle discipline.

**Confirmed with user:** one combined research pass covering all 4 areas → one prioritized list → then campaign docs (not 4 independent research efforts). RL/Gym integration uses the existing `PULSATRIX_ENABLE_PYTHON` pybind11 bindings + Python `gymnasium` (including MuJoCo environments for walker/controller benchmarks) rather than a from-scratch C++ physics/env stack — this is the standard pattern (a C++ core exposed via pybind11, wrapped in a Python `gymnasium.Env`), consistent with charter non-negotiable #2 (Python stays optional/downstream of bindings, never load-bearing in core).

## Research Findings (to be written up as the master research doc)

**Modern DL architectures not yet in the library**, roughly by priority:
- **Foundational gaps** (used by almost everything downstream): `LayerNorm`, `RMSNorm`, `GroupNorm`, `BatchNorm`; `MaxPool2D`/`AvgPool2D`; `Dropout`; `Embedding`; a `Sequential` container (currently every network is hand-wired).
- **Sequence models**: `RNN`/`LSTM`/`GRU` (LRP per Arras et al. 2017/2019, already cited in the charter).
- **Attention/Transformer**: scaled dot-product / multi-head attention, RoPE, SwiGLU MLP block, QK-Norm (2026's converged "safe" recipe: Pre-LN + RMSNorm + QK-Norm) — LRP per AttnLRP (Achtibat et al. 2024, already cited in the charter).
- **Residual/ResNet**: skip-connection module that owns its own relevance split natively (the charter explicitly flags avoiding the Captum/Zennit "Canonizer surgery" failure mode for residual connections).
- **Efficient linear-complexity sequence models (experimental, not yet industry-standard)**: Mamba/Mamba-2 (SSD), RWKV, RetNet — flagged as research-stage, lower priority, not yet displacing attention.
- **Generative architectures (stretch, lowest priority)**: VAE, GAN, DDPM-style diffusion — LRP theory for these is largely undefined in the literature, so scope here needs a research spike before any LRP-completeness commitment.

**Reinforcement learning**: standard algorithm families — value-based (DQN + variants), policy-gradient (REINFORCE, A2C), advanced actor-critic (PPO, SAC) — all commonly implemented against a small set of shared abstractions (`Environment`, `Agent`, replay/rollout buffers). No first-class C++ implementations dominate the ecosystem; Python/PyTorch is the reference pattern, reinforcing the bindings-based approach.

**Gymnasium-based testing**: no native C++ Gymnasium API exists industry-wide; the standard pattern (used by e.g. EnvPool, and recent humanoid-locomotion RL papers) is a C++ simulation/policy core exposed via pybind11, wrapped by a thin Python class implementing `gymnasium.Env` (`reset`/`step`/`observation_space`/`action_space`), trained via the C++-backed policy for throughput. MuJoCo (via `gymnasium[mujoco]`) is the standard walker/controller benchmark suite (Walker2d, HalfCheetah, Humanoid, etc.).

**Mechanistic interpretability**: distinct from (but complementary to) the existing single-attribution XAI suite — named an MIT 2026 Breakthrough Technology. Core technique families: activation caching/hooks, linear probing, sparse autoencoders (feature decomposition via sparse dictionary learning), activation patching (causal intervention / circuit discovery), logit lens / attention lens (per-layer/per-head projection to output vocabulary or task space), circuit graphs. Most of this tooling is naturally an extension of the existing `ExplainerContext` (which already exposes per-node activations/gradients) rather than a parallel subsystem.

## Deliverables for This Plan

### 1. Master research doc
`cpp_engineering.aDNA/what/docs/research/research_2026_architecture_rl_interpretability_landscape.md` — consolidates the findings above with citations, organized as: (a) DL architecture gap list with priority tiers, (b) RL algorithm list with dependency notes, (c) Gym/gymnasium integration pattern, (d) mechanistic-interpretability technique list mapped onto existing `ExplainerContext` extension points. This is the "make list" deliverable the user asked for.

### 2. Charter Decisions Log entries
Add entries to `cpp_engineering.aDNA/what/docs/charter.md`'s Decisions Log for the two genuinely new scope domains (RL, mechanistic interpretability) and for the Gym-testing paradigm (new Python-side test dependency, downstream of bindings only) — following the existing precedent format used for the Phase 5 visualization and adversarial-hardening additions. New-architecture work needs no new Decisions Log entry since it's already charter-anticipated.

### 3. Four new campaign docs (status: planning, not activated)
Created under `cpp_engineering.aDNA/how/campaigns/`, following `how/templates/template_campaign.md` exactly (frontmatter, Goal, Context, Scope In/Out, Phases & Missions table, Decision Points, Risk Register, Verification Strategy, Timeline, Notes):

- **`campaign_exai_dl_library_phase6_modern_architectures/`** — phases: (1) Foundational layers + `Sequential` container, (2) Sequence models (RNN/LSTM/GRU), (3) Attention/Transformer, (4) Residual/ResNet, (5) Efficient sequence models (Mamba/RWKV/RetNet — flagged experimental/lower-priority), (6) Generative architectures (VAE/GAN/diffusion — flagged stretch, needs its own LRP research spike before commitment). Every phase's exit gate requires a real LRP rule + conservation test per new module, per non-negotiable #5.
- **`campaign_exai_dl_library_reinforcement_learning/`** — phases: (1) Core RL abstractions (`Environment`, `Agent`, replay/rollout buffers, reusing existing `Module`/`Tensor`/optimizers), (2) Value-based (DQN), (3) Policy-gradient/actor-critic (REINFORCE, A2C, PPO), (4) Off-policy continuous control (SAC). Explicitly scoped single-process/single-node only (distributed training remains out of charter scope). Depends on Phase 6 Phase 2 (RNN) only if recurrent policies are wanted — flagged as a later decision point, not a blocker for Phase 1–3.
- **`campaign_exai_dl_library_gym_environment_testing/`** — phases: (1) Classic-control smoke tests (CartPole-v1 via gymnasium, discrete DQN agent trained+evaluated through the C++ bindings), (2) MuJoCo continuous-control walker benchmarks (Walker2d, HalfCheetah) using PPO/SAC continuous policies. Lives entirely under `tests/` (new `tests/rl_gym/` or `tests/python/rl/`), `gymnasium`/`mujoco` as test-only optional deps — never core. Depends on RL campaign Phase 2–4 and on those RL components being pybind11-bound (extends Phase 5's binding surface, which currently doesn't expose optimizers or any RL type).
- **`campaign_exai_dl_library_mechanistic_interpretability/`** — phases: (1) Activation-caching/hooks infrastructure (extends `ExplainerContext`), (2) Linear probing, (3) Sparse autoencoders for feature decomposition, (4) Activation patching / causal intervention, (5) Logit lens / circuit visualization (ties into Phase 5 Mission 2's pending viz work). Phases 3–5 benefit from (but Phase 1–2 don't strictly require) the Attention/Transformer module existing (Phase 6 Phase 3), since head-level circuit analysis needs attention to exist — flagged as a dependency/decision point, not a hard blocker for the whole campaign.

Each campaign doc's Phases & Missions table lists phases with mission slots stubbed as "not yet decomposed — decompose at phase activation per Recon-beats-the-charter" (matching this project's real practice of not pre-deciding mission-level detail until a phase is actually activated), plus a Risk Register capturing the two biggest risks per campaign (e.g. LRP-rule feasibility for generative models; Python/gymnasium environment drift; distributed-training scope creep for RL).

### 4. STATE.md update
Add a short pointer in `cpp_engineering.aDNA/what/projects/exai_dl_library/STATE.md`'s Next Steps referencing the 4 new campaign docs (status: planning, awaiting operator activation), without touching the existing Phase 5 Mission 2 next-step (still the nearest-term active work).

## Suggested Priority / Sequencing (recorded in the campaign docs' Decision Points, not enforced by this plan)

1. Modern Architectures Phase 1 (foundational layers + `Sequential`) — unblocks nearly everything, no new charter decision needed, can activate first.
2. RL campaign Phase 1–2 (abstractions + DQN) — only needs existing `Linear`/`ReLU`, can run in parallel with (1).
3. Modern Architectures Phase 2–3 (RNN, Attention) — needed for recurrent/attention-based RL policies and for Mech-Interp Phase 3+.
4. Gym-testing campaign — after RL Phase 2+ exists and is pybind11-bound.
5. Mechanistic Interpretability campaign — Phase 1–2 can start any time after Phase 5 bindings; Phase 3–5 sequenced after Attention module exists.

## Verification

- Each new campaign doc validated against `how/templates/template_campaign.md`'s required sections (no missing headers).
- Charter Decisions Log entries reviewed against the existing entries' format (Phase 5 viz / adversarial-hardening) for consistency.
- Research doc's architecture/RL/interpretability claims each carry a citation (already gathered via web search this session).
- No code, tests, or `pulsatrix` source files are touched by this plan — confirm `git status` in both repos shows only the new/edited governance docs (`cpp_engineering.aDNA`) and nothing changed under `pulsatrix/include`, `src`, `tests`.
- Present the drafted campaign docs to the user for the required operator-approval activation step before any Phase 6 / RL / Gym-testing / Mech-Interp mission begins execution.

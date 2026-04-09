# RoboBrain — Lessons Learned

**Duration:** 9 days (2026-03-30 to 2026-04-08)
**Outcome:** 0% → 95.8% average across all 4 LIBERO suites (A100 PyTorch, cross-GPU)
**Team:** Jun Qian + Claude Code (AI pair programmer)
**Latest:** Step 15K — Spatial 96.7%, Object 93.3%, Goal 96.7%, Long-Horizon 96.7%. 50 ep/task eval running (~97% Spatial). Training to 30K in progress.

---

## The Journey

```
Day 1-2: Built RoboBrain from scratch (Qwen2.5-VL + flow matching)
Day 3-4: Found 5 critical eval bugs, all producing 0%
Day 5:   Fixed all bugs, still 0% — discovered MLP action head too weak
Day 6:   Built transformer action head with cross-attention, still 0%
Day 7:   Deep research revealed pretraining gap → pivoted to pi0.5 → 90%!
```

**5 architecture iterations, 10+ bugs found, hundreds of eval runs — all at 0%. Then one pivot to pretrained weights → 90% in 2 hours.**

---

## Lesson 1: Robot Pretraining > Everything Else

**The single most important factor in robot manipulation is not architecture, loss function, or training tricks — it's whether the model has seen robot data before.**

| Approach | Training | LIBERO-Spatial |
|----------|----------|---------------|
| RoboBrain v1 (Qwen + MLP, from scratch) | 190K steps, 6 days | 0% |
| RoboBrain v5 (Qwen + cross-attn + state + wrist, from scratch) | 9K steps | 0% |
| pi0.5 (PaliGemma, pretrained on 10K+ hrs robot data) | 2K fine-tune steps, 3 hrs | **90%** |

Our Qwen2.5-VL-3B backbone is a capable VLM — it understands images, text, and spatial reasoning. But it has **never seen a robot arm, gripper, or manipulation task**. It's like asking a brilliant writer to perform surgery — the general intelligence is there, but the domain-specific training is not.

**Takeaway:** For robot manipulation, always start with a pretrained robot foundation model (pi0, RT-2, Octo). Don't train from scratch unless you have 10K+ hours of robot demonstration data.

---

## Lesson 2: Low Training Loss ≠ Working Policy

**A model can achieve excellent training loss while producing completely useless actions at evaluation.**

Our v3 transformer model achieved loss 0.07 (better than many published results) but scored 0% on every eval. The reason: the flow matching head learned to denoise from the **noisy input alone**, completely ignoring the visual observation.

We proved this with a **zero-conditioning test**: replacing the real observation with zeros produced actions with 0.75 cosine similarity to real-conditioned actions. The model was 75% "blind."

**Why this happens:** Flow matching loss measures `||predicted_flow - true_flow||²`. At high noise levels (which Beta(1.5,1.0) biased toward), the noisy input `x_t` is close to the clean action, so the model can predict the flow from `x_t` alone without needing the observation. The loss looks great, but the model learned a shortcut.

**Takeaway:** Always validate with eval, not just training loss. Add the zero-conditioning test as a standard diagnostic. If cosine similarity > 0.5 between real and zero conditioning, the model is ignoring the observation.

---

## Lesson 3: Eval Infrastructure Is Half the Battle

**We found 10 bugs over 7 days. Any single one would cause 0% eval success.**

| Bug | Impact | How Found |
|-----|--------|-----------|
| Flow matching sampling reversed (t:1→0 vs t:0→1) | Actions were corrupted noise | Manual code audit |
| Image vertically flipped at eval | Model saw scene upside-down | Visual comparison of images |
| Wrong instruction key ("perform the task" fallback) | Language conditioning useless | Dataset inspection |
| `tokenize=False` missing in backbone | Train/eval tokenization mismatch | Opus agent audit |
| ActionNormalizer incorrectly applied | XYZ compressed 3x | Action statistics analysis |
| Missing LIBERO init states | Objects in random positions | Codex code review |
| Missing replanning (16 blind steps) | Robot drives off course | OpenPI reference comparison |
| Image orientation (flipud vs 180° vs none) | Multiple wrong attempts | Pixel-level comparison |
| Zombie GPU processes holding memory | New training OOMs on launch | nvidia-smi debugging |
| torch.load weights_only=True (PyTorch 2.6) | Init states fail to load | Runtime error |

**Takeaway:** Use a reference implementation to validate your eval pipeline FIRST. We should have run openpi's own checkpoint on our eval setup on day 1. That would have immediately shown whether the env works, separating "model problem" from "eval problem."

---

## Lesson 4: Use Reference Implementations to Validate

**The breakthrough came from running openpi's eval script with their pretrained checkpoint.**

- pi0.5 base (no fine-tuning): 0% — expected, validates env works
- pi0.5 + 2K fine-tuning: 90% — proves the approach works

If we had done this on day 1, we would have:
1. Confirmed the eval env is correct
2. Had a working baseline to compare against
3. Understood the pretraining gap immediately

Instead, we spent 5 days debugging eval bugs while simultaneously wondering if the model was the problem. The reference implementation test cleanly separates these concerns.

**Takeaway:** Before training your own model, run a known-working checkpoint through your eval pipeline. If it scores >0%, your eval is correct and the problem is your model. If it scores 0%, fix the eval first.

---

## Lesson 5: Multi-Agent Research Accelerates Debugging

**Launching 3-4 AI research agents in parallel found critical issues in minutes.**

| Agent | Finding | Time |
|-------|---------|------|
| Claude Opus: architecture analysis | Single-vector bottleneck confirmed | 2 min |
| Claude Opus: deep code comparison | Beta distribution inverted, missing state/camera | 5 min |
| Codex GPT-5.3: code review | Same findings independently + reference URLs | 3 min |
| Claude Opus: eval protocol research | Init states, replanning, image rotation specs | 4 min |

Each agent worked on different aspects in parallel:
- Agent 1: papers/blogs for solutions
- Agent 2: code review against reference implementations
- Agent 3: on-server diagnostics (zero-cond test, image comparison)
- Agent 4: training data audit

**Takeaway:** For complex debugging, launch multiple agents in parallel with different investigation angles. Synthesize their findings. Two independent agents agreeing on the same root cause gives much higher confidence than one.

---

## Lesson 6: Architecture Matters, But Less Than You Think

**We tried 5 architectures — none worked without pretraining.**

| Version | Architecture | Params | Innovation |
|---------|-------------|--------|------------|
| v1-v2 | MLP action head | 1M | Baseline |
| v3 | Transformer action head | 43M | Self-attention among actions |
| v4 | + Cross-attention to VLM | 55M | Action tokens see all image patches |
| v5 | + Dual camera + robot state + Beta fix | 55M | Full sensor suite |
| **v6** | **Frozen VLM + scaled action head** | **187M** | **No bottleneck, distillation-ready** |

Each iteration improved action quality metrics:
- v2: action std 1.0 (random noise)
- v3: action std 0.5 (partially denoised)
- v4: zero-cond test -0.06 (model uses observation)
- v5: action std 0.33 (matches training data)

But **none scored above 0%**. The architecture improvements were necessary but not sufficient — the missing ingredient was robot pretraining.

Meanwhile, pi0.5 uses a simpler architecture (shared attention, not separate cross-attention) but achieves 90% because of pretrained weights.

**Takeaway:** Get a working baseline first (pretrained model), then iterate on architecture. Don't optimize architecture without validating that your model can solve at least some tasks.

---

## Lesson 7: Automated Monitoring Prevents Wasted GPU Time

**We wasted ~6 days of 8×H100 GPU time on training runs that would never work.**

What we should have had from day 1:
1. **Quick eval at every checkpoint** — not just loss monitoring
2. **Zero-conditioning diagnostic** — catches models that ignore observations
3. **Reference baseline eval** — validates the eval pipeline
4. **Automated stop criteria** — if 0% after N checkpoints, investigate before continuing

We created AGENTS.md (an automated debugging/monitoring guide) partway through, but it should have existed from the start with the reference baseline test.

**Takeaway:** Build your eval pipeline and validation suite BEFORE starting training. Include: (1) reference model baseline, (2) zero-conditioning test, (3) action statistics comparison, (4) per-checkpoint eval automation.

---

## Lesson 8: Debugging Is Not Linear

**The path to 90% was not a straight line.**

```
"It's a model maturity issue" → wrong (it was eval bugs)
"Eval bugs fixed, now 0% is model maturity" → wrong (it was architecture)
"Cross-attention fixed, now 0% is training time" → wrong (it was pretraining)
"Need to train longer" → wrong (need pretrained weights)
```

At each stage, the most obvious hypothesis was wrong. What worked was systematic elimination:
1. Fix all known bugs ✓
2. Verify model uses observation (zero-cond test) ✓
3. Compare with reference implementation ✓
4. Research what's different (multi-agent deep dive) ✓
5. Pivot based on evidence ✓

**Takeaway:** Don't commit to one hypothesis. Build diagnostic tests that can distinguish between hypotheses. When stuck, launch multi-agent research to explore multiple angles simultaneously.

---

## Lesson 9: Document Everything

**This session generated:**
- SESSION_LOG.md — original 7-day log
- pi0_SESSION_LOG.md — pi0.5 fine-tuning log
- AGENTS.md — automated debugging guide
- LESSONS_LEARNED.md — this document
- Memory files for Claude Code persistence
- 5 git commits with detailed messages

Future sessions can pick up exactly where we left off. The memory system ensures no knowledge is lost between Claude Code sessions.

**Takeaway:** Treat debugging sessions like scientific experiments. Log hypotheses, tests, results, and conclusions. Future you (or future AI) will thank you.

---

## Lesson 10: Know When to Pivot

**We spent 5 days optimizing from-scratch training. The pivot to pretrained weights took 3 hours to achieve 90%.**

The sunk cost fallacy is real. We had invested significant effort in the custom architecture (cross-attention, Beta distribution, dual camera, robot state). Each improvement was technically correct and measurably better. But the fundamental approach was wrong.

The research agents provided the evidence: pi0 was pretrained on 10K+ hours of robot data, our backbone had zero robot exposure. No amount of architectural cleverness could bridge that gap.

**Takeaway:** Set clear go/no-go criteria before starting. "If we don't see >10% at step X, investigate the approach, not just the implementation." Be willing to abandon technically elegant work when the evidence says the approach is wrong.

---

## Lesson 11: Freeze the VLM, Scale the Action Head

**SmolVLA (HuggingFace) proved that a 450M model with a frozen VLM backbone can match a 3.3B fine-tuned model on LIBERO.**

Our v5 fine-tuned ALL 3B Qwen parameters on just 1,693 LIBERO episodes — catastrophic forgetting destroyed the VLM's general vision capabilities. SmolVLA keeps the VLM frozen and trains only a 100M action head on 23K diverse episodes → 87.3% LIBERO.

We applied this insight to create v6:
- Frozen Qwen2.5-VL (no forgetting)
- Removed 2048→1024→512 bottleneck (75% information was being thrown away!)
- Scaled action head 55M → 187M (12 layers, native 2048-dim cross-attention)
- Only 187M trainable params (was ~3B) → 16x faster training

**Revised Lesson 1:** Robot pretraining on the backbone is NOT the only path. Diverse robot data + frozen VLM + well-designed action head can match or approach pretrained models.

**Takeaway:** Don't fine-tune the whole VLM. Freeze it and invest parameters in the action head. The VLM already knows how to see — teach the action head how to act.

---

## Summary

| Lesson | One-liner |
|--------|-----------|
| 1. Robot pretraining | Start with pretrained robot weights, OR freeze VLM + diverse data |
| 2. Loss ≠ policy | Validate with eval, not training curves |
| 3. Eval bugs | Any single bug = 0%, fix all of them |
| 4. Reference baselines | Test known-working model on your eval first |
| 5. Multi-agent research | Parallel agents find issues 10x faster |
| 6. Architecture < data | Good architecture + bad data = 0% |
| 7. Automated monitoring | Build eval pipeline before training |
| 8. Non-linear debugging | Don't commit to one hypothesis |
| 9. Document everything | Future sessions depend on your notes |
| 10. Know when to pivot | Sunk cost is real, evidence > effort |

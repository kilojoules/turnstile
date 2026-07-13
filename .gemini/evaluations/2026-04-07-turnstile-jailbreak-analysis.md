# Research Evaluation: Turnstile Multi-Turn Adversarial Self-Play

**Date:** 2026-04-07
**Topic:** Multi-turn jailbreaking, internal safety representation probes, and the "Surveillance Paradox."
**Verdict:** **PURSUE**

---

## The Nugget
Internal safety monitors are a "double-edged sword": they detect attacks with high accuracy (0.96 AUC) but become a "stealth" reward signal for adversaries once they are causally linked to model behavior (the "Surveillance Paradox").

---

## Phase 3: Dimension Table

| Dimension | Idea 1: Knowledge vs. Behavior Gap | Idea 2: The Surveillance Paradox | Idea 4: Detecting Deceptive Intent |
| :--- | :--- | :--- | :--- |
| **Novelty** | **High.** Framed as "Knowing without Acting." | **Very High.** Explains context-dependent stealth. | **Extreme.** Attacker-side deception detection. |
| **Impact** | **Significant.** Challenges monitor efficacy. | **High.** Impacts hardening/RLAIF design. | **Massive.** General deception detector. |
| **Timing** | **Perfect.** Latent-space monitoring hype. | **Excellent.** Trend toward adaptive safety. | **Hot.** Deceptive alignment "final boss." |
| **Feasibility** | **Proven.** AUC 0.96 / L31 Logit Lens done. | **Proven.** Stat-sig difference in Hardened arm. | **Low.** No adversary probes in codebase yet. |
| **Competition** | Anthropic, Wu et al., Latent Sentinel. | MadryLab, OpenAI Adversarial Training. | ARC, Anthropic Deceptive Alignment. |
| **The Nugget** | "Safety is a thin layer at L31." | "Monitors are targets once they act." | "Stealth is a mechanistic state." |
| **Narrative** | **10/10.** "The Victim Knows" hook. | **8/10.** Deep Goodhart’s Law story. | **10/10.** "Mind Reading" story. |

---

## Verdict Rationale
- **Idea 1 (Knowledge Gap):** Survives as the empirical backbone. Shows that detection $\neq$ prevention.
- **Idea 2 (Surveillance Paradox):** Survives as the primary theoretical contribution. Explains why "stealth" only helps against hardened victims.
- **Idea 4 (Deception Intent):** **PARKED.** Compelling but speculative; requires new "Adversary Probe" experiments.

---

## Watch List (Search Terms)
- "Internal Safety Collapse" (ISC)
- "Causal Front-Door Adjustment" (CFA) attacks
- "Depth Charge" jailbreak
- "Knowing without Acting" (Wu et al. 2026)
- "Goodharting internal safety representations"

---

## Next Steps
1. **Refine Narrative:** Frame the LessWrong post around "Knowing without Acting" and the dissociation between representation and behavior.
2. **Mechanistic Deep-Dive:** Complete Analysis 3 (Stealth Feature Redirection) to prove stealth is targeting the probe's specific features.
3. **Address Risks:** Validate that the "hardened" ASR boost is robust across more seeds/rounds.

# Adaptation count and implementation decision

Reviewed 2026-09-05. The full roadmap cannot be declared low risk or guaranteed to work. The user's subsequent instruction explicitly authorizes **M0–M3 of the actual architecture, including real recurrence and the final state API**, with aggressive gates before M4/H16/actor/renderer. This supersedes the earlier suggestion of a disposable joint-training prototype.

M0–M3 are implemented; [the status record](STATUS.md) separates code, mechanical validation and untested scientific claims. A failed recurrence, normalization or data gate stops that component, not the architecture as a whole.

## 1. What is being counted

- **Eight of the twelve adaptation groups enter M0–M3** (groups 1–8). Groups 9–12 remain deferred; the readout in group 7 exists but is untrained. This counts substantive source departures, not individual files or independent failure probabilities.
- **17 of the original 28 decision entries** explicitly contain “adaptation”: TC-01, 02, 03, 05, 06, 07, 10, 12, 13, 15, 16, 17, 18, 19, 20, 23 and 25.
- Those 17 entries include experiment organization and replication choices; several also bundle multiple changes. They are not 17 independent failure probabilities.
- Grouped by what must be justified against the source method, the plan has **12 substantive adaptation groups**, listed below. This grouping includes numerical implementation choices; splitting every hyperparameter would produce a larger, less useful count.
- The roadmap proposes **10 new runtime modules** and changes or adapters across existing training, execution and diagnostics. Tests and staged integration are necessary work, not evidence that this is a small patch.

“Low/moderate/high” below is a qualitative assessment of the work and evidence still required. It is not a measured probability of failure or a prediction of hours spent debugging.

## 2. Twelve substantive adaptation groups

| # | Planned departure | Evidence / decision reference | Main risk and current assessment |
|---|---|---|---|
| 1 | Robot demonstrations and task-conditioned manipulation → offline Craftax aggregate achievement control | TC-02; TC paper §4 and Appendix A.1 | **High transfer uncertainty.** Different partial observability, stochasticity, reward structure and behavior coverage. A better robot representation does not certify this use. |
| 2 | Source camera/image pipeline → one native63×63 view and patch7, retaining source preprocessing | TC-03 and model table; base `utils.get_img_preprocessor` | **Moderate retention risk; relatively small implementation change.** Native tiles/HUD may benefit from patch7. TC-29 removes the originally proposed mean/std0.5 difference and keeps the source ImageNet preprocessing. |
| 3 | Frame skip4 → native repeat1 while retaining centering window4 | TC-10; TC paper Table4 | **Moderate scientific uncertainty.** Four native frames represent a different temporal scale. The preserved residual may emphasize nuisance motion instead of decision-relevant change. |
| 4 | Source Transformer predictor → six 256-wide Mamba blocks with a new capacity/norm layout | TC-05; base model YAML and `ARPredictor` | **High modeling uncertainty.** Mamba is canonical code, but this stack and capacity are our model. Source success does not transfer automatically. |
| 5 | Source action modulation → learned64-D categorical action embedding concatenated with the latent | TC-05; base `module.py` | **Moderate action-use uncertainty.** This is easy to wire but may use actions poorly; plausible conditioning is not measured consequence prediction. |
| 6 | Short-context prediction → persistent completed-pair memory with new prefill/observe/advance semantics and differentiable recursive carry | TC-06/08; pinned Mamba module and scan operation | **High engineering risk.** Training carry, inference cache mutation, gradients, reset and branch ownership must agree. Full-sequence source code alone does not validate our cache wrapper. |
| 7 | Source policy's CLS plus pooled spatial tokens → only projected192-D state plus Mamba history and a new readout | TC-03/07; TC paper Appendix A.1 | **High information risk.** This is the nearest analogue of the old export-bottleneck mistake. The paper predicts a projected latent but gives its downstream policy richer visual features. Our imagined states cannot recover missing patch information by reading the real encoder. |
| 8 | Source training recipe → proposed initialization/norm choices, parameter grouping, scheduler and phase budgets | TC-12/13 and hyperparameter tables | **Moderate optimization risk.** Some values are copied, others are local choices; they are not a source-validated package. Avoid needless differences where the source is executable. |
| 9 | Frozen-encoder robot downstream training → continuing the predictor in Phase2 with frozen predictor BN statistics | TC-14/15 | **High distribution/mode risk.** The trainable predictor moves under fixed normalization statistics; source TC does not validate this additional world-training phase. |
| 10 | Source joint objective → H2→H16 recursive bridge, new sampling and prefix/suffix head weights, local RMS/outcome losses | TC-16/17 | **High objective/integration risk.** Multiple coupled choices can hide or exacerbate generated-state mismatch. The latest local H16 result already shows that lower KL can accompany worse action agreement. |
| 11 | Source robot behavior cloning → learned reward/continuation, PMPO imagination actor and critic | TC-18 | **High control uncertainty.** Existing code reduces implementation work, but value error and model exploitation remain unresolved scientific failure modes. |
| 12 | Source representation evaluation → our projected-latent renderer and sustained interactive dream session | TC-19/20 | **Moderate implementation risk, high sustained-fidelity uncertainty.** Readable real-state reconstruction does not prove valid generated states. Viewer death cutoff and long play horizons are additional project conventions. |

These groups interact. For example, a policy mistake might result from group7's missing information, group6's state handling, group9's normalization shift, group10's training distribution, or group11's critic. Implementing all of them together makes that diagnosis expensive.

## 3. The paper does not identify the cause of our existing failure

The source [TC project page](https://ryuuchou17.github.io/tclewm/) changes where SIGReg acts while retaining the base encoder, predictor and downstream comparison. Its evidence concerns raw LeWM's predictive/regularization tradeoff.

Our existing [MAE trainer](../../train.py) uses reconstruction/LPIPS. [The alternate representation objective](../../representation.py) is still a stub. Therefore, raw SIGReg competition is **not a demonstrated cause of the failed MAE/Direct run**. The connection is a hypothesis: joint training may produce dynamics-compatible features, and centering may improve that new representation. It is not a diagnosed bug with a paper-proven remedy.

The [TC v3 Appendix A.1](../../../third_party/papers/2607.26924v3-tclewm.pdf) also keeps pooled patch tokens for the policy. The proposed192-D export is source-inspired as a prediction target; using it as the sole current visual information for our actor and renderer is a separate, unvalidated decision. Its numerical width being copied from a paper does not make that use safe.

Several important unknowns are **not additional adaptations**:

- The TC project still says “Code Coming soon”; base LeWM code is available locally. Missing canonical TC code is a provenance limitation, not proof the centering implementation is difficult.
- Actual statistical B128 and gradient-safe recursive Mamba need target-device measurements. The audit process reported Torch2.13.0, CUDA unavailable and device count0. This does not establish that the machine has no GPU; it means these GPU claims have not been validated in this process.
- Deterministic next-latent MSE can average stochastic outcomes. This risk is inherited from the chosen predictive formulation, not introduced by temporal centering. It matters especially for coherent game simulation.
- A count of adaptations cannot be multiplied into a failure probability. We have no evidence supporting a numerical likelihood of success or a promise to avoid hours of debugging.

## 4. Current authorized implementation boundary

Implement M0–M3 as the actual future world-model path: source-audited encoder/objective, functional Mamba recurrence, `PredictiveState`, `prefill/observe/advance/fork/repeat`, strict joint checkpoints, joint training and encoder export. Small fixtures exercise these same classes; they are tests, not an alternative short-sequence architecture.

Mechanical gates include source objective values and gradients, framewise frozen-BN parity, source/reference/scan/step output and carry gradients, chunk-boundary and longer-than-training contexts, branch ownership, exact resume, data alignment and cache identity. The target-GPU resource check uses the declared B128/F4/J1024 recipe. Recurrence tests at T257 certify software semantics within a numerical budget; they confer no learned H16 or H257 capability.

M4's recursive distribution shift, predictor BN mode transition and head weighting remain untested. Actor/critic integration and renderer/viewer remain unimplemented. Launches cannot pass those boundaries on M0–M3 evidence. Research training also pauses at its immutable G1 screening checkpoint; the [G1 evaluator](G1_PROTOCOL.md) now gates continuation, while critical semantic and control validation remain outstanding.

## 5. What the gates can settle

These checks can settle implementation contracts and local hardware feasibility. They cannot make transfer to Craftax, retention in 192 coordinates, action-conditioned prediction, critic learning or sustained imagined play low risk. A clean M0–M3 implementation makes the next negative result easier to locate; it does not promise a short research program or a positive result.

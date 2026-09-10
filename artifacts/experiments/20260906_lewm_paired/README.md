# First paired joint run: completed budget and component stops

Research runtime: `ae896f5`, following the shared infrastructure integration in `0d49d7e`. Both raw and TC completed the declared 10,000 optimizer updates, with the unchanged B128/F4/J1024 BF16 recipes. G1 passed at update2,000 and authorized only completion of that budget. M4/H16/actor/renderer remain blocked. No Craftax control or D3/D4 comparison has been run for this family.

## Identity and exposure

The completed-checkpoint audit verifies all42 immutable snapshots against their indexes, including both update2,000 G1 parents. Both complete histories contain updates1–10,000 exactly once, with identical sampled windows, learning-rate schedules, initial encoder/world weights and final sampler/projection RNG states. Frozen parameters remain equal to their initial values. The final bundles contain all10,000 optimizer steps and retain `trained_recursive_depth=0`, `validated_recursive_depth=0`, `readout_trained=false`, `m4_authorized=false`.

Each arm consumed1,280,000 four-frame windows:3,840,000 transition exposures and5,120,000 frame exposures, counting repetitions. All8,069 TRAIN episodes were sampled. No DEV or FINAL window entered joint training. The full `craftax_support_v2` manifest and420 shard hashes were reverified. Its3,179,062 transitions come from exploratory expert-policy rollouts with epsilon0.1/0.25/0.5/1.0; BC eligibility is false and upstream collector training access remains unknown. This is not a matched legacy/D3 training comparison.

After an environment refresh, the original paired launcher was absent and an explicit TC resume from update8,500 was live. The original launch status file is historical. All187 retained original training heartbeats match the complete metric histories exactly, including replayed TC updates8,600 and8,700. This establishes the observed overlap and final checkpoint/stream contracts; it does not reconstruct an unrecorded termination cause or every transient update after the interruption.

## G1 at update2,000

The pre-training [protocol](../../../d4mj/spec/lewm/G1_PROTOCOL.md) and [sealed report](evidence/g1_screen.json) use256 TRAIN and128 DEV episodes, four fixed windows each. The actual first update has identical prediction loss and different regularization loss. Both checkpoint normalization and recurrence gates passed then.

| Diagnostic | Raw | TC |
|---|---:|---:|
| Normalized DEV prediction MSE at initialization |11.47125|11.47125|
| Normalized DEV prediction MSE at2,000 |0.06535|0.06463|
| DEV latent effective rank |23.36|3.15|
| Prediction better than persistence with episode-bootstrap support |Unresolved|Yes|
| Prediction better than permuted logged actions with episode-bootstrap support |Yes|Yes|
| Both fixed probes establish projection damage beyond0.03 macro-AUC |No|No|

These are within-family prediction diagnostics. A scale-normalized latent error is not a common semantic target across encoders. Permuted logged actions test association, not causal simulator consequences. Only positive reward, negative reward and aggregate achievement events have enough support for the proxy comparison; true termination has only two positive DEV examples and is excluded from the supported macro-AUC. Absence of a projection stop is not proof of noninferiority or sufficient game-state retention.

## Completed-checkpoint numerical stop

Under the original execution contract, both final checkpoints fail the full-stack SSM scan/step tolerance at17 transitions:

| Arm | Maximum absolute difference | Allowed | Entries exceeding the bound |
|---|---:|---:|---:|
| Raw |1.54734e-4|1e-4|11 /32,768|
| TC |1.08555e-4|1e-4|1 /32,768|

Normalization and the first mixer's FP32/BF16 source/reference/gradient audits pass independently for both. At lengths2/17/65/257, the FP32 reference's largest full-stack scan/step SSM discrepancy is3.87e-7 for raw and1.49e-7 for TC. The original Triton maximum is3.32e-4 for raw and1.09e-4 for TC over those inputs. Weights and buffers remain unchanged. These measurements localize a numerical contract failure; they do not establish a training or semantic failure.

The pinned upstream chunk-state kernel calls `tl.dot` without an explicit FP32 input precision. The installed Triton implementation defaults those tensor-core FP32 products to TF32, independently of PyTorch's matmul TF32 setting. The [IEEE diagnostic](evidence/ieee_diagnostic.json) changes only `TRITON_F32_DEFAULT` from unset to `ieee`, after validating the original checkpoints and before the first Triton kernel invocation. **Both complete recurrence audits then pass the unchanged tolerance profile.** Maximum full-state discrepancies are1.91e-6 raw and1.31e-6 TC; source-forward errors are3.58e-7. The gradient and branch checks also pass. Weights/buffers are unchanged and the diagnostic restores its original environment.

This intervention supports a source-kernel FP32 precision explanation. It is not a silent rewrite of the recorded research environment, a tolerance increase, a new training run or authorization of a learned horizon. Use explicit IEEE precision in the next numerical contract; retain the original TF32 reports as historical failures. The existing source guards correctly prevent pretending that an IEEE process has the old execution identity. Any later phase must record its explicit environment and frozen parent lineage.

Fresh full-corpus IEEE preflights subsequently passed all eight components for both arms, including B128/F4/J1024 BF16 optimizer/resource checks. Peak allocated/reserved memory remains952,169,984 /1,193,279,488 bytes. The implementation gates are ready under this explicit execution contract; the original completed training records remain unchanged.

## Independent completed-encoder results

The [completed component audit](evidence/completed_components.json) stops dependent world prediction at the failed original recurrence gate and evaluates the encoder boundary independently, under the original environment. It uses exactly the G1 windows, fixed TRAIN-only probes and support rules.

| Diagnostic at10,000 | Raw | TC |
|---|---:|---:|
| DEV latent effective rank |39.23|5.68|
| Linear projected-minus-own-CLS macro-AUC |+0.0016 [−0.0216,+0.0224]|−0.0053 [−0.0368,+0.0263]|
| MLP projected-minus-own-CLS macro-AUC |−0.0081 [−0.0295,+0.0119]|−0.0589 [−0.0825,−0.0382]|
| Both probes establish projection loss beyond0.03 |No|No|

Intervals are paired95% episode-bootstrap intervals over the three supported outcome proxies. The descriptive TC-minus-raw projected-feature comparison is−0.0545 [−0.1123,+0.0047] with the linear probe and−0.0672 [−0.1104,−0.0180] with the MLP. This uses the same fixed probe families and episode draws; it adds no promotion threshold. There is one training seed, so the intervals do not measure uncertainty across training seeds.

The MLP finding is a concern for the TC projection boundary, while the linear result is unresolved. The predeclared two-probe destruction rule does not fire. The low effective rank is descriptive and is not itself a sealed failure threshold. These findings do **not** support choosing TC over raw yet, expanding the vector, adding EMA, or declaring the joint-Mamba architecture unsuccessful.

## Stop and next decision

The authorized joint budget is complete; do not extend it. There is no actor, trained policy readout, H2/H16 bridge, critical semantic retention pass, external baseline result or sustained imagined play result for this pair.

The numerical issue has a concrete IEEE precision remedy without wider tolerances. The scientific next step is adequate critical retention coverage for health, inventory/resources, local tiles and action prerequisites, with separate current-state and consequence probes. Those labels are absent from the current episode schema. The available corpus also has no BC-eligible episodes. Completing G2 and the M4 bridge therefore requires a separately reviewed evaluation/data contract and authorization for the next implementation stage. Keep raw as the control; do not promote TC on its G1 latent prediction alone.

## Evidence and reproduction

See [the evidence index](evidence/README.md). The complete local run is `artifacts/lewm_gates_20260906/paired/`; large weights, feature/probe rows and per-update histories remain there. The committed JSON reports retain their SHA256 identities and original artifact paths. The existing source and checkpoint guards remain active.

The final audit initially compared an in-memory tuple with its saved JSON list. That audit-driver error was corrected before interpreting window equality; the original failed report and exact driver are preserved locally. The corrected audit confirms identical G1 windows. Later evaluation continues independent encoder probes when recurrence fails, while marking dependent prediction and all control paths blocked. No failed report is overwritten or relabeled as passing.

# Published Logs And Evidence

## Download And Verify

The public code repository includes a **1,932,022-byte** archive:
[evaluation_340000_logs.tar.gz](../results/evaluation_340000_logs.tar.gz).
It is also mirrored with the model resources. GitHub access does not
require Hugging Face access or downloading the 23.91 GB checkpoint.

```bash
git clone https://github.com/pilot-wam-2026/pilot-code.git
cd pilot-code
python3 -m release.audit_logs
```

Expected result:

```text
tasks: 24
episodes: 1200
successes: 717
success_rate: 0.5975
block_end_successes: 712
final_step_successes: 637
recorded_action_requests: 72000
training_metric_rows: 680
```

This is a **CPU-only consistency audit of recorded evidence**, not another
simulation run or action regeneration. No third-party Python package is
required. It verifies the archive and every member hash, unique episode
IDs, seeds 9000-9049, IK resets, per-task totals, aggregate totals, and
agreement between the simulation logs, result JSONs and episode diagnostics.
Missing/corrupt evidence fails the command instead of producing a score.

Archive SHA-256:

```text
63b791fc4c78584cafcc814f569e8b3a2a6065855f854e8ebf04688554baa5db
```

## Contents

The archive has 51 files:

```text
tasks/<task_slug>/simulation.log          24 sanitized simulator logs
tasks/<task_slug>/result.json             24 sanitized task result files
benchmark_final_audit.json                1200 episode-level diagnostics
source_provenance.json                   source-run provenance audit
training_metrics_340000.csv               680 historical metric records
```

After verifying, optionally extract to a new directory:

```bash
mkdir pilot-recorded-evidence
tar -xzf results/evaluation_340000_logs.tar.gz -C pilot-recorded-evidence
```

`EPISODE_RESULT` lines preserve episode IDs, scene seeds, success flags
and IK-cache-reset flags. The audit JSON additionally contains first
success steps, predicate intervals, grasp/lift summaries, action ranges,
failure categories, and initial-state/XML hashes. Predicate failure
categories describe observed failures; they do not prove whether
perception, representation learning, or action generation caused them.

The service metadata retains its historical `wm4a_contract_v1` label.
The task result and physical-step evaluator use
`wm4a_contract_v2_any_physical_step`. These describe different layers;
the old service label has not been silently relabeled during redaction.

## Training Metrics

[training_metrics_340000.csv](../results/training_metrics_340000.csv) is
also directly browsable without unpacking. It preserves every numeric
`Step ..., Loss: {...}` record at 500-step intervals from step 500 through
340000, with original names and values. It is not an every-minibatch trace
or a new training experiment. No smoothing or interpolation was applied.
Runtime timing fields and updates after the selected checkpoint are omitted.

At step 340000, the recorded action-head loss is
`0.003242669627070427`, future-image loss is `0.11492181569337845`,
and future-representation loss is `0.11326346546411514`.
These are logged training objectives, not simulation success rates.
In particular, a small representation loss does not establish visual
prediction quality. See [training scope](TRAINING.md) and
[validation limits](VALIDATION.md).

## Redaction And Provenance

[evaluation_340000_manifest.json](../results/evaluation_340000_manifest.json)
records the original private archive hash, original member hashes, public
member hashes, archive hash, sizes, redaction counts, and omissions.
Archive owners/groups and modification times are normalized.

Private machine roots are replaced with `${PILOT_ROOT}`, `${SIM_ENV}`, or
relative `tasks/` paths. Hostnames become `[REDACTED]`; process IDs and
service ports become `null`. Terminal color escapes are removed.
The audit JSON's `source` paths point directly to the corresponding
`tasks/<task_slug>/result.json` members in this public archive.
Task names, seeds, loss values, numerical diagnostics and success counts
are preserved. The private originals are not modified.

The source provenance's `audit_sha256`, `result_sha256`, `requests_sha256`
and `episodes_sha256` refer to **original source-run files**, not renamed
or redacted files. Use the publication manifest for public-file hashes.

The supplied private log archive did **not** include raw
`action_requests.jsonl`, `episodes.jsonl`, per-step streams, or rollout
videos. The public archive therefore does not contain them either.
The source audit reports checking all 72000 request seeds/action hashes;
the public verifier checks recorded counts and consistency, but cannot
independently redo those per-request hash checks without the raw streams.
Private policy/server/supervisor logs and machine manifests are omitted.

The original 340000 checkpoint, this recorded **59.75%** source evaluation,
the manuscript's **58.3%** experiment, and the later bounded training-resume
validation are distinct artifacts. This archive does not collapse them into
one claimed reproduction.

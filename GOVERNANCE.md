# Governance

A contribution is a skill contract. This repository's runner is the referee. It does not hand out a shell.

```
pull request
  -> diff is only a skill manifest, a reward, or a policy zip
  -> Snyk Open Source passes
  -> one job for this GitHub user in the last 7 days
  -> LiteLLM probe answers as robo-arm-gpu-probe (no cloud fallback)
  -> nvidia-smi shows the card free of vLLM and at least 8 GiB free
  -> twin eval, 50 episodes
  -> certificate on the pull request
```

Merge when the certificate says `passed`. That means Snyk passed, the gateway receipt says `admitted`, and the success rate is at least 0.8 over 50 episodes.

## What a certificate looks like

```json
{
  "skill": "barrier",
  "gym": "barrier",
  "policy": "scripted",
  "success_rate": 1.0,
  "n_episodes": 50,
  "snyk": {"status": "passed", "project": "robo-arm-gym", "commit": "abc123"},
  "gateway": {
    "admitted": true,
    "reason": "ok",
    "model": "robo-arm-gpu-probe",
    "worker": "node-d",
    "request_id": "req-example",
    "gpu_free_mib": 12000
  },
  "passed": true
}
```

The virtual key, the Snyk token, and the gateway admin key are not in that file. `governance/redact.py` drops fields whose names contain `key`, `token`, `secret`, `password`, `authorization`, or `credential`.

## Gateway

Calls go to `https://ai-gateway.fimpo.net/v1` unless `AI_GATEWAY_BASE_URL` is set on the runner. The probe model is `robo-arm-gpu-probe`. That alias has no cloud fallback. A Gemini reply is a refusal, not a confirmation that the GPU is up.

The runner then reads `nvidia-smi`. A vLLM process, or less than 8 GiB free, refuses the job before `twin train` starts.

## Snyk

Every pull request runs `.github/workflows/snyk.yml`. A train request runs Snyk again on the pull-request commit and uploads the receipt. `SNYK_TOKEN` is a GitHub secret.

## Runner

The train job is `runs-on: [self-hosted, robo-arm]`. Install it on the GPU machine with `governance/install-runner.sh`. The admit step is the only step that sees `AI_GATEWAY_KEY`. Training runs with that variable unset.

A pull request that edits `twin/train.py` or `.github/workflows/` cannot start a GPU job in the same diff. The guard that checks the diff is the copy on the default branch, not the copy in the pull request.

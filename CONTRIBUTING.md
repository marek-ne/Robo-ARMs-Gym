# Contributing

## Local

```bash
pip install -e .
python -m unittest tests.test_governance tests.test_skills_registry
python -m twin eval --skill barrier --policy scripted --episodes 50
```

A Mac is enough for the MuJoCo twin. The certificate still comes from the runner.

## A pull request that can train

One skill per pull request. The diff may contain only:

| File | What the runner does |
| --- | --- |
| `skills/<name>/skill.yaml` | Scripted eval, 50 episodes |
| `skills/<name>/reward.py` | PPO, 50,000 steps, then 50 episodes |
| `policies/<name>/*.zip` | Eval that zip, no training |

`reward.py` defines `reward(obs, action, info, default) -> float`. The default is the env reward. The function is imported by the fixed trainer. It is not run as a script, and it does not see the gateway key.

Set `gym: reach`, `gym: place`, or `gym: barrier` when the skill name is not one of those three. Without `gym`, the skill name has to be a built-in gym.

Comment `/train` on the pull request, or add the label `train-on-node`. The author of the pull request, or a collaborator, can spend the job. One successful job per GitHub user each week.

## What will be refused

- Any other path in the diff
- A change to `twin/train.py` or `.github/workflows/` in the same pull request
- Snyk failing
- A gateway reply from a cloud model
- vLLM still holding the GPU, or under 8 GiB free
- Success rate under 0.8, or fewer than 50 episodes

## Manifest

`skills/schema.json` is the contract. `python -m skills.registry create my-job --from barrier --summary "..."` copies a template and clears the certificate. That command does not train.

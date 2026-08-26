# Remote compute

Two remote options are set up. Neither stores a credential in this file.

## Kaggle (preferred for memory-heavy work: ~30 GB RAM, GPU)

The API needs a **bearer token**, not the older `kaggle.json` username/key pair.
The key in `~/.kaggle/kaggle.json` returns 401 on `kernels/*` and
`competitions/*` regardless of account verification; only `datasets/*` works
with it. The bearer token works everywhere.

Ask the user for the token, then:

```python
headers = {"Authorization": f"Bearer {KAGGLE_API_TOKEN}"}
```

Pushing a kernel - the slug must be the **full `owner/name` path** in the
`slug` field. `id` expects an integer, and a bare name plus `ownerSlug` is
rejected as an invalid slug:

```python
payload = {
    "slug": "onethybeing/my-kernel",     # full path, not a bare name
    "newTitle": "my kernel",
    "language": "python", "kernelType": "script",
    "isPrivate": True, "enableGpu": True, "enableInternet": True,
    "text": "<source>",
    "datasetDataSources": [], "competitionDataSources": [],
    "kernelDataSources": [],
}
requests.post("https://www.kaggle.com/api/v1/kernels/push", json=payload, headers=headers)
```

Then poll `kernels/status` and fetch `kernels/output` (both take `userName` and
`kernelSlug` as query parameters). A push returns HTTP 200 even on failure -
check the `error` field in the response body rather than the status code.

## Colab CLI (WSL Ubuntu, authenticated)

```powershell
wsl -d Ubuntu -u root -- colab sessions
wsl -d Ubuntu -u root -- colab new --gpu T4      # also L4, A100, H100
echo "print('hi')" | wsl -d Ubuntu -u root -- colab exec --session <ID>
```

Gives a T4 with **12 GB RAM**, which is the binding constraint - building graph
features on HI-Medium (31.9M rows) was OOM-killed at 11.6 GB. Kaggle's larger
allocation is the reason to prefer it for that job.

`colab exec` prints a traceback to stderr on long commands but the real output
still arrives at the end, so read the tail rather than trusting the exit path.
Long jobs must be launched detached (`subprocess.Popen` with `nohup`) or they
exceed the exec timeout.

The CLI needs `jupyter-kernel-client<1.0.0`; it is pinned in
`/opt/google-colab-cli`. Do not upgrade it to 1.x.

# Machine constraints

Local: 16 GB total, typically **1-2 GB free** (Chrome and VS Code hold ~5 GB).
XGBoost at `max_depth=12` was killed three times locally; depth 8 plus
`QuantileDMatrix` fits. Check free memory before starting anything large.

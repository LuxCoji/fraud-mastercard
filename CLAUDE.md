# Colab CLI handoff

Google Colab CLI is installed and authenticated in Ubuntu WSL. A live CPU Colab session exists; list sessions with:

```powershell
wsl -d Ubuntu -u root -- colab sessions
```

Run remote Python from the project terminal with:

```powershell
echo "print('hello')" | wsl -d Ubuntu -u root -- colab exec --session <SESSION_ID>
```

Useful commands: `colab new`, `colab sessions`, `colab exec`, `colab status`, and `colab stop`.

Important: the CLI needs `jupyter-kernel-client<1.0.0`; it has already been pinned in `/opt/google-colab-cli`. Do not upgrade it to 1.x until the CLI fixes compatibility. No credentials or tokens are stored in this file.

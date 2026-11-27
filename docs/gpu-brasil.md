# GPU Brasil — managing GPUs and disks

Status: **operational guide**, API docs read 2026-11-25. Used only by the
[LLM lab](llm-lab.md). Reference: <https://gpubrasil.com.br/br/api.html>
(re-read it when a call behaves differently; prices change during the day).

## Rules for agents

1. **Read freely, ask before changing.** Any `POST`, `PATCH` or `DELETE` needs
   the user's approval for that specific action, naming the target and cost.
   One approval never covers the next action.
2. **Never delete a disk** unless the user names it and asks. It is permanent.
3. **Only touch lab resources.** The account holds others; leave them alone.
4. **Never print secrets**: API key, `disk_token`, `/etc/gpubr-disk.env`,
   `/gpubrasil-template.sh` (contains the disk URL). Select output fields by
   allowlist, never by pattern; never dump raw instance or disk JSON.
5. **This repo is public**: no ids, IPs, balances or account data in git.
6. **Only allowlisted lab data on a machine** ([llm-lab.md](llm-lab.md) §3). Case text
   goes up only as an export of the anonymized, scan-filtered `corpus.db` text, never
   `data/raw/` or `corpus.db` itself. GPUs run outside Brazil. No `.env`, HF or GitHub
   tokens either.
7. **Always save to `agavus-training`.** Every lab machine is created with
   `"diskId": "name:agavus-training"`; anything worth keeping goes to
   `/workspace/disco/agavus/`. If the disk is not attached or not `pronto`, do
   not train: stop and tell the user. Never use another disk instead.
8. **Finish with an inventory**: say what is still running and what it costs.

## Authentication

`GPUBRASIL_API_KEY` is set in the user's interactive fish shell only, so agents
call it through `fish -ic`. Never copy it into a file, argument or log. The key
can create and delete everything in the account, and the docs describe no
spending cap.

The header goes through stdin, so the key is not in the process list. This
lists instances:

```sh
fish -ic 'printf "Authorization: Bearer %s" $GPUBRASIL_API_KEY |
  curl -fsS -H @- "https://gpubrasil.com.br/api/instances"' |
python3 -I -c '
import json, sys
keep = ("instance_id", "name", "status", "gpu_model_label", "price_per_hour_brl",
        "daily_stopped_brl", "supports_pause", "disk_persists_on_stop", "deleting",
        "ssh_user", "ip_address", "ssh_port", "disk_id", "error_message")
for row in json.load(sys.stdin)["data"]:
  print({key: row.get(key) for key in keep})
'
```

## Calls

Base URL `https://gpubrasil.com.br`. `:ref` is an id or `name:<name>`; ids win.
Use the **id** for anything destructive (names can repeat: `409 AMBIGUOUS_NAME`).

| Need | Call | Approval |
| --- | --- | --- |
| GPU catalog, no key needed | `GET /api/gpus?model=&min_vram_gb=&available=true&sort=price` | no |
| Machines; one machine + SSH | `GET /api/instances`, `GET /api/instances/:ref` | no |
| Cost of one machine | `GET /api/instances/:ref/usage` | no |
| Disks, SSH keys | `GET /api/disks`, `GET /api/ssh-keys` | no |
| Balance, statement | `GET /api/user/balance`, `GET /api/transactions` | no |
| Create machine | `POST /api/instances/deploy` | **yes** |
| Rename or tag machine | `PATCH /api/instances/:ref` | **yes** |
| Hibernate, wake | `POST /api/instances/:ref/stop`, `/start` | **yes** |
| Delete machine | `DELETE /api/instances/:ref` | **yes** |
| Create, rename disk | `POST /api/disks`, `PATCH /api/disks/:ref` | **yes** |
| Delete disk | `DELETE /api/disks/:ref` | **yes, rule 2** |

`/api/user/balance` also returns the user's name and email: print only
`balance_brl`.

### Create

`gpuModel` is the catalog's `gpu_key`. The first hour is charged immediately
(refunded if provisioning fails).

```json
{
  "name": "agavus-lab",
  "gpuModel": "<gpu_key>",
  "vcpuCount": 8,
  "ramGb": 32,
  "storageGb": 100,
  "sshKeyName": "<registered key>",
  "diskId": "name:agavus-training",
  "metadata": {"project": "agavus-lab"},
  "requireUniqueName": true
}
```

Poll `GET /api/instances/:ref` every 10 s until `running` (give up after
15 min), then use `connection.sshCommand`. Before training,
confirm that `disk_id` matches `agavus-training` and that the sync check below
passes with a small test file. **IP and port can change on every
wake**, so always fetch them again. Never retry a deploy blindly: list first,
since each success is billed.

Before asking for approval, show: GPU and VRAM, R$/hour, disk, expected hours,
expected total and current balance.

## Delete, don't hibernate

| | Local files | Persistent disk | Cost while off |
| --- | --- | --- | --- |
| Hibernate | kept only if `disk_persists_on_stop` | kept | `daily_stopped_brl` per day + disks |
| **Delete** | erased | kept | disks only |

Check `disk_persists_on_stop` and `supports_pause` first. When the former is
`false`, hibernating loses local files **and** still costs money: prefer
deleting. The trade-off is that the GPU may be unavailable later.

- `202` or `status: deleting` means accepted: do not repeat the `DELETE`.
- `409` while creating: retry after a few minutes.
- If the balance runs out, the machine **and its local disk** are deleted.
- Disks bill by contracted size, even when empty or unattached.

Before deleting, in order: (1) training stopped; (2) checkpoints moved to the
disk; (3) sync confirmed (below); (4) important checkpoints downloaded with
matching SHA-256; (5) user approves; (6) delete; (7) list again to confirm.

## The persistent disk

It is **not a mount**. `/workspace/disco` is an ordinary folder copied to cloud
storage (observed in the startup script, 2026-11-25):

- At startup everything is downloaded; wait until `/workspace/disco.estado`
  says `pronto` (`erro` means stop and investigate).
- Changes upload about every 60 s. Stopping right after a save can lose it.
- Deletions are not propagated: removed files can come back.
- If the quota is full, uploads pause with only a log entry.
- A file can be uploaded half-written. Write elsewhere, then move it in.

To confirm a sync, wait at least two minutes after the last write (longer for
large files), then run on the machine:

```sh
cat /workspace/disco.estado
tail -n 5 /var/log/gpubr-disk.log
find /workspace/disco -type f -newer /workspace/disco.enviado | head
```

Expect `pronto`, no errors and no files listed. This is evidence, not proof;
the checkpoint download is the backup.

| Path | Survives | Holds |
| --- | --- | --- |
| `/workspace/disco/agavus/` | yes | `setup.sh`, `pyproject.toml`, `uv.lock`, finished `runs/` |
| `/workspace/local/` | no | Python, venv, cache, code, active training |

Do not put the virtual environment on the disk: thousands of small files,
symlinks and permissions sync badly. Rebuild it from `setup.sh` and `uv.lock`.

## Connecting

- Use a dedicated SSH key with `-o IdentitiesOnly=yes -o BatchMode=yes`.
- Host keys change when a machine is recreated. Verify the fingerprint through
  the web terminal when possible; otherwise say it was accepted unverified.
- Check limits with `/sys/fs/cgroup/cpu.max` and `memory.max`; `nproc` and
  `free` show the host.
- Blackwell GPUs (`sm_120`) need PyTorch ≥ 2.7 with CUDA ≥ 12.8. The image may
  ship an older one where `cuda.is_available()` is true but training fails:
  test a real matrix product.

## Incidents

Leaked API key: the user revokes it in the panel (API Keys), then checks
machines, disks and the statement. Leaked `disk_token`: tell the user and
contact the provider; no rotation call is documented.

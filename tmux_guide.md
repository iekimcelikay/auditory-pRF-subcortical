# tmux / Remote Session Guide (brodmann → hyperion)

## Daily workflow

From your laptop:

1. Open VS Code → Remote Explorer → connect to `brodmann108`
2. Open integrated terminal — you're on brodmann
3. `tmux attach -t main` — the vscode-tunnel runs under systemd now, so there's no separate tunnel window to check
4. `ssh ecelikay@hyperion-login-01`
5. `tmux attach -t main` — you're back in your Hyperion session

Steps 3 and 5 just reconnect you to where you left off.

## If something died

| Problem | Fix |
|---|---|
| No `main` session found on brodmann (fresh boot, no tmux yet) | `tmux new -s main` — the tunnel is already up independently of tmux |
| Hyperion session died | `tmux new -s main` and carry on |
| Want to confirm the tunnel is actually alive | `~/.local/bin/vscode-tunnel tunnel status` on brodmann |
| `tmux attach -t main` says "sessio ns should be nested with care" | You're already inside a tmux session (check `echo $TMUX`). Use `tmux switch-client -t main` instead of `attach`, or `unset TMUX` first if that's stale |
| Need to see why the tunnel isn't behaving | `~/.local/bin/vscode-tunnel tunnel service log` (journalctl usually isn't readable without `systemd-journal` group membership on shared servers) |

## Why the tunnel survives reboots now

The vscode-tunnel process used to live inside a tmux window, which meant it died with brodmann and had to be manually relaunched after every reboot. It's now installed as a proper systemd service via the CLI's **own built-in installer** — do not hand-write a unit file for this, it's a trap (see gotcha below).

- **Linger** is enabled (`loginctl enable-linger ecelikay`), so the user systemd manager stays alive with zero active login sessions.
- The service (`code-tunnel.service`) was created by the tunnel CLI itself and auto-starts on boot, auto-restarts on failure.

Check it's alive:
```bash
systemctl --user status code-tunnel.service
~/.local/bin/vscode-tunnel tunnel status      # {"tunnel":{"tunnel":"Connected", ...}}
```

Restart / reinstall if ever needed:
```bash
~/.local/bin/vscode-tunnel tunnel service uninstall
~/.local/bin/vscode-tunnel tunnel service install
```

### Gotcha: two D-Bus buses, don't hand-roll the unit file

Interactive SSH shells on brodmann connect to an **ad-hoc D-Bus session bus** (`echo $DBUS_SESSION_BUS_ADDRESS`, something like `unix:abstract=/tmp/dbus-XXXXXXXX`), separate from the **standard systemd user bus** (`unix:path=/run/user/<uid>/bus`, what `systemctl --user show-environment` reports). Auth tokens stored via keyring during an interactive login live on the ad-hoc bus; a hand-written `systemd --user` unit only ever sees the standard bus and can never find them — it'll silently sit "running" but perpetually unauthenticated ("offline" in VS Code's UI), asking for a fresh device-code login every time it restarts.

The fix is to let the CLI's own installer do the work, run explicitly against the standard bus:
```bash
DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$(id -u)/bus ~/.local/bin/vscode-tunnel tunnel service install
```
It'll prompt a fresh device-code login (expected — the standard bus has no reachable keyring either), but *because it's the official installer*, it registers the credential in a way the service can actually use afterwards, rather than us reverse-engineering keyring/D-Bus plumbing ourselves.

## General principle

tmux protects against **disconnects** (closed laptop, dropped wifi, closed terminal) — the tmux server keeps running as a daemon independent of any attached client. It does **not** protect against the **host itself** going down (reboot, crash, admin action) — everything tracked by that server dies with it.

Anything you find yourself manually re-launching after every reconnect (like the tunnel used to be) is a candidate for a `systemd --user` service rather than a tmux window. Reserve tmux for actual interactive work — sessions you expect to rebuild occasionally are fine to lose; background infrastructure you depend on shouldn't be tied to whether you happen to be logged in.

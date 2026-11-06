# Raspberry Pi setup record

Date: 2026-10-31, updated 2026-11-06 (Asia/Kolkata). Status: **Pi reachable over SSH with key login; thermal kit synced, systemd install pending; no Llama benchmark yet.** The 2026-11-06 section below supersedes the Phase 0 notes and known issues.

## 2026-11-06: Pi online

- **Boot medium:** SanDisk Cruzer Blade 32 GB USB 2.0 stick (no SD card), written with Raspberry Pi Imager: Raspberry Pi OS Lite 64-bit, Debian 13 trixie, image 2026-10-06. Cloud-init settings: hostname `pi5`, user `hermoine`, SSH on, Wi-Fi `HomeJio`, timezone Asia/Kolkata.
- **Why the first boot failed:** the Pi was powered from a laptop USB port, not a 5 A supply. On a Pi 5 that blocks USB boot, so the stick's root partition was never expanded. Fix: `usb_max_current_enable=1` appended under `[all]` in `config.txt` (original kept in the session scratchpad). After that it booted and joined Wi-Fi.
- **Access:** `ssh -i ~/.ssh/pi5 hermoine@pi5.local` (192.168.29.39, MAC 2c:cf:67:bb:7d:12). Dedicated laptop key `~/.ssh/pi5`. Password login is still enabled. `sudo` needs the user's password (Imager wrote `sudo: null`; `hermoine` is in the `sudo` group).
- **Inventory:** Pi 5 Model B Rev 1.1, 4 GB (3.8 GB available), 2 GB swap, 28 GB root (25 GB free), kernel 6.18.50+rpt-rpi-2712, Python 3.13.5, internet OK. CPU flags include `asimddp`; **no `i8mm`**.
- **Idle readings:** 44–48 °C, `throttled=0x0`, EXT5V 5.15–5.18 V, about 2.0 W estimated from the PMIC rails, 2.4 GHz on all cores, governor `ondemand`, frequency steps 1.5–2.4 GHz in 100 MHz steps.
- **Cooling:** none fitted. Firmware trip points: 50/60/67.5/75 °C active, 110 °C critical; the firmware throttles around 80–85 °C.
- **Thermal kit (`deploy/pi/`, README there):** `rakshak_thermald.py` steps `scaling_max_freq` down 100 MHz at ≥ 68 °C (3-sample mean), drops to 1.5 GHz at ≥ 73 °C, and steps back up after 10 s at ≤ 60 °C, with at least 6 s between steps. It restores full speed on exit. `thermal_run.py` wraps a job under `nice`/`ionice`: pauses at 75 °C, resumes at 65 °C, kills at 80 °C, and logs CSV plus stdout. `pi_dashboard.py` serves a live dashboard on 127.0.0.1:8765 (SSH tunnel). Verified live on the Pi without sudo: dashboard API/UI/SSE, governor `--dry-run`, governor decisions on a fake sysfs tree, `thermal_run` smoke job. **Pending:** `cd ~/rakshak-pi && sudo ./install.sh` (enables both systemd services and disables Bluetooth unless `--keep-bt`). `sudo ./uninstall.sh` reverses it.
- The thermal kit caps frequency through sysfs only (resets on reboot). There are no `config.txt` frequency or voltage changes, which keeps within the limits above.

## Target and operating limits

User-specified target (not yet verified on device): Raspberry Pi 5, 4 GB RAM, Cortex-A76, Raspberry Pi OS 64-bit, connected directly to the laptop by Ethernet. Cooler, PSU, storage, OS release and kernel remain unverified.

- Before and after every heavy step: `vcgencmd measure_temp`, `vcgencmd get_throttled`, and `free -m` on the Pi. Pause heavy work at 75 °C or any throttling/under-voltage flags; never continue past 80 °C. Ask the user to check the cooler/PSU when triggered.
- Run heavy work with `nice -n 10 ionice -c3`, one job at a time. Build with at most `-j2`, only after 2 GB zram or fallback swap is installed. Abort a build below 300 MB available RAM.
- No stress tests, overclocking, voltage/frequency changes, or disabling thermal protection. No tight polling; a one-second sampler is allowed only during a benchmark.
- Ask for credentials; never guess them. Ask before laptop sudo/network changes, creating a new SSH key, or changing an already configured hostname/timezone/locale. Keep SSH password access until key login is verified.
- No live capture or attack traffic in this setup. No paid services, Modal, external posts, or access to other devices.

These user-specified limits supersede the older, higher thermal limits in the project handoff.

## Repository review

Read `README.md`, `docs/HANDOFF.md`, `docs/CONTEXT.md`, `docs/EXECUTORCH_PLAN.md`, `docs/NETWORK_IDS_PLAN.md`, all Python sources in `src/rakshak/nids/`, and `src/rakshak/llama_source.py` before discovery.

- `deploy/pi/` does not exist, so there is no existing `setup.sh` or `build_executorch.sh` to review or execute.
- The offline NIDS command requires `--model` as well as `--pcap`. The ONNX scorer validates schema version, live feature order, and family order. The live subset contains 19 of the shared schema's 33 features.
- Requested Llama runtime: native ExecuTorch v1.5.1 `llama_main`, XNNPACK, with upstream release documentation to be checked before building. No upstream build instructions have yet been verified in this setup session.
- Pinned model from `src/rakshak/llama_source.py`: `executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET`, revision `911221e0f49d3354cbd4697a9c09feee25cadec7`, file `Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte`. Project docs list `tokenizer.model` in that snapshot; verify its listing before downloading (the Python `PREBUILT_ALLOW` constant does not include it).

## Phase 0: laptop discovery

Exact network commands executed, initially in the sandbox and then repeated with approved access outside the sandbox:

```bash
ip -br addr
ip neigh
timeout 10s avahi-resolve -n raspberrypi.local
```

Initial sandbox results: both `ip` commands failed with `Cannot open netlink socket: Operation not permitted`; Avahi failed with `Failed to create client object: Access denied`. The repeated commands used no sudo and changed no network settings.

Observed outside the sandbox:

```text
lo       UNKNOWN  127.0.0.1/8 ::1/128
wlp2s0   UP       192.168.29.18/24 [global IPv6 and link-local IPv6 present]
```

`ip neigh` listed only the Wi-Fi gateway (`192.168.29.1`) and its IPv6 router entry on `wlp2s0`. No wired interface, wired subnet, or Pi neighbour was listed. Avahi returned `Failed to resolve host name 'raspberrypi.local': Timeout reached`.

No subnet scan was run: the directly attached link's interface/subnet could not be established. The Wi-Fi subnet was not scanned. No SSH attempt was made, and no username or credential was guessed.

Next required input: confirm whether the Ethernet cable is connected through a USB Ethernet adapter/dock and that the Pi is powered; provide the Pi Imager username and custom hostname if one was set. Pi internet access cannot be checked until the Pi is identified and accessible. Do not infer internet availability from the laptop's Wi-Fi connection.

## Remaining phases and measurements

| Item | Status |
|---|---|
| Phase 1: SSH key login, SSH hardening, UFW, unattended updates | Not started; existing security settings unknown |
| Phase 2: upgrades/packages, identity settings, 2 GB zram/swap | Not started |
| Inventory: OS/kernel/CPU/RAM/disk | Not measured on Pi |
| Idle temperature and throttled flags | Not measured |
| Short compile: load temperature below 70 °C, cooler check | Not run |
| Phase 3: rsync, uv environment, ONNX Runtime/nfstream, pytest counts | Not started |
| Phase 4: verified native llama_main build steps | None yet; no build attempted |
| Llama single short run: TTFT, tokens/s, peak RSS, max temperature | Not run; no benchmark numbers claimed |
| Phase 5: supplied ONNX models, 200 inputs at 1/4 threads, p50/p95 | Awaiting model paths and Pi access |
| Offline benign PCAP smoke test | Not run |

For the future Llama benchmark, use at most 64 new tokens and `/usr/bin/time -v`; stop and report if peak RSS exceeds 3.2 GB or temperature reaches 75 °C. Report that measurement as a single short run, not a latency distribution.

## Known issues

1. No Ethernet interface is visible to the laptop discovery commands. Physical connection/adapter state needs clarification before further discovery.
2. The Pi username is unknown; the default mDNS hostname did not resolve.
3. Pi internet, cooling, PSU health, memory, disk space and existing configuration have not been checked.
4. Deployment scripts are absent. Future setup/build commands need review against actual Pi state and the pinned upstream release before execution.

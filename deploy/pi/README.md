# Rakshak Pi 5 thermal kit

Passive-cooled Raspberry Pi 5 (no heatsink, USB-C supply). Three pieces:

| File | Runs as | What it does |
|---|---|---|
| `rakshak_thermald.py` | root (systemd) | Caps `scaling_max_freq` on every cpufreq policy from the temperature. |
| `thermal_run.py` | you (no root) | Runs one job under pause/resume/kill guard rails. |
| `pi_dashboard.py` | you (systemd) | Live dashboard on `127.0.0.1:8765`, reached over an SSH tunnel. |

## Thresholds

thermald works from the mean of the last 3 samples (2 s apart):

- mean >= 68 C: cap down 100 MHz
- mean >= 73 C: cap straight to 1500 MHz
- mean <= 60 C for 10 s: cap up 100 MHz, up to the ceiling (2400 MHz)
- at least 6 s between steps; a hard drop is never delayed

Job guard (`thermal_run.py`): pause (SIGSTOP) at 75 C, resume (SIGCONT) at 65 C,
kill at 80 C (SIGTERM, SIGKILL after 5 s, exit 3). Jobs run under `nice -n 10 ionice -c3`.

```
python3 ~/rakshak-pi/thermal_run.py --name kd-smoke -- python3 train.py --epochs 1
```

Outputs go to `~/rakshak-pi/run/`: `current_job.json`, `<name>-<ts>.csv`, `<name>-<ts>.log`.

## Install (on the Pi, once)

From the laptop:

```
deploy/pi/sync.sh
```

On the Pi:

```
cd ~/rakshak-pi && sudo ./install.sh          # add --keep-bt to leave bluetooth on
```

Remove with `sudo ./uninstall.sh` (restores the max frequency).

thermald state: `/run/rakshak/thermal.json`. Transitions: `/run/rakshak/thermal_events.log`.

## Dashboard

```
ssh -i ~/.ssh/pi5 -N -L 8765:127.0.0.1:8765 hermoine@pi5.local
```

Then open http://localhost:8765.

## Passive cooling tips

- Use a 27 W USB-C PD supply. A laptop charger on a weak cable causes under-voltage throttling. Check the Throttle tile.
- Stand the board on its edge or on a metal surface so the SoC area has airflow.
- Keep the case vented or open. Do not stack it under other hardware.
- A small 5 V heatsink or fan hat is the real fix. Passive operation with the governor just trades speed for staying cool.
- Keep heavy jobs on Kaggle or the laptop. The Pi is for light inference and monitoring.

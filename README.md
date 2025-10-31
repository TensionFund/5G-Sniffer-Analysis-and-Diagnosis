# 5G PRACH Detection & Sniffer Analysis
**Author:** Zain (sxh1655)  
**Repository:** University of Birmingham — Final Year MSc Project (Dissertation)

## Project overview
This project implements a diagnostic analysis pipeline for 5G PRACH (Physical Random Access Channel) detection. It integrates:
- an srsRAN **gNB** configured to log PHY receive symbols,
- a **sniffer** (NR-Scope / custom sniffer) that captures and demodulates OFDM symbols (with per-worker dumps),
- MATLAB analysis scripts that compare gNB logs and sniffer captures to investigate missed PRACH detections, and
- patches to `prach_worker.cc` to export demodulated IQ samples in the same `.fc32` float32 interleaved format used by the gNB.

This repo contains code, configuration, scripts, analysis notebooks and example logs to reproduce the experiments and the detection comparisons used in the dissertation.

---

## Key features / goals
- Produce per-thread sniffer dumps of demodulated OFDM symbols (interleaved float32 `.fc32`) so sample-level comparisons can be made with gNB logs.
- Enable offline replay of captures (file backend) for deterministic detector debugging.
- Provide MATLAB tools to align SFN/slot/timestamp, resample/align sampling rates, compute correlation/detection metrics and visualise missed detections.
- Provide reproducible configuration and instructions to run the experimental setup (gNB, sniffer, phone in Faraday case, or recorded files).

---

## Quick links
- `gnb(Base Station)/` — gNB configuration files, example YAMLs and notes.
- `sniffer/` — sniffer source, `prach_worker.cc` modifications, example YAMLs and build.
- `matlab/` — MATLAB scripts and analysis notebooks.
- `Presentation` — Project PowerPoint. thesis chapters, diagrams, experiment logs.

---

## Prerequisites (tested on Ubuntu)
- Ubuntu 20.04 / 22.04 (or similar Linux)
- srsRAN (gNB) — build from srsRAN Project. Run gNB binary with YAML config. :contentReference[oaicite:2]{index=2}
- NR-Scope / custom sniffer build environment (C++17 / CMake)
- MATLAB (R2022a or later recommended) — Signal Processing toolbox for resampling and visualization
- UHD drivers (if using USRP hardware / B210)
- Python 3.8+ for helper scripts (numpy, scipy, matplotlib)
- `git` and SSH keys or a personal access token if repo is private

---

## Notable config & file paths (project defaults)
> These reflect the experiment setup used in the dissertation and in the analysis scripts.

- **gNB YAML (example used in tests):**  
  `/home/uob/work/basestation_conf/5G_config/srsgNB_pixel6/gnb_usrp_n78_00101.yaml`

- **gNB PHY symbol logs (float32 interleaved IQ):**  
  `/home/uob/work/log/phy_rx_symbols` -> `phy_rx_symbols_*.fc32`  
  Config entries used:
  ```yaml
  log.phy_rx_symbols_filename: /home/uob/work/log/phy_rx_symbols
  log.phy_rx_symbols_port: 0
  log.phy_rx_symbols_prach: false

# YarrTdarr

**100% Free, Open-Source (GPLv3) Multi-GPU Media Optimizer** — zero staging limits, zero worker caps, zero paywalls, zero telemetry.

Built in Forgeverse by absorbing the core architecture of [Unmanic](https://github.com/Unmanic/unmanic) (GPLv3):
- **File Stability & Smart Skip:** Absorbed from Unmanic's `filetest` & `eventmonitor` — waits for Sonarr/Radarr/qBittorrent copies to settle before touching them, and skips files already in target codec/audio/container.
- **Multi-GPU NVENC Pool:** Schedules across all available NVIDIA GPUs (e.g., 4x Tesla P100) using `hevc_nvenc` (8-bit, no B-frames for Pascal) or `h264_nvenc`.
- **Plex Direct-Play Audio Optimizer:** Preserves surround tracks while ensuring an `aac` stereo compatibility track exists so Plex never transcodes audio on the fly.
- **Atomic Post-Processor:** Transcodes to fast local temp storage, runs `ffprobe` duration & stream integrity verification, and atomically replaces the source file (auto-accept enabled by default, unlimited).

# config/

This folder contains configuration templates.

- `config.example.yaml` is the tracked template.
- `config.yaml` is your local runtime configuration and is gitignored.

Setup:

```bash
cp config/config.example.yaml config/config.yaml
```

All scripts read `config/config.yaml` by default.
You can override with `--config` or `DCVC_CONFIG`.

# LitniceLM comparison

This directory is a standalone comparison project. During the skeleton phase,
the runner only prints its execution plan and performs no benchmark work.

Run plan mode from the `tools/` directory:

```bash
python3 -m vs_bench
```

The comparison is configured by the `SPEC` instance at the top of `run.py`.
There is intentionally no command-line interface.


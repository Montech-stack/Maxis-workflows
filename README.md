# Maxis Workflows

Standalone GitHub Actions render worker for the [Maxis](https://maxis.media) film pipeline.

Runs `stitch_render.py` (ffmpeg download/normalize/concat/QC) on a GitHub-hosted
runner instead of the main Browsight VPS, which only has a single vCPU and was
starving on this workload while also running the browser automation that drives
video generation. This repo is public so the workflow gets unlimited free
Actions minutes — it contains no proprietary Maxis source, only this one
standalone script.

Triggered via `repository_dispatch` (event type `stitch-render-job`) from the
main [Maxis-media](https://github.com/Montech-stack/Maxis-media) repo's
`lib/github-dispatch.ts`.

## Required repository secrets

- `SUPABASE_URL` — Supabase project URL
- `SUPABASE_SERVICE_ROLE_KEY` — Supabase service role key
- `BROWSIGHT_URL` — Browsight VPS public URL (`https://browsight.maxis.media`)
- `BROWSIGHT_API_KEY` — Browsight's API key, for uploading the finished video back

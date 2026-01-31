# Repository Guidelines

## Project Structure & Module Organization
- `stabilize_phase.py` is the main pipeline for alignment, rotation correction, and day-level refinement.
- `create_video.py` builds MP4 outputs from aligned frames.
- `util/` contains tooling (manual GUI alignment, outlier review, refinement utilities, small test scripts).
- `dep/` contains experimental benchmarks and algorithm comparison scripts.
- `input/` holds raw image sets, `output/` stores aligned results and logs, `temp/` stores intermediates.
- `requirements.txt` lists Python runtime dependencies.

## Build, Test, and Development Commands
- Install deps: `pip install -r requirements.txt` (Python 3.8+).
- Run alignment with defaults: `python stabilize_phase.py` (reads `input/`, writes `output/`).
- Align and build video: `python stabilize_phase.py --video --fps 30 --crf 23`.
- Build a video from an existing folder: `python create_video.py -i output -o timelapse.mp4`.
- Manual inspection: `python util/manual_align_gui.py --ref ref.jpg --mov mov.jpg`.
- FFmpeg must be installed and available on PATH for video generation.

## Coding Style & Naming Conventions
- Follow the existing style: 4-space indentation, PEP8-like layout.
- Use `snake_case` for functions/variables and `ALL_CAPS` for module constants.
- Prefer explicit CLI flags (e.g., `--input`, `--output`, `--fps`) and keep defaults in one place.
- Date-based folder naming uses `YYYY-MM-DD`; log files are `output/logs/[timestamp]_*.txt`.

## Testing Guidelines
- There is no formal test runner; use targeted scripts for checks and benchmarks.
- Examples: `python util/test_outlier_align.py`, `python util/test_orb_match.py`, `python dep/test_ecc_match.py`.
- Benchmark suite: `python dep/benchmark_v2.py` (requires representative input images).
- Place local datasets in `input/` and review outputs in `output/logs/`.

## Commit & Pull Request Guidelines
- Commit messages follow Conventional Commits (e.g., `feat: add day refinement`, `docs: update README`).
- Keep the subject line short and imperative; add a body if rationale is non-obvious.
- PRs should include a brief summary, commands run, and sample outputs (frame screenshot or MP4 path).
- Do not commit generated artifacts in `input/`, `output/`, `temp/`, or MP4s (gitignored).

## Security & Configuration Tips
- Keep large datasets local; share via external storage if needed.
- Validate FFmpeg availability before running video pipelines.

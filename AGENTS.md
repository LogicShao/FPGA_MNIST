# Repository Guidelines

## Project Structure & Module Organization
- `model_tools/` holds Python utilities for MNIST data transfer and UART communication.
- `hardware/` contains FPGA/Quartus projects and generated artifacts; treat `hardware/quartus_proj/` as generated output unless explicitly versioned.
- Top-level docs and configs (e.g., `.gitignore`) live in the repo root.

## Build, Test, and Development Commands
- `python model_tools/send_image.py` runs the interactive UART sender for MNIST or local `.npy`/`.bin` inputs.
- Quartus builds are run from the GUI or your local Quartus toolchain; this repo does not include a scripted build command.

## Coding Style & Naming Conventions
- Python: 4-space indentation, ASCII by default, and clear function names such as `send_via_uart` or `load_npy_image`.
- Verilog: keep module names descriptive and consistent with file names (e.g., `seg_dynamic.v`).
- Paths: prefer relative paths inside the repo (e.g., `hardware/quartus_proj/`).

## Testing Guidelines
- No automated test framework is currently configured.
- If you add tests, place them near the related module and document how to run them in this file.

## Commit & Pull Request Guidelines
- Use short, imperative commit subjects (e.g., "Add UART menu").
- Include a brief body when changes are non-trivial.
- PRs should describe the hardware/software impact and include steps to verify (screenshots or logs if UI/FPGA behavior changes).

## Security & Configuration Tips
- Do not commit generated Quartus outputs unless required for reproducibility.
- Verify `SERIAL_PORT` and `BAUD_RATE` in `model_tools/send_image.py` before running UART transfers.

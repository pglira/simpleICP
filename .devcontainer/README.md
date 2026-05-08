# Unified devcontainer

A single devcontainer for all simpleICP language implementations: C++, Python, Octave, Julia, Rust, and MATLAB.

## What's installed

- **C++**: `g++`, CMake, Ninja, `clangd`, vcpkg with `eigen3`, `cxxopts`, `nanoflann`. `VCPKG_ROOT` is set. The `postCreateCommand` runs CMake configure with `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` so the clangd extension picks up `c++/build/compile_commands.json` automatically.
- **Python 3**: `numpy`, `scipy`, `pandas`, `lmfit`, `matplotlib`, `pytest`. The `simpleicp` package is installed in editable mode from `python/` on container create.
- **Octave**: `octave` plus `octave-statistics`.
- **Julia**: `MultivariateStats`, `NearestNeighbors`, `StatsBase` precompiled in a shared depot at `/opt/julia-depot`.
- **Rust**: `rustup` stable toolchain, system-wide.
- **MATLAB**: installed via `mpm` together with the Statistics and Machine Learning Toolbox.

## MATLAB licensing — first-time activation

MATLAB uses online license activation tied to your MathWorks Account. The first launch in the container needs an X11 display because the activation dialog is GUI.

The devcontainer is already wired up for this:

- `DISPLAY=${localEnv:DISPLAY}` is forwarded from the host.
- `/tmp/.X11-unix` is mounted into the container.
- Your `~/.matlab` directory is bind-mounted to `/home/vscode/.matlab`, so the activation persists across container rebuilds.

Steps:

1. On the host (Linux), make sure X11 is running and allow local connections if needed:
   ```bash
   xhost +local:
   ```
2. Open the workspace in the container.
3. Run `matlab` once. The activation dialog appears — sign in with your MathWorks Account, pick your license, and confirm.
4. After that, `matlab -batch "<command>"` works headlessly in the same and future containers (as long as `~/.matlab` stays mounted).

If you're on macOS/Windows, you need an X11 server (XQuartz, VcXsrv) and likely have to set `DISPLAY` manually.

## Notes

- The image is `linux/amd64` only. On Apple Silicon hosts, run with `--platform=linux/amd64`.
- Build is large (~12 GB); MATLAB dominates.
- To pin a different MATLAB release, edit `MATLAB_RELEASE` in the `Dockerfile`.

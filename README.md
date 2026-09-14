# Fractal Terrain Generator :mountain:

A real-time fractal terrain viewer built with Vulkan, for the "Fractals VU" course assignment
(Diamond-Square terrain generation). Terrain is generated on the CPU using the Diamond-Square
algorithm driven by a deterministic coordinate-hash PRNG (not a stateful RNG), streamed in as
tileable chunks around the camera, and rendered with a directional light, a transparent water
plane, and a GUI for tweaking generation parameters live.

Built on top of [Vulkan Launchpad](https://github.com/cg-tuwien/VulkanLaunchpad), a thin C++/Vulkan
teaching framework, plus [Dear ImGui](https://github.com/ocornut/imgui) for the control panel.

## Features

- Diamond-Square heightfield generation with a GUI-adjustable Hurst exponent, height scale, and
  reseed button
- Smooth ~2-second blend transition when the Hurst exponent or seed changes, instead of an instant pop
- Infinite, chunked terrain streaming around the camera (chunks generate/unload as you move, with
  seamlessly matching edges between neighbors)
- Trackball camera (orbit/pan/zoom) and a WASD + mouselook fly camera, toggleable at runtime
- A transparent, GUI-adjustable water plane
- Resizable window

## Controls

| Input | Action |
|---|---|
| `C` | Toggle between trackball and fly camera |
| Left-drag | Orbit the camera (trackball mode) |
| Right-drag | Pan the camera (trackball mode) |
| Scroll | Zoom (trackball mode) |
| Mouse | Look around (fly mode) |
| `W`/`A`/`S`/`D` | Move (fly mode) |
| `Space` / `Left Ctrl` | Move up / down (fly mode) |
| `Left Shift` | Move faster (fly mode) |
| `R` | Reseed the terrain |
| `F1` | Toggle wireframe |
| `F2` | Cycle backface-culling mode |
| `N` | Toggle normals debug view |
| `Esc` | Quit |

The Hurst exponent, height scale, water level, camera speed, view radius, and reseed button are all
adjustable live from the "Terrain Settings" GUI panel.

## Building

### Prerequisites (all platforms)

- [Git](https://git-scm.com/)
- [CMake](https://cmake.org/) 3.14 or newer
- A C++17 compiler
- A recent [Vulkan SDK](https://vulkan.lunarg.com/sdk/home)

Clone the repository with submodules (Vulkan Launchpad, Dear ImGui, and their own dependencies are
pulled in this way):

```bash
git clone --recurse-submodules https://github.com/kimsis/FractalTerrainGenerator.git
cd FractalTerrainGenerator
```

If you already cloned without `--recurse-submodules`, run `git submodule update --init --recursive`
from the repo root instead. Local fixes to the Vulkan Launchpad submodule (see `patches/`) are
applied automatically on every CMake configure, on all three platforms — no manual step needed.

### Windows

- Install [Git for Windows](https://git-scm.com/download/win) (select "Git from the command line
  and also from 3rd-party software" so `git` is on your `PATH`).
- Install a recent [Vulkan SDK for Windows](https://vulkan.lunarg.com/sdk/home#windows).
- Install [Visual Studio 2022 Community](https://visualstudio.microsoft.com/vs/community/) (or the
  [Build Tools for Visual Studio 2022](https://visualstudio.microsoft.com/downloads/?q=build+tools)
  if you don't need the full IDE), selecting the **Desktop development with C++** workload.
- Install [CMake](https://cmake.org/download/) (or use the one bundled with Visual Studio), version
  3.14+, with "Add CMake to the system PATH" enabled if installing standalone.

Then either:
- **Open the folder in Visual Studio**: `File -> Open -> Folder...`, select the repo root. Visual
  Studio will run CMake automatically; once it finishes, `Build -> Build All` (`Ctrl+Shift+B`), then
  select `VulkanLaunchpadStarter.exe (in workspaceRoot)` as the startup item to run/debug it.
- **Or from a Developer Command Prompt / Developer PowerShell**:
  ```powershell
  cmake -S . -B build
  cmake --build build --config Release
  ```
  The executable is written under `build\Release\` (or `build\Debug\`). Run it from the repository
  root, since it loads shaders/settings via paths relative to the executable and its parent
  directories (e.g. `assets/shaders/...`).

### macOS

- Install Xcode Command Line Tools: `xcode-select --install` (this also installs `git`).
- Install a recent [Vulkan SDK for macOS](https://vulkan.lunarg.com/sdk/home#mac), making sure to
  tick **System Global Installation** during setup so CMake can find it.
- Install [CMake](https://cmake.org/download/) 3.14+, e.g. via
  [Homebrew](https://formulae.brew.sh/formula/cmake): `brew install cmake`.

Then, from the repo root:

```bash
cmake -S . -B build -G Ninja   # or omit -G Ninja to use Xcode/Makefiles
cmake --build build
./build/VulkanLaunchpadStarter
```

(Run it from the repo root — asset paths are resolved relative to it.)

### Linux

Requirements: a C++ compiler, Git, CMake, a Vulkan SDK/loader + validation layers, and an X11 or
Wayland development environment (via GLFW).

```bash
# Debian/Ubuntu
sudo apt install git cmake build-essential xorg-dev libvulkan-dev vulkan-headers vulkan-validationlayers

# Fedora
sudo dnf install cmake gcc-c++ libXinerama-devel vulkan-loader-devel vulkan-headers vulkan-validation-layers-devel
sudo dnf -y groupinstall "X Software Development"

# Arch/Manjaro
sudo pacman -Sy cmake base-devel vulkan-validation-layers
```

Then, from the repo root:

```bash
cmake -S . -B build -G Ninja
cmake --build build
./build/VulkanLaunchpadStarter
```

## Troubleshooting

**Submodule checkout is slow or stuck.** Clone the repo manually instead of relying on CMake's
auto-update:
```bash
git clone --recurse-submodules https://github.com/kimsis/FractalTerrainGenerator.git
```

**macOS: CMake cannot find a C/CXX compiler.** If you previously had another Xcode Command Line
Tools installation, try `xcode-select --reset`.

**macOS: CMake cannot find Vulkan.** This usually means the Vulkan SDK wasn't installed with
**System Global Installation** ticked. Re-run the SDK's `MaintenanceTool.app` and add that
component.

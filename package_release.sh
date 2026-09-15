#!/usr/bin/env bash
# Packages a distributable .zip containing everything the app actually needs at runtime — the
# Release executable, the 4 shader files, the 3 settings .ini files it loads, and imgui.ini — into
# the repo root. See docs/PLAN.md or ask Claude for the full breakdown of why exactly these files
# and not the rest of assets/ (mostly dead leftovers from the original demo scene).
#
# Requires a bash environment (e.g. Git Bash/MSYS2/WSL on Windows) and a `zip` binary. Git for
# Windows' bundled bash does NOT include `zip` by default — install it separately (e.g. via MSYS2's
# `pacman -S zip`) if the check below fails.
set -euo pipefail

for tool in cmake zip unzip; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "Error: '$tool' not found on PATH. Install it before running this script." >&2
        exit 1
    fi
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$REPO_ROOT/build-release"
EXECUTABLE_NAME="VulkanLaunchpadStarter"

case "$(uname -s)" in
    Linux*) PLATFORM="Linux" ;;
    Darwin*) PLATFORM="macOS" ;;
    MINGW*|MSYS*|CYGWIN*) PLATFORM="Windows" ;;
    *) PLATFORM="$(uname -s)" ;;
esac

PACKAGE_NAME="FractalTerrainViewer${PLATFORM}"
ZIP_PATH="$REPO_ROOT/${PACKAGE_NAME}.zip"

STAGING_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGING_DIR"' EXIT

echo "Building Release configuration..."
cmake --build "$BUILD_DIR" --config Release

# Single-config generators (Ninja/Makefiles, the Linux/macOS default) put the binary directly in
# build-release/; multi-config generators (Visual Studio, the Windows default) nest it under a
# Release/ subfolder and add .exe. Check both so this script also works if run via Git Bash/WSL
# against a Windows build tree.
CANDIDATES=(
    "$BUILD_DIR/$EXECUTABLE_NAME"
    "$BUILD_DIR/Release/$EXECUTABLE_NAME.exe"
    "$BUILD_DIR/$EXECUTABLE_NAME.exe"
)
EXECUTABLE_PATH=""
for candidate in "${CANDIDATES[@]}"; do
    if [ -f "$candidate" ]; then
        EXECUTABLE_PATH="$candidate"
        break
    fi
done
if [ -z "$EXECUTABLE_PATH" ]; then
    echo "Error: executable not found. Checked:" >&2
    printf '  %s\n' "${CANDIDATES[@]}" >&2
    exit 1
fi

# Ship the executable under the app's own name, not the starter template's — keep the .exe
# extension if that's what was actually built.
case "$EXECUTABLE_PATH" in
    *.exe) DEST_EXECUTABLE_NAME="FractalTerrainViewer.exe" ;;
    *) DEST_EXECUTABLE_NAME="FractalTerrainViewer" ;;
esac

PKG_DIR="$STAGING_DIR/$PACKAGE_NAME"
mkdir -p "$PKG_DIR/assets/shaders" "$PKG_DIR/assets/settings"

cp "$EXECUTABLE_PATH" "$PKG_DIR/$DEST_EXECUTABLE_NAME"

for f in terrain.vert terrain.frag water.vert water.frag; do
    src="$REPO_ROOT/assets/shaders/$f"
    [ -f "$src" ] || { echo "Error: missing $src" >&2; exit 1; }
    cp "$src" "$PKG_DIR/assets/shaders/"
done

for f in window.ini camera_terrain.ini renderer_standard.ini; do
    src="$REPO_ROOT/assets/settings/$f"
    [ -f "$src" ] || { echo "Error: missing $src" >&2; exit 1; }
    cp "$src" "$PKG_DIR/assets/settings/"
done

if [ -f "$REPO_ROOT/imgui.ini" ]; then
    cp "$REPO_ROOT/imgui.ini" "$PKG_DIR/"
else
    echo "Note: imgui.ini not found at repo root, skipping (optional — recreated on first run)."
fi

rm -f "$ZIP_PATH"
(cd "$STAGING_DIR" && zip -r "$ZIP_PATH" "$PACKAGE_NAME" >/dev/null)

echo
echo "Created $ZIP_PATH"
unzip -l "$ZIP_PATH"

#!/usr/bin/env bash
# Applies local patches to the VulkanLaunchpad submodule. These fix real bugs
# in that upstream framework (e.g. a Vulkan validation error), but since it's
# a third-party repo (cg-tuwien/VulkanLaunchpad) we don't have write access
# to, the fix can't be committed there directly - only carried here as a
# patch and re-applied to every fresh checkout.
#
# Safe to re-run: a patch that's already applied is detected and skipped.
# Called automatically from the root CMakeLists.txt's configure step, so
# normally you don't need to run this by hand - it's here for manual/CI use.
set -e
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

SUBMODULE_DIR="external/VulkanLaunchpad"
PATCH_DIR="patches"

if [ ! -d "$SUBMODULE_DIR" ]; then
    echo "ERROR: $SUBMODULE_DIR not found - run 'git submodule update --init --recursive' first." >&2
    exit 1
fi

shopt -s nullglob
for patch in "$PATCH_DIR"/*.patch; do
    # git -C resolves relative paths against SUBMODULE_DIR, not our cwd - use an absolute path.
    patch="$REPO_ROOT/$patch"
    name="$(basename "$patch")"
    if git -C "$SUBMODULE_DIR" apply --check "$patch" 2>/dev/null; then
        echo "Applying $name..."
        git -C "$SUBMODULE_DIR" apply "$patch"
    elif git -C "$SUBMODULE_DIR" apply --reverse --check "$patch" 2>/dev/null; then
        echo "$name already applied, skipping."
    else
        echo "ERROR: $name does not apply cleanly and isn't already applied." >&2
        echo "       VulkanLaunchpad may have changed upstream in a conflicting way - review manually." >&2
        exit 1
    fi
done

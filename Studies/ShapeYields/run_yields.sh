#!/bin/bash
# Run shape_yields.py with the FLAF environment restored.
#
# Pinning the interpreter is not enough. soft/flaf_env's matplotlib links against a
# libstdc++ newer than /lib64's, so without the environment's LD_LIBRARY_PATH it dies with
#   ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.15' not found
# and with a bare `python3` it dies earlier still on `No module named 'uproot'`, because
# PATH then resolves to /usr/bin/python3. Sourcing env.sh gets both right.
#
# No `set -e` here: env.sh returns a non-zero status of its own in some shells, and with
# errexit the wrapper then exits silently before running anything.
#
# Usage: bash Studies/ShapeYields/run_yields.sh <args for shape_yields.py...>

WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$WT/env.sh"

echo "python3 = $(command -v python3)"
python3 -c "import uproot, matplotlib; print('uproot', uproot.__version__, '| matplotlib', matplotlib.__version__)" || exit 1
echo

exec python3 "$WT/Studies/ShapeYields/shape_yields.py" "$@"

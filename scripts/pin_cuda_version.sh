#!/bin/sh
# Print the CUDARC_CUDA_VERSION value cudarc's build script expects for this
# machine's CUDA toolkit, or nothing if the toolkit doesn't need pinning.
#
# cudarc (pulled in by candle's `cuda` feature) panics when the installed CUDA
# toolkit is newer than the versions it knows about, e.g.
# `Unsupported cuda toolkit version: 13.5`. Newer minors keep the same API
# surface, so pin to the newest minor the same major series supports.
#
# This lives in a script (instead of a `$(shell ...)` in the Makefile) because
# GNU Make 3.81 has an expansion-buffer bug: nested `$(...)` references inside
# a long `$(shell ...)` argument overflow the buffer and make reports a bogus
# `unterminated call to function 'shell'`.
#
# usage: pin_cuda_version.sh <toolkit version, e.g. 13.5> <max major> <max minor>
#        invoked by the Makefile as: pin_cuda_version.sh "$(CUDA_TOOLKIT_VERSION)"
set -eu

v="${1:-}"
max_major="${2:-13}"
max_minor="${3:-4}"
[ -n "$v" ] || exit 0

maj=${v%%.*}
min=${v#*.}

if [ "$maj" = "$max_major" ] && [ "$min" -gt "$max_minor" ] 2>/dev/null; then
	printf '%s0%s0\n' "$maj" "$max_minor"
fi

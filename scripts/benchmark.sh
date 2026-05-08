#!/usr/bin/env bash
#
# Benchmark all simpleICP implementations across all datasets.
#
# For each (implementation, dataset), invokes the implementation and parses
# the "Finished in N.NNN seconds!" line that every implementation prints.
# This gives algorithm-only timing (excludes language startup and file I/O),
# matching the methodology behind the existing README table.
#
# At the end, prints a Markdown table you can paste into README.md.
#
# Optional env vars:
#   IMPLS=cpp,python,julia,matlab          # subset of implementations
#   DATASETS=dragon,airborne,terrestrial,bunny  # subset of datasets

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

IMPLS="${IMPLS:-cpp,julia,matlab,python}"
DATASETS="${DATASETS:-dragon,airborne,terrestrial,bunny}"

RESULTS_FILE="$(mktemp)"
trap 'rm -f "${RESULTS_FILE}"' EXIT

# Dataset metadata: key | label | fixed file | movable file | max_overlap_distance ("" if none)
DATASET_DRAGON="Dragon|data/dragon1.xyz|data/dragon2.xyz|"
DATASET_AIRBORNE="Airborne Lidar|data/airborne_lidar1.xyz|data/airborne_lidar2.xyz|"
DATASET_TERRESTRIAL="Terrestrial Lidar|data/terrestrial_lidar1.xyz|data/terrestrial_lidar2.xyz|"
DATASET_BUNNY="Bunny|data/bunny_part1.xyz|data/bunny_part2.xyz|1"

dataset_meta() {
    case "$1" in
        dragon)      echo "${DATASET_DRAGON}" ;;
        airborne)    echo "${DATASET_AIRBORNE}" ;;
        terrestrial) echo "${DATASET_TERRESTRIAL}" ;;
        bunny)       echo "${DATASET_BUNNY}" ;;
        *) return 1 ;;
    esac
}

# Run a command, capture combined stdout+stderr, extract the elapsed time
# from the "Finished in N.NNN seconds!" line. Echoes only the number.
extract_elapsed() {
    local output="$1"
    echo "${output}" | grep -oE "Finished in [0-9]+\.[0-9]+ seconds" \
        | tail -n 1 \
        | grep -oE "[0-9]+\.[0-9]+" \
        || echo ""
}

run_cpp() {
    local label="$1" fixed="$2" movable="$3" max_overlap="$4"
    local bin="${REPO_ROOT}/c++/build/simpleicp"
    if [[ ! -x "${bin}" ]]; then
        echo "  C++ binary missing — run scripts/build_all.sh first" >&2
        return 1
    fi
    local args=(--fixed "${REPO_ROOT}/${fixed}" --movable "${REPO_ROOT}/${movable}")
    [[ -n "${max_overlap}" ]] && args+=(--max_overlap_distance "${max_overlap}")
    "${bin}" "${args[@]}" 2>&1
}

run_python() {
    local label="$1" fixed="$2" movable="$3" max_overlap="$4"
    local kwargs=""
    [[ -n "${max_overlap}" ]] && kwargs="max_overlap_distance=${max_overlap}"
    PYTHONPATH="${REPO_ROOT}/python" python3 - <<PY 2>&1
import numpy as np
from simpleicp import PointCloud, SimpleICP
X_fix = np.genfromtxt("${REPO_ROOT}/${fixed}")
X_mov = np.genfromtxt("${REPO_ROOT}/${movable}")
icp = SimpleICP()
icp.add_point_clouds(
    PointCloud(X_fix, columns=["x","y","z"]),
    PointCloud(X_mov, columns=["x","y","z"]),
)
icp.run(${kwargs})
PY
}

run_julia() {
    local label="$1" fixed="$2" movable="$3" max_overlap="$4"
    local kwargs=""
    [[ -n "${max_overlap}" ]] && kwargs=", max_overlap_distance=${max_overlap}"
    julia --startup-file=no --eval "
        using Pkg
        for p in [\"MultivariateStats\", \"NearestNeighbors\", \"StatsBase\"]
            haskey(Pkg.project().dependencies, p) || Pkg.add(p)
        end
        using DelimitedFiles
        include(\"${REPO_ROOT}/julia/simpleicp.jl\")
        X_fix = readdlm(\"${REPO_ROOT}/${fixed}\")
        X_mov = readdlm(\"${REPO_ROOT}/${movable}\")
        simpleicp(X_fix, X_mov${kwargs})
    " 2>&1
}

run_matlab() {
    local label="$1" fixed="$2" movable="$3" max_overlap="$4"
    local extra=""
    [[ -n "${max_overlap}" ]] && extra=", 'maxOverlapDistance', ${max_overlap}"
    # matlab -batch rejects strings with leading whitespace/newlines; keep on one line.
    matlab -batch "addpath('${REPO_ROOT}/matlab'); XFix = dlmread('${REPO_ROOT}/${fixed}'); XMov = dlmread('${REPO_ROOT}/${movable}'); simpleicp(XFix, XMov${extra});" 2>&1
}

run_one() {
    local impl="$1" dataset_key="$2"
    local meta label fixed movable max_overlap output elapsed
    meta="$(dataset_meta "${dataset_key}")"
    IFS='|' read -r label fixed movable max_overlap <<<"${meta}"

    echo "==> [${impl}] ${label}"
    if output="$("run_${impl}" "${label}" "${fixed}" "${movable}" "${max_overlap}")"; then
        elapsed="$(extract_elapsed "${output}")"
        if [[ -z "${elapsed}" ]]; then
            echo "  failed to parse timing from output:" >&2
            echo "${output}" | tail -n 20 >&2
            echo -e "${dataset_key}\t${impl}\tERR" >>"${RESULTS_FILE}"
        else
            echo "    ${elapsed}s"
            echo -e "${dataset_key}\t${impl}\t${elapsed}" >>"${RESULTS_FILE}"
        fi
    else
        echo "  command failed" >&2
        echo "${output}" | tail -n 20 >&2
        echo -e "${dataset_key}\t${impl}\tERR" >>"${RESULTS_FILE}"
    fi
}

# Format a cell from a TSV lookup, padded to a fixed width and ending in 's'.
fmt_cell() {
    local impl="$1" dataset_key="$2" width="$3"
    local raw
    raw="$(awk -F'\t' -v d="${dataset_key}" -v i="${impl}" '$1==d && $2==i {print $3}' "${RESULTS_FILE}")"
    if [[ -z "${raw}" ]]; then
        printf "%${width}s" "?"
    elif [[ "${raw}" == "-" || "${raw}" == "ERR" ]]; then
        printf "%${width}s" "${raw}"
    else
        printf "%${width}s" "${raw}s"
    fi
}

emit_table() {
    local impls=(cpp julia matlab python)
    local headers=("C++" "Julia" "Matlab" "Python")
    local widths=(5 5 6 6)
    local label_width=19

    echo
    echo "Markdown table:"
    echo
    printf "| %-${label_width}s |" "Dataset"
    for i in "${!headers[@]}"; do printf " %${widths[$i]}s |" "${headers[$i]}"; done
    echo
    printf "| :%s |" "$(printf -- '-%.0s' $(seq 1 $((label_width-1))))"
    for i in "${!widths[@]}"; do
        printf " %s: |" "$(printf -- '-%.0s' $(seq 1 $((widths[$i]-1))))"
    done
    echo

    for dataset_key in dragon airborne terrestrial bunny; do
        local meta label
        meta="$(dataset_meta "${dataset_key}")"
        label="${meta%%|*}"
        printf "| %-${label_width}s |" "*${label}*"
        for i in "${!impls[@]}"; do
            printf " %s |" "$(fmt_cell "${impls[$i]}" "${dataset_key}" "${widths[$i]}")"
        done
        echo
    done
}

IFS=',' read -ra IMPL_ARR <<<"${IMPLS}"
IFS=',' read -ra DATASET_ARR <<<"${DATASETS}"

for impl in "${IMPL_ARR[@]}"; do
    if ! declare -f "run_${impl}" >/dev/null; then
        echo "Unknown implementation: ${impl}" >&2
        exit 1
    fi
    for dataset_key in "${DATASET_ARR[@]}"; do
        run_one "${impl}" "${dataset_key}"
    done
done

emit_table

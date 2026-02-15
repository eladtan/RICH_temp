#!/usr/bin/env bash

# Shared validation helpers for local regression benchmarks.

REGRESSION_CHECK_MSG=""
REGRESSION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RICH_ROOT="$(cd "${REGRESSION_ROOT}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

set_check_msg() {
    REGRESSION_CHECK_MSG="$1"
}

has_fatal_markers() {
    local file_path="$1"
    [[ -f "$file_path" ]] || return 1

    grep -Eiq "(UniversalError|Segmentation fault|Floating point exception|core dumped|terminate called|Aborted|stack trace|Error!)" "$file_path"
}

check_no_fatal_markers() {
    local stdout_log="$1"
    local stderr_log="$2"

    if has_fatal_markers "$stdout_log"; then
        set_check_msg "fatal marker found in stdout"
        return 1
    fi
    if has_fatal_markers "$stderr_log"; then
        set_check_msg "fatal marker found in stderr"
        return 1
    fi
    return 0
}

is_nonempty_and_newer() {
    local file_path="$1"
    local start_epoch="$2"
    local file_epoch

    [[ -s "$file_path" ]] || return 1
    file_epoch=$(stat -c %Y "$file_path" 2>/dev/null || true)
    [[ -n "$file_epoch" ]] || return 1
    [[ "$file_epoch" -ge "$start_epoch" ]]
}

last_numeric_token() {
    local file_path="$1"
    awk '
        NF {
            for (i = NF; i >= 1; --i) {
                if ($i ~ /^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/) {
                    val = $i
                    break
                }
            }
        }
        END {
            if (val == "") {
                exit 1
            }
            print val
        }
    ' "$file_path"
}

is_finite_number() {
    local value="$1"
    [[ "$value" =~ ^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$ ]] || return 1
    awk -v v="$value" 'BEGIN { if (v == v) exit 0; exit 1 }'
}

check_sod_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    : "${run_dir}" "${run_start_epoch}"

    local profile_file="${run_dir}/sod_profile.txt"
    local checker_stdout="${run_dir}/sod_check.stdout.log"
    local checker_stderr="${run_dir}/sod_check.stderr.log"

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi

    if ! is_nonempty_and_newer "$profile_file" "$run_start_epoch"; then
        set_check_msg "missing or stale sod_profile.txt"
        return 1
    fi

    "${PYTHON_BIN}" "${REGRESSION_ROOT}/lib/check_sod_profile.py" \
        --profile "$profile_file" \
        --rich-root "$RICH_ROOT" \
        --max-density-gof "${SOD_MAX_DENSITY_GOF:-0.14}" \
        --max-pressure-gof "${SOD_MAX_PRESSURE_GOF:-0.06}" \
        >"$checker_stdout" 2>"$checker_stderr"
    if [[ $? -ne 0 ]]; then
        set_check_msg "Sod exact-profile comparison failed"
        return 1
    fi

    set_check_msg "Sod exact-profile comparison passed"
    return 0
}

check_sedov_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local sedov_profile="${run_dir}/sedov_profile.txt"
    local checker_stdout="${run_dir}/sedov_check.stdout.log"
    local checker_stderr="${run_dir}/sedov_check.stderr.log"

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi

    if ! is_nonempty_and_newer "$sedov_profile" "$run_start_epoch"; then
        set_check_msg "missing or stale sedov_profile.txt"
        return 1
    fi

    "${PYTHON_BIN}" "${REGRESSION_ROOT}/lib/check_sedov_exact.py" \
        --profile "$sedov_profile" \
        --rich-root "$RICH_ROOT" \
        --max-density-rel-l1 "${SEDOV_MAX_DENSITY_REL_L1:-0.70}" \
        --max-pressure-rel-l1 "${SEDOV_MAX_PRESSURE_REL_L1:-0.90}" \
        --max-velocity-rel-l1 "${SEDOV_MAX_VELOCITY_REL_L1:-0.90}" \
        >"$checker_stdout" 2>"$checker_stderr"
    if [[ $? -ne 0 ]]; then
        set_check_msg "Sedov exact-ODE comparison failed"
        return 1
    fi

    set_check_msg "Sedov exact-ODE comparison passed"
    return 0
}

check_till_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local time_file="${run_dir}/time.txt"
    local tgas_file="${run_dir}/Tgas.txt"
    local trad_file="${run_dir}/Trad.txt"
    local final_time
    local final_tgas
    local final_trad
    local rel_diff

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi

    for f in "$time_file" "$tgas_file" "$trad_file"; do
        if ! is_nonempty_and_newer "$f" "$run_start_epoch"; then
            set_check_msg "missing or stale output file: $(basename "$f")"
            return 1
        fi
    done

    if grep -Eiq "(nan|inf)" "$time_file" "$tgas_file" "$trad_file"; then
        set_check_msg "non-finite value marker found in Till outputs"
        return 1
    fi

    final_time=$(last_numeric_token "$time_file") || {
        set_check_msg "could not parse final time"
        return 1
    }
    final_tgas=$(last_numeric_token "$tgas_file") || {
        set_check_msg "could not parse final Tgas"
        return 1
    }
    final_trad=$(last_numeric_token "$trad_file") || {
        set_check_msg "could not parse final Trad"
        return 1
    }

    if ! is_finite_number "$final_time"; then
        set_check_msg "final time is not finite"
        return 1
    fi
    if ! is_finite_number "$final_tgas"; then
        set_check_msg "final Tgas is not finite"
        return 1
    fi
    if ! is_finite_number "$final_trad"; then
        set_check_msg "final Trad is not finite"
        return 1
    fi

    if ! awk -v t="$final_time" 'BEGIN { exit !(t > 0) }'; then
        set_check_msg "final time is not positive"
        return 1
    fi

    rel_diff=$(
        awk -v tg="$final_tgas" -v tr="$final_trad" '
            BEGIN {
                den = (tg > tr ? tg : tr);
                if (den <= 0) den = 1e-99;
                val = (tg > tr ? tg - tr : tr - tg) / den;
                printf "%.12e", val;
            }'
    )

    if ! awk -v r="$rel_diff" 'BEGIN { exit !(r < 1e-2) }'; then
        set_check_msg "Till final Tgas/Trad mismatch >= 1%"
        return 1
    fi

    set_check_msg "Till final Tgas and Trad agree within 1%"
    return 0
}

check_amr_random_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local metrics_file="${run_dir}/amr_random_metrics.txt"
    local mode
    local max_drift
    local threshold
    local pass_flag
    local expected_threshold

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi

    if ! is_nonempty_and_newer "$metrics_file" "$run_start_epoch"; then
        set_check_msg "missing or stale amr_random_metrics.txt"
        return 1
    fi

    mode=$(awk '$1 == "mode" { print $2 }' "$metrics_file")
    max_drift=$(awk '$1 == "max_drift" { print $2 }' "$metrics_file")
    threshold=$(awk '$1 == "threshold" { print $2 }' "$metrics_file")
    pass_flag=$(awk '$1 == "pass" { print $2 }' "$metrics_file")

    if [[ -z "$mode" || -z "$max_drift" || -z "$threshold" || -z "$pass_flag" ]]; then
        set_check_msg "failed to parse AMR random metrics"
        return 1
    fi

    if ! is_finite_number "$max_drift"; then
        set_check_msg "amr_random max_drift is not finite"
        return 1
    fi
    if ! is_finite_number "$threshold"; then
        set_check_msg "amr_random threshold is not finite"
        return 1
    fi
    if [[ "$mode" != "serial" && "$mode" != "mpi" ]]; then
        set_check_msg "amr_random mode must be serial or mpi"
        return 1
    fi
    if [[ "$pass_flag" != "0" && "$pass_flag" != "1" ]]; then
        set_check_msg "amr_random pass flag must be 0 or 1"
        return 1
    fi

    if [[ "$mode" == "serial" ]]; then
        expected_threshold="${AMR_RANDOM_MAX_DRIFT_SERIAL:-1e-8}"
    else
        expected_threshold="${AMR_RANDOM_MAX_DRIFT_MPI:-1e-6}"
    fi

    if ! awk -v d="$max_drift" -v t="$expected_threshold" 'BEGIN { exit !(d <= t) }'; then
        set_check_msg "amr_random max_drift exceeds ${mode} threshold (${max_drift} > ${expected_threshold})"
        return 1
    fi

    if ! awk -v d="$max_drift" -v t="$threshold" 'BEGIN { exit !(d <= t) }'; then
        set_check_msg "amr_random max_drift exceeds test-reported threshold (${max_drift} > ${threshold})"
        return 1
    fi

    if [[ "$pass_flag" != "1" ]]; then
        set_check_msg "amr_random test reported pass=0"
        return 1
    fi

    set_check_msg "AMR random drift check passed (${mode}, max_drift=${max_drift})"
    return 0
}

check_voronoi_volume_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local metrics_file="${run_dir}/voronoi_volume_metrics.txt"
    local rel_error
    local pass_flag
    local max_rel_error

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi

    if ! is_nonempty_and_newer "$metrics_file" "$run_start_epoch"; then
        set_check_msg "missing or stale voronoi_volume_metrics.txt"
        return 1
    fi

    rel_error=$(awk '$1 == "rel_error" { print $2 }' "$metrics_file")
    pass_flag=$(awk '$1 == "pass" { print $2 }' "$metrics_file")

    if [[ -z "$rel_error" || -z "$pass_flag" ]]; then
        set_check_msg "failed to parse voronoi volume metrics"
        return 1
    fi

    if ! is_finite_number "$rel_error"; then
        set_check_msg "voronoi_volume rel_error is not finite"
        return 1
    fi
    if [[ "$pass_flag" != "0" && "$pass_flag" != "1" ]]; then
        set_check_msg "voronoi_volume pass flag must be 0 or 1"
        return 1
    fi

    max_rel_error="${VORONOI_VOLUME_MAX_REL_ERROR:-1e-10}"

    if ! awk -v r="$rel_error" -v t="$max_rel_error" 'BEGIN { exit !(r < t) }'; then
        set_check_msg "voronoi_volume rel_error exceeds threshold (${rel_error} >= ${max_rel_error})"
        return 1
    fi

    if [[ "$pass_flag" != "1" ]]; then
        set_check_msg "voronoi_volume test reported pass=0"
        return 1
    fi

    set_check_msg "Voronoi volume check passed (rel_error=${rel_error})"
    return 0
}

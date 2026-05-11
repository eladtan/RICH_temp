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
    local attempt

    # Retry a few times to tolerate NFS attribute-cache staleness.
    # On shared filesystems the head node may briefly see stale metadata
    # for files written by SLURM compute nodes.
    for attempt in 1 2 3; do
        # Force NFS to re-read the parent directory and file attributes
        ls "$(dirname "$file_path")" > /dev/null 2>&1 || true

        if [[ -s "$file_path" ]]; then
            file_epoch=$(stat -c %Y "$file_path" 2>/dev/null || true)
            if [[ -n "$file_epoch" && "$file_epoch" -ge "$start_epoch" ]]; then
                return 0
            fi
        fi

        [[ "$attempt" -lt 3 ]] && sleep 2
    done

    return 1
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
        --max-density-gof "${SOD_MAX_DENSITY_GOF:-2e-2}" \
        --max-pressure-gof "${SOD_MAX_PRESSURE_GOF:-2e-2}" \
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
        --max-density-rel-l1 "${SEDOV_MAX_DENSITY_REL_L1:-0.50}" \
        --max-pressure-rel-l1 "${SEDOV_MAX_PRESSURE_REL_L1:-0.30}" \
        --max-velocity-rel-l1 "${SEDOV_MAX_VELOCITY_REL_L1:-0.60}" \
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

    # Energy conservation check: |E_final - E_initial| / E_initial < 1e-8
    local etotal_file="${run_dir}/Etotal.txt"
    if is_nonempty_and_newer "$etotal_file" "$run_start_epoch"; then
        local e_initial
        local e_final
        local energy_rel_err
        e_initial=$(head -n 1 "$etotal_file" | tr -d '[:space:]') || {
            set_check_msg "could not parse initial Etotal"
            return 1
        }
        e_final=$(last_numeric_token "$etotal_file") || {
            set_check_msg "could not parse final Etotal"
            return 1
        }
        if ! is_finite_number "$e_initial" || ! is_finite_number "$e_final"; then
            set_check_msg "Etotal contains non-finite values"
            return 1
        fi
        energy_rel_err=$(
            awk -v ei="$e_initial" -v ef="$e_final" '
                BEGIN {
                    if (ei <= 0) ei = 1e-99;
                    val = (ef > ei ? ef - ei : ei - ef) / ei;
                    printf "%.12e", val;
                }'
        )
        if ! awk -v r="$energy_rel_err" 'BEGIN { exit !(r < 1e-8) }'; then
            set_check_msg "Till energy conservation failed: relative error ${energy_rel_err} >= 1e-8"
            return 1
        fi
        set_check_msg "Till passed: Tgas/Trad agree within 1%, energy conserved (rel err ${energy_rel_err})"
    else
        set_check_msg "Till final Tgas and Trad agree within 1% (Etotal.txt not found, energy check skipped)"
    fi
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

check_lane_self_gravity_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local metrics_file="${run_dir}/lane_gravity_metrics.txt"
    local final_metric
    local pass_flag
    local max_metric

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi

    if ! is_nonempty_and_newer "$metrics_file" "$run_start_epoch"; then
        set_check_msg "missing or stale lane_gravity_metrics.txt"
        return 1
    fi

    final_metric=$(awk '$1 == "final_metric" { print $2 }' "$metrics_file")
    pass_flag=$(awk '$1 == "pass" { print $2 }' "$metrics_file")

    if [[ -z "$final_metric" || -z "$pass_flag" ]]; then
        set_check_msg "failed to parse lane self-gravity metrics"
        return 1
    fi

    if ! is_finite_number "$final_metric"; then
        set_check_msg "lane_self_gravity final_metric is not finite"
        return 1
    fi
    if [[ "$pass_flag" != "0" && "$pass_flag" != "1" ]]; then
        set_check_msg "lane_self_gravity pass flag must be 0 or 1"
        return 1
    fi

    max_metric="${LANE_GRAVITY_MAX_METRIC:-4e-2}"

    if ! awk -v m="$final_metric" -v t="$max_metric" 'BEGIN { am = (m < 0 ? -m : m); exit !(am < t) }'; then
        set_check_msg "lane_self_gravity |final_metric| exceeds threshold (${final_metric}, threshold ${max_metric})"
        return 1
    fi

    if [[ "$pass_flag" != "1" ]]; then
        set_check_msg "lane_self_gravity test reported pass=0"
        return 1
    fi

    set_check_msg "Lane self-gravity equilibrium check passed (final_metric=${final_metric})"
    return 0
}

check_mach2_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local profile_file="${run_dir}/mach2_profile.txt"
    local checker_stdout="${run_dir}/mach2_check.stdout.log"
    local checker_stderr="${run_dir}/mach2_check.stderr.log"

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi

    if ! is_nonempty_and_newer "$profile_file" "$run_start_epoch"; then
        set_check_msg "missing or stale mach2_profile.txt"
        return 1
    fi

    "${PYTHON_BIN}" "${REGRESSION_ROOT}/lib/check_mach2_profile.py" \
        --profile "$profile_file" \
        --rich-root "$RICH_ROOT" \
        --time 0.01 \
        --max-density-rel-l1 "${MACH2_MAX_DENSITY_REL_L1:-0.025}" \
        --max-temperature-rel-l1 "${MACH2_MAX_TEMPERATURE_REL_L1:-0.025}" \
        >"$checker_stdout" 2>"$checker_stderr"
    if [[ $? -ne 0 ]]; then
        set_check_msg "Mach2 radiative shock profile comparison failed"
        return 1
    fi

    set_check_msg "Mach2 radiative shock profile comparison passed"
    return 0
}

check_eulerian_diffusion_freefree_case_common() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local figure_suffix="$5"
    local profile_file="${run_dir}/temperature_profile.txt"
    local shock_file="${run_dir}/shock_position.txt"
    local checker_stdout="${run_dir}/freefree_check.stdout.log"
    local checker_stderr="${run_dir}/freefree_check.stderr.log"

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi

    if ! is_nonempty_and_newer "$profile_file" "$run_start_epoch"; then
        set_check_msg "missing or stale temperature_profile.txt"
        return 1
    fi

    if ! is_nonempty_and_newer "$shock_file" "$run_start_epoch"; then
        set_check_msg "missing or stale shock_position.txt"
        return 1
    fi

    "${PYTHON_BIN}" "${REGRESSION_ROOT}/lib/check_eulerian_diffusion_freefree_1d.py" \
        --profile "$profile_file" \
        --output-dir "$run_dir" \
        --figure-suffix "$figure_suffix" \
        >"$checker_stdout" 2>"$checker_stderr"
    if [[ $? -ne 0 ]]; then
        set_check_msg "free-free 1D profile validation/plot generation failed"
        return 1
    fi

    if ! is_nonempty_and_newer "${run_dir}/temperature_vs_x${figure_suffix}.png" "$run_start_epoch"; then
        set_check_msg "temperature_vs_x${figure_suffix}.png missing or stale after checker"
        return 1
    fi

    if ! is_nonempty_and_newer "${run_dir}/trad_vs_x${figure_suffix}.png" "$run_start_epoch"; then
        set_check_msg "trad_vs_x${figure_suffix}.png missing or stale after checker"
        return 1
    fi

    if ! is_nonempty_and_newer "${run_dir}/velocity_vs_x${figure_suffix}.png" "$run_start_epoch"; then
        set_check_msg "velocity_vs_x${figure_suffix}.png missing or stale after checker"
        return 1
    fi

    set_check_msg "free-free 1D profile valid and temperature/trad/velocity plots generated (${figure_suffix:-default})"
    return 0
}

check_eulerian_diffusion_freefree_1d_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    check_eulerian_diffusion_freefree_case_common "$run_dir" "$run_start_epoch" "$stdout_log" "$stderr_log" ""
}

check_eulerian_diffusion_freefree_1d_32_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    check_eulerian_diffusion_freefree_case_common "$run_dir" "$run_start_epoch" "$stdout_log" "$stderr_log" "_32"
}

check_eulerian_diffusion_freefree_1d_32_limited_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    check_eulerian_diffusion_freefree_case_common "$run_dir" "$run_start_epoch" "$stdout_log" "$stderr_log" "_32_limited"
}

check_eulerian_diffusion_freefree_suite_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"

    local compare_dir="${REGRESSION_ROOT}/cases/eulerian_diffusion_freefree_compare"
    local cases_root="${REGRESSION_ROOT}/cases"

    local profile_512="${cases_root}/eulerian_diffusion_freefree_1d/temperature_profile.txt"
    local profile_512_limited="${cases_root}/eulerian_diffusion_freefree_1d_512_limited/temperature_profile.txt"
    local profile_32="${cases_root}/eulerian_diffusion_freefree_1d_32/temperature_profile.txt"
    local profile_32_limited="${cases_root}/eulerian_diffusion_freefree_1d_32_limited/temperature_profile.txt"

    local compare_tgas="${compare_dir}/temperature_vs_x_compare_512_512_limited_32_32_limited.png"
    local compare_trad="${compare_dir}/trad_vs_x_compare_512_512_limited_32_32_limited.png"
    local compare_density="${compare_dir}/density_vs_x_compare_512_512_limited_32_32_limited.png"
    local compare_velocity="${compare_dir}/velocity_vs_x_compare_512_512_limited_32_32_limited.png"

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi

    for f in \
        "$profile_512" \
        "$profile_512_limited" \
        "$profile_32" \
        "$profile_32_limited" \
        "$compare_tgas" \
        "$compare_trad" \
        "$compare_density" \
        "$compare_velocity"; do
        if ! is_nonempty_and_newer "$f" "$run_start_epoch"; then
            set_check_msg "missing or stale output: ${f}"
            return 1
        fi
    done

    set_check_msg "free-free suite ran 4 cases and generated 4-way comparison figures"
    return 0
}

check_eulerian_diffusion_freefree_multigroup_suite_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"

    local compare_dir="${REGRESSION_ROOT}/cases/eulerian_diffusion_freefree_multigroup_compare"
    local cases_root="${REGRESSION_ROOT}/cases"

    local profile_512="${cases_root}/eulerian_diffusion_freefree_multigroup_1d/temperature_profile.txt"
    local profile_512_limited="${cases_root}/eulerian_diffusion_freefree_multigroup_1d_512_limited/temperature_profile.txt"
    local profile_32="${cases_root}/eulerian_diffusion_freefree_multigroup_1d_32/temperature_profile.txt"
    local profile_32_limited="${cases_root}/eulerian_diffusion_freefree_multigroup_1d_32_limited/temperature_profile.txt"

    local compare_tgas="${compare_dir}/temperature_vs_x_compare_mg32_512_512_limited_32_32_limited.png"
    local compare_trad="${compare_dir}/trad_vs_x_compare_mg32_512_512_limited_32_32_limited.png"
    local compare_density="${compare_dir}/density_vs_x_compare_mg32_512_512_limited_32_32_limited.png"
    local compare_velocity="${compare_dir}/velocity_vs_x_compare_mg32_512_512_limited_32_32_limited.png"

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi

    for f in \
        "$profile_512" \
        "$profile_512_limited" \
        "$profile_32" \
        "$profile_32_limited" \
        "$compare_tgas" \
        "$compare_trad" \
        "$compare_density" \
        "$compare_velocity"; do
        if ! is_nonempty_and_newer "$f" "$run_start_epoch"; then
            set_check_msg "missing or stale output: ${f}"
            return 1
        fi
    done

    set_check_msg "multigroup free-free suite ran 4 cases and generated 4-way comparison figures"
    return 0
}

check_marshak_wave_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local profile_file="${run_dir}/marshak_profile.txt"
    local problem_file="${run_dir}/problem_number.txt"
    local checker_stdout="${run_dir}/marshak_check.stdout.log"
    local checker_stderr="${run_dir}/marshak_check.stderr.log"

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi

    if ! is_nonempty_and_newer "$profile_file" "$run_start_epoch"; then
        set_check_msg "missing or stale marshak_profile.txt"
        return 1
    fi

    if [[ ! -f "$problem_file" ]]; then
        set_check_msg "missing problem_number.txt"
        return 1
    fi

    local prob_num
    prob_num=$(cat "$problem_file" | tr -d '[:space:]')

    "${PYTHON_BIN}" "${REGRESSION_ROOT}/lib/check_marshak_wave.py" \
        --problem "$prob_num" \
        --profile "$profile_file" \
        --max-tgas-rel-l1 "${MARSHAK_MAX_TGAS_REL_L1:-1e-2}" \
        --max-trad-rel-l1 "${MARSHAK_MAX_TRAD_REL_L1:-1e-2}" \
        >"$checker_stdout" 2>"$checker_stderr"
    if [[ $? -ne 0 ]]; then
        set_check_msg "Marshak wave problem ${prob_num} profile comparison failed"
        return 1
    fi

    set_check_msg "Marshak wave problem ${prob_num} profile comparison passed"
    return 0
}

check_spherical_collapse_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local metrics_file="${run_dir}/collapse_metrics.txt"
    local max_density_scatter
    local max_velocity_scatter
    local active_shell_volume_cv_max
    local active_shell_volume_ratio_max
    local guard_shell_volume_ratio_max
    local pass_flag

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi

    if ! is_nonempty_and_newer "$metrics_file" "$run_start_epoch"; then
        set_check_msg "missing or stale collapse_metrics.txt"
        return 1
    fi

    max_density_scatter=$(awk '$1 == "max_density_scatter" { print $2 }' "$metrics_file")
    max_velocity_scatter=$(awk '$1 == "max_velocity_scatter" { print $2 }' "$metrics_file")
    active_shell_volume_cv_max=$(awk '$1 == "active_shell_volume_cv_max" { print $2 }' "$metrics_file")
    active_shell_volume_ratio_max=$(awk '$1 == "active_shell_volume_ratio_max" { print $2 }' "$metrics_file")
    guard_shell_volume_ratio_max=$(awk '$1 == "guard_shell_volume_ratio_max" { print $2 }' "$metrics_file")
    pass_flag=$(awk '$1 == "pass" { print $2 }' "$metrics_file")

    if [[ -z "$max_density_scatter" || -z "$max_velocity_scatter" || -z "$active_shell_volume_cv_max" || -z "$active_shell_volume_ratio_max" || -z "$guard_shell_volume_ratio_max" || -z "$pass_flag" ]]; then
        set_check_msg "failed to parse spherical collapse metrics"
        return 1
    fi

    if ! is_finite_number "$max_density_scatter"; then
        set_check_msg "spherical_collapse max_density_scatter is not finite"
        return 1
    fi
    if ! is_finite_number "$max_velocity_scatter"; then
        set_check_msg "spherical_collapse max_velocity_scatter is not finite"
        return 1
    fi
    if ! is_finite_number "$active_shell_volume_cv_max"; then
        set_check_msg "spherical_collapse active_shell_volume_cv_max is not finite"
        return 1
    fi
    if ! is_finite_number "$active_shell_volume_ratio_max"; then
        set_check_msg "spherical_collapse active_shell_volume_ratio_max is not finite"
        return 1
    fi
    if ! is_finite_number "$guard_shell_volume_ratio_max"; then
        set_check_msg "spherical_collapse guard_shell_volume_ratio_max is not finite"
        return 1
    fi
    if [[ "$pass_flag" != "0" && "$pass_flag" != "1" ]]; then
        set_check_msg "spherical_collapse pass flag must be 0 or 1"
        return 1
    fi

    local max_scatter="${COLLAPSE_MAX_DENSITY_SCATTER:-0.1}"

    if ! awk -v d="$max_density_scatter" -v t="$max_scatter" 'BEGIN { exit !(d < t) }'; then
        set_check_msg "spherical_collapse max_density_scatter exceeds threshold (${max_density_scatter} >= ${max_scatter})"
        return 1
    fi

    local max_vel_scatter="${COLLAPSE_MAX_VELOCITY_SCATTER:-0.1}"

    if ! awk -v v="$max_velocity_scatter" -v t="$max_vel_scatter" 'BEGIN { exit !(v < t) }'; then
        set_check_msg "spherical_collapse max_velocity_scatter exceeds threshold (${max_velocity_scatter} >= ${max_vel_scatter})"
        return 1
    fi

    if [[ "$pass_flag" != "1" ]]; then
        set_check_msg "spherical_collapse test reported pass=0"
        return 1
    fi

    set_check_msg "Spherical collapse symmetry check passed (density_scatter=${max_density_scatter}, velocity_scatter=${max_velocity_scatter})"
    return 0
}

check_spherical_symmetry_tools_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local metrics_file="${run_dir}/spherical_symmetry_tools_metrics.txt"
    local pass_flag
    local flux_flag
    local update_flag
    local angular_flux_flag
    local angular_update_flag
    local angular_recenter_flag
    local angular_avg_flag

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi

    if ! is_nonempty_and_newer "$metrics_file" "$run_start_epoch"; then
        set_check_msg "missing or stale spherical_symmetry_tools_metrics.txt"
        return 1
    fi

    pass_flag=$(awk '$1 == "pass" { print $2 }' "$metrics_file")
    flux_flag=$(awk '$1 == "spherical_momentum_flux_ok" { print $2 }' "$metrics_file")
    update_flag=$(awk '$1 == "spherical_momentum_update_ok" { print $2 }' "$metrics_file")
    angular_flux_flag=$(awk '$1 == "angular_momentum_flux_ok" { print $2 }' "$metrics_file")
    angular_update_flag=$(awk '$1 == "angular_momentum_update_ok" { print $2 }' "$metrics_file")
    angular_recenter_flag=$(awk '$1 == "angular_momentum_recenter_ok" { print $2 }' "$metrics_file")
    angular_avg_flag=$(awk '$1 == "angular_momentum_avg_ok" { print $2 }' "$metrics_file")
    if [[ "$pass_flag" != "1" || "$flux_flag" != "1" || "$update_flag" != "1" ||
        "$angular_flux_flag" != "1" || "$angular_update_flag" != "1" ||
        "$angular_recenter_flag" != "1" || "$angular_avg_flag" != "1" ]]; then
        set_check_msg "spherical_symmetry_tools reported failure"
        return 1
    fi

    set_check_msg "spherical_symmetry_tools metrics passed"
    return 0
}

check_gresho_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local profile_file="${run_dir}/gresho_profile.txt"
    local test_type_file="${run_dir}/test_type.txt"
    local checker_stdout="${run_dir}/gresho_check.stdout.log"
    local checker_stderr="${run_dir}/gresho_check.stderr.log"

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi

    if ! is_nonempty_and_newer "$profile_file" "$run_start_epoch"; then
        set_check_msg "missing or stale gresho_profile.txt"
        return 1
    fi

    local test_type="euler"
    if [[ -f "$test_type_file" ]]; then
        test_type=$(cat "$test_type_file" | tr -d '[:space:]')
    fi

    local max_l1
    if [[ "$test_type" == "lagrangian" ]]; then
        max_l1="${GRESHO_LAGRANGIAN_MAX_L1:-0.05}"
    else
        max_l1="${GRESHO_EULER_MAX_L1:-0.1}"
    fi

    "${PYTHON_BIN}" "${REGRESSION_ROOT}/lib/check_gresho_profile.py" \
        --profile "$profile_file" \
        --max-vtheta-rel-l1 "$max_l1" \
        >"$checker_stdout" 2>"$checker_stderr"
    if [[ $? -ne 0 ]]; then
        set_check_msg "Gresho vortex (${test_type}) profile comparison failed"
        return 1
    fi

    set_check_msg "Gresho vortex (${test_type}) profile comparison passed"
    return 0
}

check_desmore2012_mc_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local profile_file="${run_dir}/desmore2012_mc_profile.txt"
    local reference_file="${REGRESSION_ROOT}/cases/desmore2012_mc/data/densmore2012_fig4_mc.csv"
    local checker_stdout="${run_dir}/desmore2012_mc_check.stdout.log"
    local checker_stderr="${run_dir}/desmore2012_mc_check.stderr.log"

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi

    if ! is_nonempty_and_newer "$profile_file" "$run_start_epoch"; then
        set_check_msg "missing or stale desmore2012_mc_profile.txt"
        return 1
    fi

    if [[ ! -f "$reference_file" ]]; then
        set_check_msg "missing reference file: ${reference_file}"
        return 1
    fi

    "${PYTHON_BIN}" "${REGRESSION_ROOT}/lib/check_desmore2012_mc.py" \
        --profile "$profile_file" \
        --reference "$reference_file" \
        --max-tgas-l1 "${DESMORE2012_MC_MAX_TGAS_L1:-0.05}" \
        >"$checker_stdout" 2>"$checker_stderr"
    if [[ $? -ne 0 ]]; then
        set_check_msg "Densmore 2012 MC gas temperature comparison failed"
        return 1
    fi

    set_check_msg "Densmore 2012 MC gas temperature comparison passed"
    return 0
}

check_desmore2012_mc_serial_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local profile_file="${run_dir}/desmore2012_mc_serial_profile.txt"
    local reference_file="${REGRESSION_ROOT}/cases/desmore2012_mc/data/densmore2012_fig4_mc.csv"
    local checker_stdout="${run_dir}/desmore2012_mc_serial_check.stdout.log"
    local checker_stderr="${run_dir}/desmore2012_mc_serial_check.stderr.log"

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi

    if ! is_nonempty_and_newer "$profile_file" "$run_start_epoch"; then
        set_check_msg "missing or stale desmore2012_mc_serial_profile.txt"
        return 1
    fi

    if [[ ! -f "$reference_file" ]]; then
        set_check_msg "missing reference file: ${reference_file}"
        return 1
    fi

    "${PYTHON_BIN}" "${REGRESSION_ROOT}/lib/check_desmore2012_mc.py" \
        --profile "$profile_file" \
        --reference "$reference_file" \
        --max-tgas-l1 "${DESMORE2012_MC_SERIAL_MAX_TGAS_L1:-0.05}" \
        >"$checker_stdout" 2>"$checker_stderr"
    if [[ $? -ne 0 ]]; then
        set_check_msg "Densmore 2012 serial MC+RW gas temperature comparison failed"
        return 1
    fi

    set_check_msg "Densmore 2012 serial MC+RW gas temperature comparison passed"
    return 0
}

check_yee_vortex_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local profile_file="${run_dir}/vortex_profile.txt"
    local checker_stdout="${run_dir}/vortex_check.stdout.log"
    local checker_stderr="${run_dir}/vortex_check.stderr.log"

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi

    if ! is_nonempty_and_newer "$profile_file" "$run_start_epoch"; then
        set_check_msg "missing or stale vortex_profile.txt"
        return 1
    fi

    "${PYTHON_BIN}" "${REGRESSION_ROOT}/lib/check_yee_vortex.py" \
        --profile "$profile_file" \
        --max-density-l1 "${YEE_VORTEX_MAX_DENSITY_L1:-0.05}" \
        >"$checker_stdout" 2>"$checker_stderr"
    if [[ $? -ne 0 ]]; then
        set_check_msg "Yee isentropic vortex density L1 check failed"
        return 1
    fi

    set_check_msg "Yee isentropic vortex density L1 check passed"
    return 0
}

check_spherical_gauss_linear_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local metrics_file="${run_dir}/gauss_linear_metrics.txt"
    local scalar_err
    local vel_err
    local faces_checked

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi

    if ! is_nonempty_and_newer "$metrics_file" "$run_start_epoch"; then
        set_check_msg "missing or stale gauss_linear_metrics.txt"
        return 1
    fi

    scalar_err=$(awk '$1 == "scalar_max_rel_error" { print $2 }' "$metrics_file")
    vel_err=$(awk '$1 == "velocity_max_rel_error" { print $2 }' "$metrics_file")
    faces_checked=$(awk '$1 == "faces_checked" { print $2 }' "$metrics_file")

    if [[ -z "$scalar_err" || -z "$vel_err" || -z "$faces_checked" ]]; then
        set_check_msg "failed to parse spherical gauss linear metrics"
        return 1
    fi

    if ! is_finite_number "$scalar_err"; then
        set_check_msg "scalar_max_rel_error is not finite"
        return 1
    fi
    if ! is_finite_number "$vel_err"; then
        set_check_msg "velocity_max_rel_error is not finite"
        return 1
    fi

    local max_scalar="${GAUSS_LINEAR_MAX_SCALAR_REL:-1e-8}"
    local max_vel="${GAUSS_LINEAR_MAX_VEL_REL:-0.1}"

    if ! awk -v e="$scalar_err" -v t="$max_scalar" 'BEGIN { exit !(e < t) }'; then
        set_check_msg "scalar_max_rel_error exceeds threshold (${scalar_err} >= ${max_scalar})"
        return 1
    fi

    if ! awk -v e="$vel_err" -v t="$max_vel" 'BEGIN { exit !(e < t) }'; then
        set_check_msg "velocity_max_rel_error exceeds threshold (${vel_err} >= ${max_vel})"
        return 1
    fi

    if ! awk -v n="$faces_checked" 'BEGIN { exit !(n > 0) }'; then
        set_check_msg "no faces were checked"
        return 1
    fi

    local cart_scalar_err
    local cart_vel_err
    cart_scalar_err=$(awk '$1 == "cart_scalar_max_rel_error" { print $2 }' "$metrics_file")
    cart_vel_err=$(awk '$1 == "cart_velocity_max_rel_error" { print $2 }' "$metrics_file")

    if [[ -z "$cart_scalar_err" || -z "$cart_vel_err" ]]; then
        set_check_msg "failed to parse Cartesian gauss linear metrics"
        return 1
    fi

    if ! is_finite_number "$cart_scalar_err"; then
        set_check_msg "cart_scalar_max_rel_error is not finite"
        return 1
    fi
    if ! is_finite_number "$cart_vel_err"; then
        set_check_msg "cart_velocity_max_rel_error is not finite"
        return 1
    fi

    if ! awk -v s="$scalar_err" -v c="$cart_scalar_err" 'BEGIN { exit !(s < c) }'; then
        set_check_msg "spherical scalar error not less than Cartesian (${scalar_err} >= ${cart_scalar_err})"
        return 1
    fi

    set_check_msg "Spherical Gauss linear test passed (sph_scalar_rel=${scalar_err}, cart_scalar_rel=${cart_scalar_err}, sph_vel_rel=${vel_err}, cart_vel_rel=${cart_vel_err}, faces=${faces_checked})"
    return 0
}

check_cartesian_gauss_linear_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local metrics_file="${run_dir}/cart_gauss_linear_metrics.txt"
    local cart_scalar_err
    local cart_vel_err
    local faces_checked

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi

    if ! is_nonempty_and_newer "$metrics_file" "$run_start_epoch"; then
        set_check_msg "missing or stale cart_gauss_linear_metrics.txt"
        return 1
    fi

    cart_scalar_err=$(awk '$1 == "cart_scalar_max_rel_error" { print $2 }' "$metrics_file")
    cart_vel_err=$(awk '$1 == "cart_velocity_max_rel_error" { print $2 }' "$metrics_file")
    faces_checked=$(awk '$1 == "faces_checked" { print $2 }' "$metrics_file")

    if [[ -z "$cart_scalar_err" || -z "$cart_vel_err" || -z "$faces_checked" ]]; then
        set_check_msg "failed to parse Cartesian gauss linear metrics"
        return 1
    fi

    if ! is_finite_number "$cart_scalar_err"; then
        set_check_msg "cart_scalar_max_rel_error is not finite"
        return 1
    fi
    if ! is_finite_number "$cart_vel_err"; then
        set_check_msg "cart_velocity_max_rel_error is not finite"
        return 1
    fi

    local max_scalar="${CART_GAUSS_LINEAR_MAX_SCALAR_REL:-1e-6}"
    local max_vel="${CART_GAUSS_LINEAR_MAX_VEL_REL:-0.1}"

    if ! awk -v e="$cart_scalar_err" -v t="$max_scalar" 'BEGIN { exit !(e < t) }'; then
        set_check_msg "cart_scalar_max_rel_error exceeds threshold (${cart_scalar_err} >= ${max_scalar})"
        return 1
    fi

    if ! awk -v e="$cart_vel_err" -v t="$max_vel" 'BEGIN { exit !(e < t) }'; then
        set_check_msg "cart_velocity_max_rel_error exceeds threshold (${cart_vel_err} >= ${max_vel})"
        return 1
    fi

    if ! awk -v n="$faces_checked" 'BEGIN { exit !(n > 0) }'; then
        set_check_msg "no faces were checked"
        return 1
    fi

    local sph_scalar_err
    local sph_vel_err
    sph_scalar_err=$(awk '$1 == "sph_scalar_max_rel_error" { print $2 }' "$metrics_file")
    sph_vel_err=$(awk '$1 == "sph_velocity_max_rel_error" { print $2 }' "$metrics_file")

    if [[ -z "$sph_scalar_err" || -z "$sph_vel_err" ]]; then
        set_check_msg "failed to parse spherical gauss linear metrics from Cartesian test"
        return 1
    fi

    if ! is_finite_number "$sph_scalar_err"; then
        set_check_msg "sph_scalar_max_rel_error is not finite"
        return 1
    fi
    if ! is_finite_number "$sph_vel_err"; then
        set_check_msg "sph_velocity_max_rel_error is not finite"
        return 1
    fi

    if ! awk -v c="$cart_scalar_err" -v s="$sph_scalar_err" 'BEGIN { exit !(c < s) }'; then
        set_check_msg "Cartesian scalar error not less than spherical (${cart_scalar_err} >= ${sph_scalar_err})"
        return 1
    fi

    local max_sph_vel="${CART_GAUSS_LINEAR_MAX_SPH_VEL_REL:-0.5}"
    if ! awk -v e="$sph_vel_err" -v t="$max_sph_vel" 'BEGIN { exit !(e < t) }'; then
        set_check_msg "sph_velocity_max_rel_error exceeds threshold (${sph_vel_err} >= ${max_sph_vel})"
        return 1
    fi

    set_check_msg "Cartesian Gauss linear test passed (cart_scalar_rel=${cart_scalar_err}, sph_scalar_rel=${sph_scalar_err}, cart_vel_rel=${cart_vel_err}, sph_vel_rel=${sph_vel_err}, faces=${faces_checked})"
    return 0
}

check_rayleigh_taylor_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local ek_file="${run_dir}/rt_kinetic_energy.txt"
    local slice_file="${run_dir}/rt_density_slice.txt"
    local checker_stdout="${run_dir}/rt_check.stdout.log"
    local checker_stderr="${run_dir}/rt_check.stderr.log"

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi

    if ! is_nonempty_and_newer "$ek_file" "$run_start_epoch"; then
        set_check_msg "missing or stale rt_kinetic_energy.txt"
        return 1
    fi

    local plot_dir="${run_dir}"
    local slice_arg=""
    if [[ -s "$slice_file" ]]; then
        slice_arg="--slice ${slice_file}"
    fi

    "${PYTHON_BIN}" "${REGRESSION_ROOT}/lib/check_rayleigh_taylor.py" \
        --profile "$ek_file" \
        ${slice_arg} \
        --max-growth-rate-rel-error "${RT_MAX_GROWTH_RATE_REL_ERROR:-0.25}" \
        --plot-dir "$plot_dir" \
        >"$checker_stdout" 2>"$checker_stderr"
    if [[ $? -ne 0 ]]; then
        set_check_msg "Rayleigh-Taylor growth rate comparison failed"
        return 1
    fi

    set_check_msg "Rayleigh-Taylor growth rate comparison passed"
    return 0
}


check_moving_slab_mc_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local spectrum_file="${run_dir}/moving_slab_mc_spectrum.txt"
    local checker_stdout="${run_dir}/moving_slab_mc_check.stdout.log"
    local checker_stderr="${run_dir}/moving_slab_mc_check.stderr.log"

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi

    if ! is_nonempty_and_newer "$spectrum_file" "$run_start_epoch"; then
        set_check_msg "missing or stale moving_slab_mc_spectrum.txt"
        return 1
    fi

    "${PYTHON_BIN}" "${REGRESSION_ROOT}/lib/check_moving_slab_mc.py" \
        --spectrum "$spectrum_file" \
        --max-ferror "${MOVING_SLAB_MC_MAX_FERROR:-0.30}" \
        --plot-dir "$run_dir" \
        >"$checker_stdout" 2>"$checker_stderr"
    if [[ $? -ne 0 ]]; then
        set_check_msg "Moving slab MC spectrum comparison failed"
        return 1
    fi

    set_check_msg "Moving slab MC spectrum comparison passed"
    return 0
}

check_moving_slab_mc_32_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local spectrum_file="${run_dir}/moving_slab_mc_32_spectrum.txt"
    local checker_stdout="${run_dir}/moving_slab_mc_32_check.stdout.log"
    local checker_stderr="${run_dir}/moving_slab_mc_32_check.stderr.log"

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi

    if ! is_nonempty_and_newer "$spectrum_file" "$run_start_epoch"; then
        set_check_msg "missing or stale moving_slab_mc_32_spectrum.txt"
        return 1
    fi

    "${PYTHON_BIN}" "${REGRESSION_ROOT}/lib/check_moving_slab_mc_32.py" \
        --spectrum "$spectrum_file" \
        --max-ferror "${MOVING_SLAB_MC_32_MAX_FERROR:-0.30}" \
        --plot-dir "$run_dir" \
        >"$checker_stdout" 2>"$checker_stderr"
    if [[ $? -ne 0 ]]; then
        set_check_msg "Moving slab MC 32-group spectrum comparison failed"
        return 1
    fi

    set_check_msg "Moving slab MC 32-group spectrum comparison passed"
    return 0
}

check_amr_distributed_clip_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local metrics_file="${run_dir}/amr_distributed_clip_metrics.txt"

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi

    if ! is_nonempty_and_newer "$metrics_file" "$run_start_epoch"; then
        set_check_msg "missing or stale amr_distributed_clip_metrics.txt"
        return 1
    fi

    local mass_reldiff energy_reldiff threshold pass_flag
    mass_reldiff=$(awk '$1 == "mass_reldiff" { print $2 }' "$metrics_file")
    energy_reldiff=$(awk '$1 == "energy_reldiff" { print $2 }' "$metrics_file")
    threshold=$(awk '$1 == "threshold" { print $2 }' "$metrics_file")
    pass_flag=$(awk '$1 == "pass" { print $2 }' "$metrics_file")

    if [[ -z "$mass_reldiff" || -z "$energy_reldiff" || -z "$threshold" || -z "$pass_flag" ]]; then
        set_check_msg "failed to parse AMR distributed clip metrics"
        return 1
    fi

    if ! is_finite_number "$mass_reldiff"; then
        set_check_msg "mass_reldiff is not finite"
        return 1
    fi
    if ! is_finite_number "$energy_reldiff"; then
        set_check_msg "energy_reldiff is not finite"
        return 1
    fi

    local expected_threshold="${AMR_DISTRIBUTED_CLIP_THRESHOLD:-1e-6}"

    if ! awk -v d="$mass_reldiff" -v t="$expected_threshold" 'BEGIN { exit !(d <= t) }'; then
        set_check_msg "mass_reldiff exceeds threshold (${mass_reldiff} > ${expected_threshold})"
        return 1
    fi

    if ! awk -v d="$energy_reldiff" -v t="$expected_threshold" 'BEGIN { exit !(d <= t) }'; then
        set_check_msg "energy_reldiff exceeds threshold (${energy_reldiff} > ${expected_threshold})"
        return 1
    fi

    if [[ "$pass_flag" != "1" ]]; then
        set_check_msg "amr_distributed_clip test reported pass=0"
        return 1
    fi

    set_check_msg "AMR distributed clip conservation check passed (mass_reldiff=${mass_reldiff}, energy_reldiff=${energy_reldiff})"
    return 0
}

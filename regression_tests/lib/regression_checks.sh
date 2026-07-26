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

check_radiation_direction_sampling_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local metrics_file="${run_dir}/radiation_direction_sampling_metrics.txt"

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi
    if ! is_nonempty_and_newer "$metrics_file" "$run_start_epoch"; then
        set_check_msg "missing or stale radiation_direction_sampling_metrics.txt"
        return 1
    fi

    local pure_fourth mixed_fourth chi_square voronoi_error omega_error pass_flag
    pure_fourth=$(awk '$1 == "pure_fourth_moment_error" { print $2 }' "$metrics_file")
    mixed_fourth=$(awk '$1 == "max_mixed_fourth_moment_error" { print $2 }' "$metrics_file")
    chi_square=$(awk '$1 == "reduced_chi_square" { print $2 }' "$metrics_file")
    voronoi_error=$(awk '$1 == "max_voronoi_area_rel_error" { print $2 }' "$metrics_file")
    omega_error=$(awk '$1 == "solid_angle_rel_error" { print $2 }' "$metrics_file")
    pass_flag=$(awk '$1 == "pass" { print $2 }' "$metrics_file")

    if [[ -z "$pure_fourth" || -z "$mixed_fourth" ||
          -z "$chi_square" || -z "$voronoi_error" ||
          -z "$omega_error" || -z "$pass_flag" ]]; then
        set_check_msg "failed to parse radiation direction sampling metrics"
        return 1
    fi
    if ! is_finite_number "$pure_fourth" ||
       ! is_finite_number "$mixed_fourth" ||
       ! is_finite_number "$chi_square" ||
       ! is_finite_number "$voronoi_error" ||
       ! is_finite_number "$omega_error"; then
        set_check_msg "radiation direction sampling metrics are not finite"
        return 1
    fi
    if [[ "$pass_flag" != "1" ]]; then
        set_check_msg "radiation angular geometry test reported pass=0"
        return 1
    fi

    set_check_msg "Radiation directions isotropic and observer Voronoi areas valid (pure4=${pure_fourth}, mixed4=${mixed_fourth}, chi2=${chi_square}, voronoi=${voronoi_error})"
    return 0
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
    local max_temp_rel_diff="${TILL_MAX_TEMP_REL_DIFF:-1e-2}"
    local max_energy_rel_err="${TILL_MAX_ENERGY_REL_ERR:-1e-8}"

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

    if ! awk -v r="$rel_diff" -v t="$max_temp_rel_diff" 'BEGIN { exit !(r < t) }'; then
        set_check_msg "Till final Tgas/Trad mismatch: ${rel_diff} >= ${max_temp_rel_diff}"
        return 1
    fi

    # Energy conservation check: |E_final - E_initial| / E_initial below the selected Till threshold.
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
        if ! awk -v r="$energy_rel_err" -v t="$max_energy_rel_err" 'BEGIN { exit !(r < t) }'; then
            set_check_msg "Till energy conservation failed: relative error ${energy_rel_err} >= ${max_energy_rel_err}"
            return 1
        fi
        set_check_msg "Till passed: Tgas/Trad rel diff ${rel_diff}, energy rel err ${energy_rel_err}"
    else
        set_check_msg "Till final Tgas/Trad rel diff ${rel_diff} (Etotal.txt not found, energy check skipped)"
    fi
    return 0
}

check_till_mc_case() {
    TILL_MAX_TEMP_REL_DIFF="${TILL_MC_MAX_TEMP_REL_DIFF:-2e-1}" \
    TILL_MAX_ENERGY_REL_ERR="${TILL_MC_MAX_ENERGY_REL_ERR:-5e-2}" \
    check_till_case "$@"
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

check_lane_self_gravity_fmm_case() {
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
        set_check_msg "failed to parse lane self-gravity FMM metrics"
        return 1
    fi

    if ! is_finite_number "$final_metric"; then
        set_check_msg "lane_self_gravity_fmm final_metric is not finite"
        return 1
    fi
    if [[ "$pass_flag" != "0" && "$pass_flag" != "1" ]]; then
        set_check_msg "lane_self_gravity_fmm pass flag must be 0 or 1"
        return 1
    fi

    max_metric="${LANE_GRAVITY_FMM_MAX_METRIC:-${LANE_GRAVITY_MAX_METRIC:-4e-2}}"

    if ! awk -v m="$final_metric" -v t="$max_metric" 'BEGIN { am = (m < 0 ? -m : m); exit !(am < t) }'; then
        set_check_msg "lane_self_gravity_fmm |final_metric| exceeds threshold (${final_metric}, threshold ${max_metric})"
        return 1
    fi

    if [[ "$pass_flag" != "1" ]]; then
        set_check_msg "lane_self_gravity_fmm test reported pass=0"
        return 1
    fi

    set_check_msg "Lane self-gravity FMM equilibrium check passed (final_metric=${final_metric})"
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
    pass_flag=$(awk '$1 == "pass" { print $2 }' "$metrics_file")

    if [[ -z "$max_density_scatter" || -z "$max_velocity_scatter" || -z "$pass_flag" ]]; then
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

check_desmore2012_mc_ddmc_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local profile_file="${run_dir}/desmore2012_mc_ddmc_profile.txt"
    local reference_file="${REGRESSION_ROOT}/cases/desmore2012_mc/data/densmore2012_fig4_mc.csv"
    local checker_stdout="${run_dir}/desmore2012_mc_ddmc_check.stdout.log"
    local checker_stderr="${run_dir}/desmore2012_mc_ddmc_check.stderr.log"

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi

    if ! is_nonempty_and_newer "$profile_file" "$run_start_epoch"; then
        set_check_msg "missing or stale desmore2012_mc_ddmc_profile.txt"
        return 1
    fi

    if [[ ! -f "$reference_file" ]]; then
        set_check_msg "missing reference file: ${reference_file}"
        return 1
    fi

    "${PYTHON_BIN}" "${REGRESSION_ROOT}/lib/check_desmore2012_mc.py" \
        --profile "$profile_file" \
        --reference "$reference_file" \
        --max-tgas-l1 "${DESMORE2012_MC_DDMC_MAX_TGAS_L1:-0.05}" \
        >"$checker_stdout" 2>"$checker_stderr"
    if [[ $? -ne 0 ]]; then
        set_check_msg "Densmore 2012 MC+DDMC gas temperature comparison failed"
        return 1
    fi

    set_check_msg "Densmore 2012 MC+DDMC gas temperature comparison passed"
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

check_spherical_gauss_tangential_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local metrics_file="${run_dir}/spherical_gauss_tangential_metrics.txt"

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi
    if ! is_nonempty_and_newer "$metrics_file" "$run_start_epoch"; then
        set_check_msg "missing or stale spherical_gauss_tangential_metrics.txt"
        return 1
    fi

    local max_abs max_rel faces pass_flag
    max_abs=$(awk '$1 == "sph_tangential_velocity_max_abs" { print $2 }' "$metrics_file")
    max_rel=$(awk '$1 == "sph_tangential_velocity_max_rel" { print $2 }' "$metrics_file")
    faces=$(awk '$1 == "faces_checked" { print $2 }' "$metrics_file")
    pass_flag=$(awk '$1 == "pass" { print $2 }' "$metrics_file")

    if [[ -z "$max_abs" || -z "$max_rel" || -z "$faces" || -z "$pass_flag" ]]; then
        set_check_msg "failed to parse spherical tangential metrics"
        return 1
    fi
    if ! is_finite_number "$max_abs" || ! is_finite_number "$max_rel"; then
        set_check_msg "spherical tangential metrics are not finite"
        return 1
    fi
    if ! awk -v n="$faces" 'BEGIN { exit !(n > 0) }'; then
        set_check_msg "spherical tangential checked zero faces"
        return 1
    fi

    local max_abs_allowed="${SPH_TANGENTIAL_MAX_ABS:-1e-8}"
    local max_rel_allowed="${SPH_TANGENTIAL_MAX_REL:-1e-8}"
    if ! awk -v e="$max_abs" -v t="$max_abs_allowed" 'BEGIN { exit !(e < t) }'; then
        set_check_msg "spherical tangential abs error too large: ${max_abs} >= ${max_abs_allowed}"
        return 1
    fi
    if ! awk -v e="$max_rel" -v t="$max_rel_allowed" 'BEGIN { exit !(e < t) }'; then
        set_check_msg "spherical tangential rel error too large: ${max_rel} >= ${max_rel_allowed}"
        return 1
    fi
    if [[ "$pass_flag" != "1" ]]; then
        set_check_msg "spherical tangential test reported pass=0"
        return 1
    fi

    set_check_msg "Spherical tangential face-basis check passed (abs=${max_abs}, rel=${max_rel})"
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

check_spherical_density_hardening_ddmc_push_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local summary_file="${run_dir}/spherical_push_summary.txt"
    local checker_stdout="${run_dir}/spherical_push_check.stdout.log"
    local checker_stderr="${run_dir}/spherical_push_check.stderr.log"

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi

    local required_outputs=(
        "spherical_push_summary.txt"
        "spherical_push_imc_radial_profile.txt"
        "spherical_push_ddmc_radial_profile.txt"
        "spherical_push_imc_spectrum.txt"
        "spherical_push_ddmc_spectrum.txt"
        "spherical_push_imc_angular.txt"
        "spherical_push_ddmc_angular.txt"
        "spherical_push_ddmc_diagnostics.txt"
        "spherical_push_dt_history_imc.txt"
        "spherical_push_dt_history_ddmc.txt"
        "spherical_push_snapshot_init_imc.h5"
        "spherical_push_snapshot_final_imc.h5"
        "spherical_push_snapshot_init_ddmc.h5"
        "spherical_push_snapshot_final_ddmc.h5"
        "spherical_push_final_imc.h5"
        "spherical_push_final_ddmc.h5"
    )
    for outfile in "${required_outputs[@]}"; do
        if ! is_nonempty_and_newer "${run_dir}/${outfile}" "$run_start_epoch"; then
            set_check_msg "missing or stale ${outfile}"
            return 1
        fi
    done

    "${PYTHON_BIN}" "${REGRESSION_ROOT}/lib/check_spherical_density_hardening_ddmc_push.py" \
        --run-dir "$run_dir" \
        --max-shell-vr-l1 "${SPHERICAL_PUSH_MAX_SHELL_VR_L1:-0.35}" \
        --max-shell-momentum-rel "${SPHERICAL_PUSH_MAX_SHELL_MOM_REL:-0.35}" \
        --max-erad-l1 "${SPHERICAL_PUSH_MAX_ERAD_L1:-0.35}" \
        --max-tgas-l1 "${SPHERICAL_PUSH_MAX_TGAS_L1:-0.25}" \
        --max-spectrum-l1 "${SPHERICAL_PUSH_MAX_SPECTRUM_L1:-0.45}" \
        --max-hardness-rel "${SPHERICAL_PUSH_MAX_HARDNESS_REL:-0.40}" \
        --max-cone-fraction-rel "${SPHERICAL_PUSH_MAX_CONE_FRAC_REL:-0.45}" \
        --min-thick-groups "${SPHERICAL_PUSH_MIN_THICK_GROUPS:-4}" \
        --min-thin-groups "${SPHERICAL_PUSH_MIN_THIN_GROUPS:-4}" \
        >"$checker_stdout" 2>"$checker_stderr"
    if [[ $? -ne 0 ]]; then
        set_check_msg "Spherical density-hardening DDMC push comparison failed"
        return 1
    fi

    set_check_msg "Spherical density-hardening DDMC push comparison passed"
    return 0
}

check_fmm_gravity_serial_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local metrics_file="${run_dir}/fmm_gravity_serial_metrics.txt"
    local max_scaled_error
    local max_relative_potential_error
    local m2l_count
    local p2p_pairs
    local order2_scaled_error
    local order6_scaled_error
    local pass_flag

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi

    if ! is_nonempty_and_newer "$metrics_file" "$run_start_epoch"; then
        set_check_msg "missing or stale fmm_gravity_serial_metrics.txt"
        return 1
    fi

    max_scaled_error=$(awk '$1 == "max_scaled_error" { print $2 }' "$metrics_file")
    max_relative_potential_error=$(awk '$1 == "max_relative_potential_error" { print $2 }' "$metrics_file")
    m2l_count=$(awk '$1 == "m2l_count" { print $2 }' "$metrics_file")
    p2p_pairs=$(awk '$1 == "p2p_pairs" { print $2 }' "$metrics_file")
    order2_scaled_error=$(awk '$1 == "order2_scaled_error" { print $2 }' "$metrics_file")
    order6_scaled_error=$(awk '$1 == "order6_scaled_error" { print $2 }' "$metrics_file")
    pass_flag=$(awk '$1 == "pass" { print $2 }' "$metrics_file")

    if [[ -z "$max_scaled_error" || -z "$max_relative_potential_error" || -z "$m2l_count" || -z "$p2p_pairs" || -z "$order2_scaled_error" || -z "$order6_scaled_error" || -z "$pass_flag" ]]; then
        set_check_msg "failed to parse fmm gravity serial metrics"
        return 1
    fi

    if ! is_finite_number "$max_scaled_error" || ! is_finite_number "$max_relative_potential_error" || ! is_finite_number "$order2_scaled_error" || ! is_finite_number "$order6_scaled_error"; then
        set_check_msg "fmm gravity serial metrics are not finite"
        return 1
    fi

    if ! awk -v e="$max_scaled_error" 'BEGIN { exit !(e < 2e-5) }'; then
        set_check_msg "fmm gravity serial max_scaled_error too large (${max_scaled_error})"
        return 1
    fi
    if ! awk -v e="$max_relative_potential_error" 'BEGIN { exit !(e < 5e-5) }'; then
        set_check_msg "fmm gravity serial max_relative_potential_error too large (${max_relative_potential_error})"
        return 1
    fi
    if ! awk -v n="$m2l_count" 'BEGIN { exit !(n > 0) }'; then
        set_check_msg "fmm gravity serial did not exercise M2L (${m2l_count})"
        return 1
    fi
    if ! awk -v n="$p2p_pairs" 'BEGIN { exit !(n < 552) }'; then
        set_check_msg "fmm gravity serial remained all-P2P (${p2p_pairs})"
        return 1
    fi
    if ! awk -v low="$order2_scaled_error" -v high="$order6_scaled_error" 'BEGIN { exit !(high < low) }'; then
        set_check_msg "fmm gravity serial order convergence failed (p2=${order2_scaled_error}, p6=${order6_scaled_error})"
        return 1
    fi
    if [[ "$pass_flag" != "1" ]]; then
        set_check_msg "fmm gravity serial test reported pass=0"
        return 1
    fi

    set_check_msg "FMM gravity serial check passed"
    return 0
}

check_fmm_gravity_mpi_guard_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local metrics_file="${run_dir}/fmm_gravity_mpi_guard_metrics.txt"
    local constructor_accepted
    local potential_option_rejected
    local pass_flag

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi

    if ! is_nonempty_and_newer "$metrics_file" "$run_start_epoch"; then
        set_check_msg "missing or stale fmm_gravity_mpi_guard_metrics.txt"
        return 1
    fi

    constructor_accepted=$(awk '$1 == "constructor_accepted" { print $2 }' "$metrics_file")
    potential_option_rejected=$(awk '$1 == "potential_option_rejected" { print $2 }' "$metrics_file")
    pass_flag=$(awk '$1 == "pass" { print $2 }' "$metrics_file")

    if [[ -z "$constructor_accepted" || -z "$potential_option_rejected" || -z "$pass_flag" ]]; then
        set_check_msg "failed to parse FMM MPI guard metrics"
        return 1
    fi
    if [[ "$constructor_accepted" != "1" ]]; then
        set_check_msg "FastMultipoleAcceleration3D MPI construction was not accepted"
        return 1
    fi
    if [[ "$potential_option_rejected" != "1" ]]; then
        set_check_msg "FastMultipoleAcceleration3D did not reject computePotential in MPI mode"
        return 1
    fi
    if [[ "$pass_flag" != "1" ]]; then
        set_check_msg "FMM MPI guard test reported pass=0"
        return 1
    fi

    set_check_msg "FMM MPI guard check passed (constructor accepted, potential option rejected)"
    return 0
}

check_fmm_gravity_mpi_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local metrics_file="${run_dir}/fmm_gravity_mpi_metrics.txt"
    local ranks
    local max_scaled_error
    local first_epoch
    local second_epoch
    local third_epoch
    local leaf_epoch
    local error_within_tolerance
    local topology_reused
    local rebuild_count_reused
    local leaf_topology_rebuilt
    local topology_rebuilt
    local leaf_only_rebuild
    local root_process_rebuild
    local finite_stats
    local mismatched_domain_rejected
    local leaf_storage_reused
    local root_storage_reset
    local count_only_topology_reused
    local count_only_local_plan_reused
    local pass_flag

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi
    if ! is_nonempty_and_newer "$metrics_file" "$run_start_epoch"; then
        set_check_msg "missing or stale fmm_gravity_mpi_metrics.txt"
        return 1
    fi

    ranks=$(awk '$1 == "ranks" { print $2 }' "$metrics_file")
    max_scaled_error=$(awk '$1 == "max_scaled_error" { print $2 }' "$metrics_file")
    first_epoch=$(awk '$1 == "first_epoch" { print $2 }' "$metrics_file")
    second_epoch=$(awk '$1 == "second_epoch" { print $2 }' "$metrics_file")
    third_epoch=$(awk '$1 == "third_epoch" { print $2 }' "$metrics_file")
    leaf_epoch=$(awk '$1 == "leaf_epoch" { print $2 }' "$metrics_file")
    error_within_tolerance=$(awk '$1 == "error_within_tolerance" { print $2 }' "$metrics_file")
    topology_reused=$(awk '$1 == "topology_reused" { print $2 }' "$metrics_file")
    rebuild_count_reused=$(awk '$1 == "rebuild_count_reused" { print $2 }' "$metrics_file")
    leaf_topology_rebuilt=$(awk '$1 == "leaf_topology_rebuilt" { print $2 }' "$metrics_file")
    topology_rebuilt=$(awk '$1 == "topology_rebuilt" { print $2 }' "$metrics_file")
    leaf_only_rebuild=$(awk '$1 == "leaf_only_rebuild" { print $2 }' "$metrics_file")
    root_process_rebuild=$(awk '$1 == "root_process_rebuild" { print $2 }' "$metrics_file")
    finite_stats=$(awk '$1 == "finite_stats" { print $2 }' "$metrics_file")
    mismatched_domain_rejected=$(awk '$1 == "mismatched_domain_rejected" { print $2 }' "$metrics_file")
    leaf_storage_reused=$(awk '$1 == "leaf_storage_reused" { print $2 }' "$metrics_file")
    root_storage_reset=$(awk '$1 == "root_storage_reset" { print $2 }' "$metrics_file")
    count_only_topology_reused=$(awk '$1 == "count_only_topology_reused" { print $2 }' "$metrics_file")
    count_only_local_plan_reused=$(awk '$1 == "count_only_local_plan_reused" { print $2 }' "$metrics_file")
    pass_flag=$(awk '$1 == "pass" { print $2 }' "$metrics_file")

    if [[ -z "$ranks" || -z "$max_scaled_error" || -z "$first_epoch" ||
          -z "$second_epoch" || -z "$leaf_epoch" || -z "$third_epoch" ||
          -z "$error_within_tolerance" || -z "$topology_reused" ||
          -z "$rebuild_count_reused" || -z "$leaf_topology_rebuilt" ||
          -z "$topology_rebuilt" || -z "$leaf_only_rebuild" ||
          -z "$root_process_rebuild" ||
          -z "$count_only_topology_reused" ||
          -z "$count_only_local_plan_reused" ||
          -z "$leaf_storage_reused" || -z "$root_storage_reset" ||
          -z "$finite_stats" || -z "$mismatched_domain_rejected" ||
          -z "$pass_flag" ]]; then
        set_check_msg "failed to parse distributed FMM gravity metrics"
        return 1
    fi
    if ! is_finite_number "$max_scaled_error"; then
        set_check_msg "distributed FMM max_scaled_error is not finite"
        return 1
    fi
    if ! awk -v n="$ranks" 'BEGIN { exit !(n >= 3) }'; then
        set_check_msg "distributed FMM test did not use enough ranks (${ranks})"
        return 1
    fi
    if ! awk -v e="$max_scaled_error" 'BEGIN { exit !(e < 2e-4) }' ||
       [[ "$error_within_tolerance" != "1" ]]; then
        set_check_msg "distributed FMM max_scaled_error too large (${max_scaled_error})"
        return 1
    fi
    if [[ "$first_epoch" != "$second_epoch" || "$topology_reused" != "1" ]]; then
        set_check_msg "distributed FMM failed to reuse topology after a mass-only change (${first_epoch} -> ${second_epoch})"
        return 1
    fi
    if [[ "$rebuild_count_reused" != "1" ]]; then
        set_check_msg "distributed FMM rebuild count changed after a mass-only update"
        return 1
    fi
    if ! awk -v second="$second_epoch" -v leaf="$leaf_epoch" 'BEGIN { exit !(leaf > second) }' ||
       [[ "$leaf_topology_rebuilt" != "1" || "$leaf_only_rebuild" != "1" ]]; then
        set_check_msg "distributed FMM failed the leaf-only LET rebuild (${second_epoch} -> ${leaf_epoch})"
        return 1
    fi
    if ! awk -v leaf="$leaf_epoch" -v third="$third_epoch" 'BEGIN { exit !(third > leaf) }' ||
       [[ "$topology_rebuilt" != "1" || "$root_process_rebuild" != "1" ]]; then
        set_check_msg "distributed FMM failed the full rebuild after a root breach (${leaf_epoch} -> ${third_epoch})"
        return 1
    fi
    if [[ "$leaf_storage_reused" != "1" ]]; then
        set_check_msg "distributed FMM did not recycle LET build storage on a leaf-only rebuild"
        return 1
    fi
    if [[ "$root_storage_reset" != "1" ]]; then
        set_check_msg "distributed FMM did not reset LET build storage on a full process rebuild"
        return 1
    fi
    if [[ "$count_only_topology_reused" != "1" ]]; then
        set_check_msg "distributed FMM rebuilt topology after a count-only leaf occupancy change"
        return 1
    fi
    if [[ "$count_only_local_plan_reused" != "1" ]]; then
        set_check_msg "distributed FMM failed to reuse the local plan after a count-only change"
        return 1
    fi
    if [[ "$finite_stats" != "1" ]]; then
        set_check_msg "distributed FMM emitted invalid timing, mass, or memory statistics"
        return 1
    fi
    if [[ "$mismatched_domain_rejected" != "1" ]]; then
        set_check_msg "distributed FMM did not collectively reject mismatched domains"
        return 1
    fi
    if [[ "$pass_flag" != "1" ]]; then
        set_check_msg "distributed FMM gravity test reported pass=0"
        return 1
    fi

    set_check_msg "Distributed FMM gravity reuse passed (ranks=${ranks}, scaled_error=${max_scaled_error})"
    return 0
}

check_fmm_sparse_rank_waves_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local metrics_file="${run_dir}/fmm_sparse_rank_waves_metrics.txt"
    local ranks
    local single_wave_error
    local many_wave_error
    local wave_difference
    local single_wave_count
    local many_wave_count
    local span_ratio
    local single_wave_accurate
    local many_wave_accurate
    local waves_agree
    local waves_split
    local pathology_present
    local finite
    local pass_flag

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi
    if ! is_nonempty_and_newer "$metrics_file" "$run_start_epoch"; then
        set_check_msg "missing or stale fmm_sparse_rank_waves_metrics.txt"
        return 1
    fi

    ranks=$(awk '$1 == "ranks" { print $2 }' "$metrics_file")
    single_wave_error=$(awk '$1 == "single_wave_error" { print $2 }' "$metrics_file")
    many_wave_error=$(awk '$1 == "many_wave_error" { print $2 }' "$metrics_file")
    wave_difference=$(awk '$1 == "wave_difference" { print $2 }' "$metrics_file")
    single_wave_count=$(awk '$1 == "single_wave_count" { print $2 }' "$metrics_file")
    many_wave_count=$(awk '$1 == "many_wave_count" { print $2 }' "$metrics_file")
    span_ratio=$(awk '$1 == "span_ratio" { print $2 }' "$metrics_file")
    single_wave_accurate=$(awk '$1 == "single_wave_accurate" { print $2 }' "$metrics_file")
    many_wave_accurate=$(awk '$1 == "many_wave_accurate" { print $2 }' "$metrics_file")
    waves_agree=$(awk '$1 == "waves_agree" { print $2 }' "$metrics_file")
    waves_split=$(awk '$1 == "waves_split" { print $2 }' "$metrics_file")
    pathology_present=$(awk '$1 == "pathology_present" { print $2 }' "$metrics_file")
    finite=$(awk '$1 == "finite" { print $2 }' "$metrics_file")
    pass_flag=$(awk '$1 == "pass" { print $2 }' "$metrics_file")

    if [[ -z "$ranks" || -z "$single_wave_error" || -z "$many_wave_error" ||
          -z "$wave_difference" || -z "$single_wave_count" ||
          -z "$many_wave_count" || -z "$span_ratio" ||
          -z "$single_wave_accurate" || -z "$many_wave_accurate" ||
          -z "$waves_agree" || -z "$waves_split" ||
          -z "$pathology_present" || -z "$finite" || -z "$pass_flag" ]]; then
        set_check_msg "failed to parse sparse-rank LET wave metrics"
        return 1
    fi
    if ! is_finite_number "$single_wave_error" ||
       ! is_finite_number "$many_wave_error" ||
       ! is_finite_number "$wave_difference"; then
        set_check_msg "sparse-rank LET wave errors are not finite"
        return 1
    fi
    if ! awk -v n="$ranks" 'BEGIN { exit !(n >= 3) }'; then
        set_check_msg "sparse-rank LET wave test did not use enough ranks (${ranks})"
        return 1
    fi
    if [[ "$pathology_present" != "1" ]]; then
        set_check_msg "sparse-rank geometry was not reproduced (span_ratio=${span_ratio}); the test would pass vacuously"
        return 1
    fi
    if [[ "$waves_split" != "1" ]]; then
        set_check_msg "tiny wave budget did not split the LET payload (${single_wave_count} -> ${many_wave_count})"
        return 1
    fi
    if [[ "$single_wave_accurate" != "1" ]]; then
        set_check_msg "single-wave LET disagreed with direct summation (${single_wave_error})"
        return 1
    fi
    if [[ "$many_wave_accurate" != "1" ]]; then
        set_check_msg "multi-wave LET disagreed with direct summation (${many_wave_error})"
        return 1
    fi
    if [[ "$waves_agree" != "1" ]]; then
        set_check_msg "wave count changed the result, so an interaction was dropped or double counted (${wave_difference})"
        return 1
    fi
    if [[ "$finite" != "1" ]]; then
        set_check_msg "sparse-rank LET wave test produced non-finite output"
        return 1
    fi
    if [[ "$pass_flag" != "1" ]]; then
        set_check_msg "sparse-rank LET wave test reported pass=0"
        return 1
    fi

    set_check_msg "Sparse-rank LET waves passed (ranks=${ranks}, span_ratio=${span_ratio}, waves=${single_wave_count}->${many_wave_count}, error=${many_wave_error})"
    return 0
}

check_fmm_patch_key_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local metrics_file="${run_dir}/fmm_patch_key_metrics.txt"
    local pass_flag

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi
    if ! is_nonempty_and_newer "$metrics_file" "$run_start_epoch"; then
        set_check_msg "missing or stale fmm_patch_key_metrics.txt"
        return 1
    fi
    pass_flag=$(awk '$1 == "pass" { print $2 }' "$metrics_file")
    if [[ "$pass_flag" != "1" ]]; then
        set_check_msg "FMM patch key unit test failed"
        return 1
    fi
    set_check_msg "FMM patch key unit test passed"
    return 0
}

check_fmm_packet_v5_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local metrics_file="${run_dir}/fmm_packet_v5_metrics.txt"
    local version
    local pass_flag

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi
    if ! is_nonempty_and_newer "$metrics_file" "$run_start_epoch"; then
        set_check_msg "missing or stale fmm_packet_v5_metrics.txt"
        return 1
    fi
    version=$(awk '$1 == "protocol_version" { print $2 }' "$metrics_file")
    pass_flag=$(awk '$1 == "pass" { print $2 }' "$metrics_file")
    if [[ "$version" != "5" ]]; then
        set_check_msg "FMM packet v5 test reported wrong protocol version (${version})"
        return 1
    fi
    if [[ "$pass_flag" != "1" ]]; then
        set_check_msg "FMM packet v5 round-trip test failed"
        return 1
    fi
    set_check_msg "FMM packet v5 round-trip test passed"
    return 0
}

check_fmm_patch_forest_local_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local metrics_file="${run_dir}/fmm_patch_forest_local_metrics.txt"
    local pass_flag

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi
    if ! is_nonempty_and_newer "$metrics_file" "$run_start_epoch"; then
        set_check_msg "missing or stale fmm_patch_forest_local_metrics.txt"
        return 1
    fi
    pass_flag=$(awk '$1 == "pass" { print $2 }' "$metrics_file")
    if [[ "$pass_flag" != "1" ]]; then
        set_check_msg "FMM patch forest local validation failed"
        return 1
    fi
    set_check_msg "FMM patch forest local validation passed"
    return 0
}

check_fmm_process_pair_coverage_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local metrics_file="${run_dir}/fmm_process_pair_coverage_metrics.txt"
    local ranks
    local cases
    local pass_flag

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi
    if ! is_nonempty_and_newer "$metrics_file" "$run_start_epoch"; then
        set_check_msg "missing or stale fmm_process_pair_coverage_metrics.txt"
        return 1
    fi

    ranks=$(awk '$1 == "ranks" { print $2 }' "$metrics_file")
    cases=$(awk '$1 == "cases" { print $2 }' "$metrics_file")
    pass_flag=$(awk '$1 == "pass" { print $2 }' "$metrics_file")

    if [[ -z "$ranks" || -z "$cases" || -z "$pass_flag" ]]; then
        set_check_msg "failed to parse FMM process-pair coverage metrics"
        return 1
    fi
    if ! awk -v n="$ranks" 'BEGIN { exit !(n > 1) }'; then
        set_check_msg "FMM process-pair coverage test requires multiple ranks (${ranks})"
        return 1
    fi
    if [[ "$cases" != "3" ]]; then
        set_check_msg "FMM process-pair coverage did not run all cases (${cases})"
        return 1
    fi
    if [[ "$pass_flag" != "1" ]]; then
        set_check_msg "FMM process-pair coverage test reported pass=0"
        return 1
    fi

    set_check_msg "FMM process-pair coverage check passed (ranks=${ranks}, cases=${cases})"
    return 0
}

check_fmm_peer_exchange_rebuild_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local metrics_file="${run_dir}/fmm_peer_exchange_rebuild_metrics.txt"
    local ranks
    local patterns
    local cycles
    local repeats_per_pattern
    local rounds
    local rebuild_rounds
    local reuse_rounds
    local pass_flag

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi
    if ! is_nonempty_and_newer "$metrics_file" "$run_start_epoch"; then
        set_check_msg "missing or stale fmm_peer_exchange_rebuild_metrics.txt"
        return 1
    fi

    ranks=$(awk '$1 == "ranks" { print $2 }' "$metrics_file")
    patterns=$(awk '$1 == "patterns" { print $2 }' "$metrics_file")
    cycles=$(awk '$1 == "cycles" { print $2 }' "$metrics_file")
    repeats_per_pattern=$(awk '$1 == "repeats_per_pattern" { print $2 }' "$metrics_file")
    rounds=$(awk '$1 == "rounds" { print $2 }' "$metrics_file")
    rebuild_rounds=$(awk '$1 == "rebuild_rounds" { print $2 }' "$metrics_file")
    reuse_rounds=$(awk '$1 == "reuse_rounds" { print $2 }' "$metrics_file")
    pass_flag=$(awk '$1 == "pass" { print $2 }' "$metrics_file")

    if [[ -z "$ranks" || -z "$patterns" || -z "$cycles" ||
          -z "$repeats_per_pattern" || -z "$rounds" ||
          -z "$rebuild_rounds" || -z "$reuse_rounds" ||
          -z "$pass_flag" ]]; then
        set_check_msg "failed to parse FMM peer-exchange rebuild metrics"
        return 1
    fi
    if ! awk -v n="$ranks" 'BEGIN { exit !(n > 1) }'; then
        set_check_msg "FMM peer-exchange rebuild test requires multiple ranks (${ranks})"
        return 1
    fi
    if [[ "$patterns" != "7" || "$cycles" != "4" ||
          "$repeats_per_pattern" != "2" || "$rounds" != "56" ||
          "$rebuild_rounds" != "28" || "$reuse_rounds" != "28" ]]; then
        set_check_msg "FMM peer-exchange rebuild test did not run the full graph-transition matrix"
        return 1
    fi
    if [[ "$pass_flag" != "1" ]]; then
        set_check_msg "FMM peer-exchange rebuild test reported pass=0"
        return 1
    fi

    set_check_msg "FMM peer-exchange rebuild check passed (ranks=${ranks}, rounds=${rounds})"
    return 0
}

check_fmm_operator_cache_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local metrics_file="${run_dir}/fmm_operator_cache_metrics.txt"

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi
    if ! is_nonempty_and_newer "$metrics_file" "$run_start_epoch"; then
        set_check_msg "missing or stale fmm_operator_cache_metrics.txt"
        return 1
    fi

    if ! awk '
        function finite_number(v) {
            return v ~ /^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/ &&
                   (v + 0) == (v + 0)
        }
        { value[$1] = $2 }
        END {
            required[1] = "particles"
            required[2] = "cache_budget_bytes"
            required[3] = "first_cache_bytes"
            required[4] = "first_cache_entries"
            required[5] = "first_cache_max_entries"
            required[6] = "first_cache_misses"
            required[7] = "first_cache_bypasses"
            required[8] = "second_cache_hits"
            required[9] = "zero_cache_bytes"
            required[10] = "zero_cache_entries"
            required[11] = "zero_cache_misses"
            required[12] = "zero_cache_bypasses"
            required[13] = "repeated_max_difference"
            required[14] = "fallback_max_difference"
            required[15] = "canonical_max_difference"
            required[16] = "kernel_operator_relative_difference"
            required[17] = "kernel_translation_relative_difference"
            required[18] = "canonical_cache_entries"
            required[19] = "canonical_cache_hits"
            required[20] = "canonical_cache_misses"
            required[21] = "canonical_cache_bypasses"
            required[22] = "canonical_integer_hits"
            required[23] = "canonical_integer_misses"
            required[24] = "dyadic_root_aligned"
            required[25] = "pass"
            for (i = 1; i <= 25; ++i)
                if (!(required[i] in value)) exit 1
            if (!(value["particles"] > 0 && value["cache_budget_bytes"] > 0 &&
                  value["first_cache_bytes"] <= value["cache_budget_bytes"] &&
                  value["first_cache_entries"] <= value["first_cache_max_entries"] &&
                  value["first_cache_misses"] > 0 &&
                  value["first_cache_bypasses"] > 0 &&
                  value["second_cache_hits"] > 0 &&
                  value["zero_cache_bytes"] == 0 &&
                  value["zero_cache_entries"] == 0 &&
                  value["zero_cache_misses"] > 0 &&
                  value["zero_cache_bypasses"] == value["zero_cache_misses"] &&
                  value["canonical_cache_entries"] > 0 &&
                  value["canonical_cache_hits"] > 0 &&
                  value["canonical_cache_misses"] > 0 &&
                  value["canonical_cache_bypasses"] == 0 &&
                  value["canonical_integer_hits"] == value["canonical_cache_hits"] &&
                  value["canonical_integer_misses"] == value["canonical_cache_misses"] &&
                  value["dyadic_root_aligned"] == 1 &&
                  finite_number(value["repeated_max_difference"]) &&
                  finite_number(value["fallback_max_difference"]) &&
                  finite_number(value["canonical_max_difference"]) &&
                  finite_number(value["kernel_operator_relative_difference"]) &&
                  finite_number(value["kernel_translation_relative_difference"]) &&
                  value["repeated_max_difference"] <= 5e-12 &&
                  value["fallback_max_difference"] <= 5e-12 &&
                  value["canonical_max_difference"] <= 5e-12 &&
                  value["kernel_operator_relative_difference"] <= 5e-12 &&
                  value["kernel_translation_relative_difference"] <= 5e-12 &&
                  value["pass"] == 1)) exit 1
        }
    ' "$metrics_file"; then
        set_check_msg "bounded FMM operator-cache validation failed"
        return 1
    fi

    local cache_bytes warm_hits bypasses
    cache_bytes=$(awk '$1 == "first_cache_bytes" { print $2 }' "$metrics_file")
    warm_hits=$(awk '$1 == "second_cache_hits" { print $2 }' "$metrics_file")
    bypasses=$(awk '$1 == "first_cache_bypasses" { print $2 }' "$metrics_file")
    set_check_msg "Bounded FMM operator cache passed (bytes=${cache_bytes}, warm_hits=${warm_hits}, bypasses=${bypasses})"
    return 0
}

check_fmm_mpi_scaling_benchmark_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local metrics_file="${run_dir}/fmm_mpi_scaling_benchmark_metrics.txt"
    local row_count
    local small_particles
    local large_particles
    local ranks_per_node
    local pass_flag
    local metric

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi
    if ! is_nonempty_and_newer "$metrics_file" "$run_start_epoch"; then
        set_check_msg "missing or stale fmm_mpi_scaling_benchmark_metrics.txt"
        return 1
    fi

    row_count=$(awk '$1 == "row_count" { print $2 }' "$metrics_file")
    small_particles=$(awk '$1 == "small_particles" { print $2 }' "$metrics_file")
    large_particles=$(awk '$1 == "large_particles" { print $2 }' "$metrics_file")
    ranks_per_node=$(awk '$1 == "ranks_per_node" { print $2 }' "$metrics_file")
    pass_flag=$(awk '$1 == "pass" { value = $2 } END { print value }' "$metrics_file")

    if [[ "$row_count" != "4" || "$small_particles" != "1000000" ||
          "$large_particles" != "10000000" || -z "$ranks_per_node" ]] ||
       ! awk -v n="$ranks_per_node" 'BEGIN { exit !(n >= 1 && int(n) == n) }'; then
        set_check_msg "distributed FMM scaling benchmark matrix is incomplete"
        return 1
    fi
    if [[ "$pass_flag" != "1" ]]; then
        set_check_msg "distributed FMM scaling benchmark reported pass=0"
        return 1
    fi

    if ! awk '
        function finite_number(v) {
            return v ~ /^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/ &&
                   (v + 0) == (v + 0)
        }
        $1 == "row" {
            rows++
            particles = $2 + 0
            nodes = $3 + 0
            ranks = $4 + 0
            unique_nodes = $5 + 0
            ranks_per_node = $6 + 0
            repeats = $7 + 0
            local_min = $8 + 0
            local_max = $9 + 0
            fmm_best = $10
            fmm_mean = $11
            quad_best = $12
            quad_mean = $13
            fmm_error = $14
            quad_error = $15
            speedup = $16
            fmm_rate = $17
            quad_rate = $18
            quad_walk = $23
            fmm_checksum = $24
            quad_checksum = $25
            finite_flag = $26
            run_pass = $27
            warm_best = $28
            warm_mean = $29
            cold_over_warm = $30
            persistent_bytes = $31 + 0
            local_tree_bytes = $32 + 0
            local_multipole_bytes = $33 + 0
            local_local_bytes = $34 + 0
            let_plan_bytes = $35 + 0
            operator_cache_bytes = $36 + 0
            operator_cache_budget = $37 + 0
            local_cache_bytes = $38 + 0
            local_cache_entries = $39 + 0
            local_cache_max_entries = $40 + 0
            local_cache_hits = $41 + 0
            local_cache_misses = $42 + 0
            local_cache_bypasses = $43 + 0
            let_cache_bytes = $44 + 0
            let_cache_entries = $45 + 0
            let_cache_max_entries = $46 + 0
            let_cache_hits = $47 + 0
            let_cache_misses = $48 + 0
            let_cache_bypasses = $49 + 0
            process_cache_misses = $50 + 0
            process_cache_bypasses = $51 + 0
            topology_reused = $52 + 0
            probe_count = $53 + 0
            fmm_mean_error = $54
            quad_mean_error = $55

            if (NF != 55 ||
                !((particles == 1000000 || particles == 10000000) &&
                  (nodes == 8 || nodes == 16))) bad = 1
            key = particles ":" nodes
            seen[key]++
            if (!(ranks_per_node in rpn_seen)) rpn_count++
            rpn_seen[ranks_per_node] = 1
            if (ranks_per_node < 1 || repeats < 1 ||
                ranks != nodes * ranks_per_node || unique_nodes != nodes ||
                local_min <= 0 || local_max < local_min ||
                local_max > 1.02 * local_min + 1) bad = 1
            if (!finite_number(fmm_best) || !finite_number(fmm_mean) ||
                !finite_number(quad_best) || !finite_number(quad_mean) ||
                !finite_number(fmm_error) || !finite_number(quad_error) ||
                !finite_number(speedup) || !finite_number(fmm_rate) ||
                !finite_number(quad_rate) || !finite_number(quad_walk) ||
                !finite_number(fmm_checksum) || !finite_number(quad_checksum) ||
                !finite_number(warm_best) || !finite_number(warm_mean) ||
                !finite_number(cold_over_warm) ||
                !finite_number(fmm_mean_error) ||
                !finite_number(quad_mean_error)) bad = 1
            if (!(fmm_best > 0 && fmm_mean > 0 && warm_best > 0 &&
                  warm_mean > 0 && cold_over_warm > 0 && quad_best > 0 &&
                  quad_mean > 0 && speedup > 0 && fmm_rate > 0 &&
                  quad_rate > 0 && quad_walk >= 0)) bad = 1
            if (!(persistent_bytes > 0 && local_tree_bytes > 0 &&
                  local_multipole_bytes > 0 && local_local_bytes > 0 &&
                  let_plan_bytes > 0 && operator_cache_budget >= 0 &&
                  operator_cache_bytes <= operator_cache_budget &&
                  local_cache_bytes <= operator_cache_bytes &&
                  let_cache_bytes <= operator_cache_bytes &&
                  local_cache_entries <= local_cache_max_entries &&
                  let_cache_entries <= let_cache_max_entries &&
                  local_cache_hits >= 0 && local_cache_misses >= 0 &&
                  local_cache_bypasses >= 0 &&
                  local_cache_bypasses <= local_cache_misses &&
                  let_cache_hits >= 0 && let_cache_misses >= 0 &&
                  let_cache_bypasses >= 0 &&
                  let_cache_bypasses <= let_cache_misses &&
                  process_cache_misses >= 0 &&
                  process_cache_bypasses == process_cache_misses)) bad = 1
            if (!(fmm_error < 5e-3 && quad_error < 5e-2)) bad = 1
            if (!(probe_count == 100 && fmm_mean_error >= 0 &&
                  fmm_mean_error <= fmm_error && fmm_mean_error <= 1e-3 &&
                  quad_mean_error >= 0 && quad_mean_error <= quad_error)) bad = 1
            if (finite_flag != 1 || run_pass != 1 || topology_reused != 1) bad = 1
        }
        END {
            complete = rows == 4 && rpn_count == 1 &&
                seen["1000000:8"] == 1 && seen["1000000:16"] == 1 &&
                seen["10000000:8"] == 1 && seen["10000000:16"] == 1
            exit !(complete && !bad)
        }
    ' "$metrics_file"; then
        set_check_msg "distributed FMM scaling benchmark row validation failed"
        return 1
    fi

    for metric in \
        fmm_small_8_to_16_speedup \
        fmm_small_8_to_16_efficiency \
        fmm_large_8_to_16_speedup \
        fmm_large_8_to_16_efficiency \
        quadrupole_small_8_to_16_speedup \
        quadrupole_small_8_to_16_efficiency \
        quadrupole_large_8_to_16_speedup \
        quadrupole_large_8_to_16_efficiency \
        fmm_warm_small_8_to_16_speedup \
        fmm_warm_small_8_to_16_efficiency \
        fmm_warm_large_8_to_16_speedup \
        fmm_warm_large_8_to_16_efficiency \
        fmm_small_8_cold_to_warm_speedup \
        fmm_small_16_cold_to_warm_speedup \
        fmm_large_8_cold_to_warm_speedup \
        fmm_large_16_cold_to_warm_speedup; do
        local value
        value=$(awk -v key="$metric" '$1 == key { print $2 }' "$metrics_file")
        if [[ -z "$value" ]] || ! is_finite_number "$value" ||
           ! awk -v v="$value" 'BEGIN { exit !(v > 0) }'; then
            set_check_msg "invalid scaling metric: ${metric}=${value:-missing}"
            return 1
        fi
    done

    set_check_msg "Distributed FMM/quadrupole scaling benchmark completed (1e6 and 1e7 particles on 8 and 16 nodes)"
    return 0
}

check_fmm_quadrupole_benchmark_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local metrics_file="${run_dir}/fmm_quadrupole_benchmark_metrics.txt"
    local row_count
    local largest_resolution
    local largest_fmm_seconds
    local largest_quadrupole_seconds
    local largest_fmm_error
    local largest_quadrupole_error
    local max_fmm_error
    local max_quadrupole_error
    local largest_m2l
    local largest_p2p_pairs
    local pass_flag

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi
    if ! is_nonempty_and_newer "$metrics_file" "$run_start_epoch"; then
        set_check_msg "missing or stale fmm_quadrupole_benchmark_metrics.txt"
        return 1
    fi

    row_count=$(awk '$1 == "row_count" { print $2 }' "$metrics_file")
    largest_resolution=$(awk '$1 == "largest_resolution" { print $2 }' "$metrics_file")
    largest_fmm_seconds=$(awk '$1 == "largest_fmm_seconds" { print $2 }' "$metrics_file")
    largest_quadrupole_seconds=$(awk '$1 == "largest_quadrupole_seconds" { print $2 }' "$metrics_file")
    largest_fmm_error=$(awk '$1 == "largest_fmm_scaled_error" { print $2 }' "$metrics_file")
    largest_quadrupole_error=$(awk '$1 == "largest_quadrupole_scaled_error" { print $2 }' "$metrics_file")
    max_fmm_error=$(awk '$1 == "max_fmm_scaled_error" { print $2 }' "$metrics_file")
    max_quadrupole_error=$(awk '$1 == "max_quadrupole_scaled_error" { print $2 }' "$metrics_file")
    largest_m2l=$(awk '$1 == "largest_fmm_m2l" { print $2 }' "$metrics_file")
    largest_p2p_pairs=$(awk '$1 == "largest_fmm_p2p_pairs" { print $2 }' "$metrics_file")
    pass_flag=$(awk '$1 == "pass" { print $2 }' "$metrics_file")

    if [[ "$row_count" != "5" || "$largest_resolution" != "16384" ]]; then
        set_check_msg "FMM/quadrupole benchmark resolution sweep is incomplete"
        return 1
    fi
    if ! is_finite_number "$largest_fmm_seconds" ||
       ! is_finite_number "$largest_quadrupole_seconds" ||
       ! is_finite_number "$largest_fmm_error" ||
       ! is_finite_number "$largest_quadrupole_error" ||
       ! is_finite_number "$max_fmm_error" ||
       ! is_finite_number "$max_quadrupole_error"; then
        set_check_msg "FMM/quadrupole benchmark emitted non-finite metrics"
        return 1
    fi
    if ! awk -v t="$largest_fmm_seconds" 'BEGIN { exit !(t > 0) }' ||
       ! awk -v t="$largest_quadrupole_seconds" 'BEGIN { exit !(t > 0) }'; then
        set_check_msg "FMM/quadrupole benchmark emitted invalid runtimes"
        return 1
    fi
    if ! awk -v e="$max_fmm_error" 'BEGIN { exit !(e < 5e-3) }'; then
        set_check_msg "FMM benchmark scaled error too large (${max_fmm_error})"
        return 1
    fi
    if ! awk -v e="$max_quadrupole_error" 'BEGIN { exit !(e < 5e-2) }'; then
        set_check_msg "quadrupole benchmark scaled error too large (${max_quadrupole_error})"
        return 1
    fi
    if ! awk -v fmm="$largest_fmm_seconds" -v tree="$largest_quadrupole_seconds" \
        'BEGIN { exit !(fmm < tree) }'; then
        set_check_msg "FMM is not faster at N=16384 (${largest_fmm_seconds}s vs ${largest_quadrupole_seconds}s)"
        return 1
    fi
    if ! awk -v fmm="$largest_fmm_error" -v tree="$largest_quadrupole_error" \
        'BEGIN { exit !(fmm <= 1.25 * tree) }'; then
        set_check_msg "FMM accuracy is not comparable at N=16384 (${largest_fmm_error} vs ${largest_quadrupole_error})"
        return 1
    fi
    if ! awk -v n="$largest_m2l" 'BEGIN { exit !(n > 0) }'; then
        set_check_msg "FMM benchmark did not exercise M2L"
        return 1
    fi
    if ! awk -v n="$largest_p2p_pairs" 'BEGIN { exit !(n < 268419072) }'; then
        set_check_msg "FMM benchmark remained all-P2P (${largest_p2p_pairs})"
        return 1
    fi
    if [[ "$pass_flag" != "1" ]]; then
        set_check_msg "FMM/quadrupole benchmark reported pass=0"
        return 1
    fi

    set_check_msg "FMM/quadrupole benchmark check passed"
    return 0
}

check_ddmc_moving_interface_ab_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local metrics_file="${run_dir}/ddmc_moving_interface_ab_metrics.txt"

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi
    if ! is_nonempty_and_newer "$metrics_file" "$run_start_epoch"; then
        set_check_msg "missing or stale ddmc_moving_interface_ab_metrics.txt"
        return 1
    fi

    local static_admitted corrected_admitted expected_factor measured_factor
    local factor_rel_error static_weight_error static_gu corrected_gu
    local static_fallback corrected_fallback static_bypass corrected_bypass pass_flag
    static_admitted=$(awk '$1 == "static_admitted" { print $2 }' "$metrics_file")
    corrected_admitted=$(awk '$1 == "corrected_admitted" { print $2 }' "$metrics_file")
    expected_factor=$(awk '$1 == "expected_moving_factor" { print $2 }' "$metrics_file")
    measured_factor=$(awk '$1 == "measured_moving_factor" { print $2 }' "$metrics_file")
    factor_rel_error=$(awk '$1 == "moving_factor_rel_error" { print $2 }' "$metrics_file")
    static_weight_error=$(awk '$1 == "static_weight_error" { print $2 }' "$metrics_file")
    static_gu=$(awk '$1 == "static_gu_applied" { print $2 }' "$metrics_file")
    corrected_gu=$(awk '$1 == "corrected_gu_applied" { print $2 }' "$metrics_file")
    static_fallback=$(awk '$1 == "static_gu_fallback" { print $2 }' "$metrics_file")
    corrected_fallback=$(awk '$1 == "corrected_gu_fallback" { print $2 }' "$metrics_file")
    static_bypass=$(awk '$1 == "static_bypass" { print $2 }' "$metrics_file")
    corrected_bypass=$(awk '$1 == "corrected_bypass" { print $2 }' "$metrics_file")
    pass_flag=$(awk '$1 == "pass" { print $2 }' "$metrics_file")

    if [[ -z "$static_admitted" || -z "$corrected_admitted" ||
          -z "$expected_factor" || -z "$measured_factor" ||
          -z "$factor_rel_error" || -z "$static_weight_error" ||
          -z "$static_gu" || -z "$corrected_gu" ||
          -z "$static_fallback" || -z "$corrected_fallback" ||
          -z "$static_bypass" || -z "$corrected_bypass" ||
          -z "$pass_flag" ]]; then
        set_check_msg "failed to parse DDMC moving-interface A/B metrics"
        return 1
    fi

    if ! is_finite_number "$expected_factor" ||
       ! is_finite_number "$measured_factor" ||
       ! is_finite_number "$factor_rel_error" ||
       ! is_finite_number "$static_weight_error"; then
        set_check_msg "DDMC moving-interface A/B metrics are not finite"
        return 1
    fi

    if [[ "$static_admitted" != "$corrected_admitted" ]]; then
        set_check_msg "moving correction changed the static admission decisions (${static_admitted} != ${corrected_admitted})"
        return 1
    fi
    if ! awk -v n="$static_admitted" 'BEGIN { exit !(n > 500) }'; then
        set_check_msg "too few admitted interface packets (${static_admitted})"
        return 1
    fi
    if ! awk -v g="$expected_factor" 'BEGIN { d = g - 1; if (d < 0) d = -d; exit !(d > 0.02) }'; then
        set_check_msg "A/B setup does not produce a meaningful moving factor (${expected_factor})"
        return 1
    fi

    local max_factor_error="${DDMC_MOVING_AB_MAX_FACTOR_REL_ERROR:-1e-9}"
    local max_static_error="${DDMC_MOVING_AB_MAX_STATIC_WEIGHT_ERROR:-1e-10}"
    if ! awk -v e="$factor_rel_error" -v t="$max_factor_error" 'BEGIN { exit !(e <= t) }'; then
        set_check_msg "measured G_U factor error too large (${factor_rel_error} > ${max_factor_error})"
        return 1
    fi
    if ! awk -v e="$static_weight_error" -v t="$max_static_error" 'BEGIN { exit !(e <= t) }'; then
        set_check_msg "static admitted weight changed (${static_weight_error} > ${max_static_error})"
        return 1
    fi

    if [[ "$static_gu" != "0" || "$corrected_gu" == "0" ||
          "$static_fallback" != "0" || "$corrected_fallback" != "0" ||
          "$static_bypass" != "0" || "$corrected_bypass" != "0" ]]; then
        set_check_msg "unexpected G_U diagnostics (off=${static_gu}, on=${corrected_gu}, fallback=${static_fallback}/${corrected_fallback}, bypass=${static_bypass}/${corrected_bypass})"
        return 1
    fi
    if [[ "$pass_flag" != "1" ]]; then
        set_check_msg "DDMC moving-interface A/B executable reported pass=0"
        return 1
    fi

    set_check_msg "DDMC moving-interface A/B passed (G_expected=${expected_factor}, G_measured=${measured_factor}, rel_error=${factor_rel_error}, admitted=${static_admitted})"
    return 0
}

check_ddmc_mpi_zero_cell_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local metrics_file="${run_dir}/ddmc_mpi_zero_cell_metrics.txt"

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi
    if ! is_nonempty_and_newer "$metrics_file" "$run_start_epoch"; then
        set_check_msg "missing or stale ddmc_mpi_zero_cell_metrics.txt"
        return 1
    fi

    local zero_ranks cross_rank_faces remote_leaks flux_reductions
    local invalid_geometry reciprocity_max cross_rank_reciprocity
    local rate_conductance_consistency weight_rel_error pass_flag
    zero_ranks=$(awk '$1 == "zero_rank_count" { print $2 }' "$metrics_file")
    cross_rank_faces=$(awk '$1 == "cross_rank_faces" { print $2 }' "$metrics_file")
    remote_leaks=$(awk '$1 == "remote_resident_leaks" { print $2 }' "$metrics_file")
    flux_reductions=$(awk '$1 == "mpi_face_flux_reductions" { print $2 }' "$metrics_file")
    invalid_geometry=$(awk '$1 == "invalid_geometry" { print $2 }' "$metrics_file")
    reciprocity_max=$(awk '$1 == "reciprocity_max" { print $2 }' "$metrics_file")
    cross_rank_reciprocity=$(awk '$1 == "cross_rank_reciprocity_rel_error" { print $2 }' "$metrics_file")
    rate_conductance_consistency=$(awk '$1 == "rate_conductance_consistency_max" { print $2 }' "$metrics_file")
    weight_rel_error=$(awk '$1 == "weight_rel_error" { print $2 }' "$metrics_file")
    pass_flag=$(awk '$1 == "pass" { print $2 }' "$metrics_file")

    if [[ -z "$zero_ranks" || -z "$cross_rank_faces" ||
          -z "$remote_leaks" || -z "$flux_reductions" ||
          -z "$invalid_geometry" || -z "$reciprocity_max" ||
          -z "$cross_rank_reciprocity" ||
          -z "$rate_conductance_consistency" ||
          -z "$weight_rel_error" || -z "$pass_flag" ]]; then
        set_check_msg "failed to parse DDMC zero-cell MPI metrics"
        return 1
    fi
    if ! is_finite_number "$reciprocity_max" ||
       ! is_finite_number "$cross_rank_reciprocity" ||
       ! is_finite_number "$rate_conductance_consistency" ||
       ! is_finite_number "$weight_rel_error"; then
        set_check_msg "DDMC zero-cell MPI metrics are not finite"
        return 1
    fi

    if ! awk -v n="$zero_ranks" 'BEGIN { exit !(n > 0) }'; then
        set_check_msg "DDMC zero-cell MPI test did not create an empty rank"
        return 1
    fi
    if ! awk -v n="$cross_rank_faces" 'BEGIN { exit !(n > 0) }'; then
        set_check_msg "DDMC zero-cell MPI test found no cross-rank faces"
        return 1
    fi
    if ! awk -v n="$remote_leaks" 'BEGIN { exit !(n > 0) }'; then
        set_check_msg "DDMC zero-cell MPI test sampled no remote resident leaks"
        return 1
    fi
    if ! awk -v n="$flux_reductions" 'BEGIN { exit !(n > 0) }'; then
        set_check_msg "DDMC zero-cell MPI test performed no face-flux reductions"
        return 1
    fi
    if [[ "$invalid_geometry" != "0" ]]; then
        set_check_msg "DDMC zero-cell MPI test reported invalid leakage geometry (${invalid_geometry})"
        return 1
    fi

    local max_reciprocity="${DDMC_MPI_MAX_RECIPROCITY_ERROR:-1e-10}"
    local max_cross_rank_reciprocity="${DDMC_MPI_MAX_CROSS_RANK_RECIPROCITY_ERROR:-1e-12}"
    local max_rate_consistency="${DDMC_MPI_MAX_RATE_CONDUCTANCE_ERROR:-1e-12}"
    local max_weight_error="${DDMC_MPI_MAX_WEIGHT_REL_ERROR:-1e-10}"
    if ! awk -v e="$reciprocity_max" -v t="$max_reciprocity" 'BEGIN { exit !(e <= t) }'; then
        set_check_msg "DDMC MPI reciprocity error too large (${reciprocity_max} > ${max_reciprocity})"
        return 1
    fi
    if ! awk -v e="$cross_rank_reciprocity" -v t="$max_cross_rank_reciprocity" 'BEGIN { exit !(e <= t) }'; then
        set_check_msg "DDMC cross-rank reciprocity error too large (${cross_rank_reciprocity} > ${max_cross_rank_reciprocity})"
        return 1
    fi
    if ! awk -v e="$rate_conductance_consistency" -v t="$max_rate_consistency" 'BEGIN { exit !(e <= t) }'; then
        set_check_msg "DDMC rate/conductance mismatch too large (${rate_conductance_consistency} > ${max_rate_consistency})"
        return 1
    fi
    if ! awk -v e="$weight_rel_error" -v t="$max_weight_error" 'BEGIN { exit !(e <= t) }'; then
        set_check_msg "DDMC MPI particle weight is not conserved (${weight_rel_error} > ${max_weight_error})"
        return 1
    fi
    if [[ "$pass_flag" != "1" ]]; then
        set_check_msg "DDMC zero-cell MPI executable reported pass=0"
        return 1
    fi

    set_check_msg "DDMC zero-cell/cross-rank MPI passed (zero_ranks=${zero_ranks}, cross_faces=${cross_rank_faces}, remote_leaks=${remote_leaks}, reciprocity=${cross_rank_reciprocity}, weight_error=${weight_rel_error})"
    return 0
}


check_fmm_patch_let_mpi_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local metrics_file="${run_dir}/fmm_patch_let_mpi_metrics.txt"

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi
    if ! is_nonempty_and_newer "$metrics_file" "$run_start_epoch"; then
        set_check_msg "missing or stale fmm_patch_let_mpi_metrics.txt"
        return 1
    fi

    local pass error waves nodes mass_pass
    pass="$(awk '$1 == "pass" {print $2}' "$metrics_file" | tail -n 1)"
    error="$(awk '$1 == "max_relative_error" {print $2}' "$metrics_file" | tail -n 1)"
    waves="$(awk '$1 == "max_wave_count" {print $2}' "$metrics_file" | tail -n 1)"
    nodes="$(awk '$1 == "max_process_nodes" {print $2}' "$metrics_file" | tail -n 1)"
    mass_pass="$(awk '$1 == "root_mass_pass" {print $2}' "$metrics_file" | tail -n 1)"
    if [[ "$pass" != "1" || "$mass_pass" != "1" ]]; then
        set_check_msg "patch LET MPI reported failure (pass=${pass}, root_mass_pass=${mass_pass})"
        return 1
    fi
    if [[ -z "$error" ]] || ! awk -v e="$error" 'BEGIN { exit !(e < 0.08) }'; then
        set_check_msg "patch LET MPI error exceeds tolerance (${error})"
        return 1
    fi
    if [[ -z "$waves" ]] || ! awk -v w="$waves" 'BEGIN { exit !(w > 1) }'; then
        set_check_msg "patch LET test did not exercise multiple waves (${waves})"
        return 1
    fi
    if [[ -z "$nodes" ]] || ! awk -v n="$nodes" 'BEGIN { exit !(n > 8) }'; then
        set_check_msg "patch process tree did not contain multiple patch leaves (${nodes})"
        return 1
    fi

    set_check_msg "patch-aware distributed LET passed (error=${error}, waves=${waves}, process_nodes=${nodes})"
    return 0
}


check_fmm_patch_process_pair_coverage_case() {
    local run_dir="$1"
    local run_start_epoch="$2"
    local stdout_log="$3"
    local stderr_log="$4"
    local metrics_file="${run_dir}/fmm_patch_process_pair_coverage_metrics.txt"

    if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
        return 1
    fi
    if ! is_nonempty_and_newer "$metrics_file" "$run_start_epoch"; then
        set_check_msg "missing or stale fmm_patch_process_pair_coverage_metrics.txt"
        return 1
    fi
    local pass
    pass="$(awk '$1 == "pass" {print $2}' "$metrics_file" | tail -n 1)"
    if [[ "$pass" != "1" ]]; then
        set_check_msg "patch process-pair coverage failed"
        return 1
    fi
    set_check_msg "patch process-pair exact-one coverage passed"
    return 0
}

"""Compare fixed PID gain candidates on the declared development scenario.

This is calibration on one scenario, not independent performance evaluation.
"""
from dataclasses import replace
import json
from pathlib import Path

from run_three_axis import Config, simulate


def main():
    baseline = Config(kp=(1.8, 1.5, 1.2), ki=(.10, .08, .06))
    candidates = [(1., 1.), (1., 2.), (1., 3.), (1., 4.),
                  (1.2, 2.), (1.2, 3.), (.8, 2.), (.8, 3.),
                  (1., 0.), (1., .25), (1., .5), (1.2, .5),
                  (1.5, .5), (1.5, 1.)]
    records = []
    for proportional_scale, integral_scale in candidates:
        config = replace(baseline,
                         kp=tuple(v*proportional_scale for v in baseline.kp),
                         ki=tuple(v*integral_scale for v in baseline.ki))
        _, metrics = simulate(config)
        records.append({
            'kp': config.kp, 'ki': config.ki, 'kd': config.kd,
            'success_last_5s': metrics['success_last_5s'],
            'max_attitude_error_last_5s_deg': metrics['max_attitude_error_last_5s_deg'],
            'max_rate_norm_last_5s_deg_s': metrics['max_rate_norm_last_5s_deg_s'],
            'settling_time_s': metrics['settling_time_s'],
            'control_cost_N2m2_s': metrics['control_cost_N2m2_s'],
        })
    out = Path(__file__).resolve().parent / 'outputs'
    out.mkdir(exist_ok=True)
    result = {'purpose': 'PID calibration on the fixed development scenario; not held-out testing',
              'seed': baseline.seed, 'thresholds_unchanged': True,
              'disturbance_and_noise_unchanged': True, 'candidates': records}
    (out / 'pid_calibration.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()

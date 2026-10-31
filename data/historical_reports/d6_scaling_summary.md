# State vs Mechanism Coverage Scaling

Held-out variant: `mass_high_delay_2`; all reports use test groups 192-255.

| windows | state rollout MSE | mechanism rollout MSE | mechanism - state | paired 95% CI |
| ---: | ---: | ---: | ---: | ---: |
| 96 | 1.0946 +/- 0.0565 (2 seeds) | 1.2122 +/- 0.0511 (2 seeds) | 0.1176 (10.7%) | [0.0982, 0.1374] |
| 192 | 1.2645 +/- 0.0271 (2 seeds) | 1.2871 +/- 0.0327 (2 seeds) | 0.0226 (1.8%) | [-0.0055, 0.0545] |
| 384 | 1.3838 +/- 0.0163 (3 seeds) | 1.3361 +/- 0.0216 (4 seeds) | -0.0477 (-3.4%) | [-0.0778, -0.0181] |

All paired intervals are bootstrapped over the 64 held-out groups after averaging seeds.

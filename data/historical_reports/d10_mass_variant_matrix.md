# Equal-Unique-Window Factor Ablation: mass_only_unique

State seeds: 3; mechanism seeds: 2; test groups: 64; each training arm has 384 unique windows.

| variant | state rollout | mechanism rollout | delta | relative | paired 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| delay_2 | 1.4071 +/- 0.0078 | 1.2941 +/- 0.0325 | -0.1130 | -8.0% | [-0.1314, -0.0950] |
| mass_high | 1.3966 +/- 0.0090 | 1.2892 +/- 0.0156 | -0.1074 | -7.7% | [-0.1220, -0.0929] |
| mass_high_delay_2 | 1.4101 +/- 0.0092 | 1.2955 +/- 0.0303 | -0.1146 | -8.1% | [-0.1323, -0.0967] |
| nominal | 1.3957 +/- 0.0054 | 1.2850 +/- 0.0189 | -0.1107 | -7.9% | [-0.1268, -0.0947] |

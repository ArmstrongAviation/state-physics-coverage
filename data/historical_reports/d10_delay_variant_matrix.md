# Equal-Unique-Window Factor Ablation: delay_only_unique

State seeds: 3; mechanism seeds: 2; test groups: 64; each training arm has 384 unique windows.

| variant | state rollout | mechanism rollout | delta | relative | paired 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| delay_2 | 1.4071 +/- 0.0078 | 1.4127 +/- 0.0059 | 0.0056 | 0.4% | [-0.0160, 0.0260] |
| mass_high | 1.3966 +/- 0.0090 | 1.4110 +/- 0.0035 | 0.0144 | 1.0% | [-0.0014, 0.0306] |
| mass_high_delay_2 | 1.4101 +/- 0.0092 | 1.4155 +/- 0.0069 | 0.0054 | 0.4% | [-0.0147, 0.0249] |
| nominal | 1.3957 +/- 0.0054 | 1.4093 +/- 0.0034 | 0.0135 | 1.0% | [-0.0060, 0.0337] |

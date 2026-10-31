# Minimal revision experiments

Primary held-out composition: mass_high_delay_2. All arms: 384 window records, 10,000 updates, 3 training seeds.

| Endpoint | State | Paired | Unpaired |
|---|---:|---:|---:|
| rollout_mse | 1.410086 +/- 0.009232 | 1.362306 +/- 0.003905 | 1.410930 +/- 0.010024 |
| rollout_block_euclidean_m | 0.148707 +/- 0.003639 | 0.159086 +/- 0.008652 | 0.152849 +/- 0.012702 |
| rollout_effector_euclidean_m | 0.144120 +/- 0.003249 | 0.152742 +/- 0.010057 | 0.147721 +/- 0.012614 |
| observed_block_euclidean_m | 0.115065 +/- 0.003046 | 0.117797 +/- 0.005197 | 0.115831 +/- 0.000797 |
| true_state_persistence_block_euclidean_m | 0.134604 +/- 0.000000 | 0.134604 +/- 0.000000 | 0.134604 +/- 0.000000 |

Differences below use a two-way bootstrap of training seeds and paired test episodes; conditional group-only intervals are also saved in summary.json.

## rollout_mse
- paired_minus_state: -0.047780; seed+group 95% CI [-0.07286231356491853, -0.023773940822488963]
- unpaired_minus_state: 0.000845; seed+group 95% CI [-0.025350006497292604, 0.0290813813574561]
- paired_minus_unpaired: -0.048624; seed+group 95% CI [-0.07317157518420436, -0.023818100187539448]

## rollout_block_euclidean_m
- paired_minus_state: 0.010379; seed+group 95% CI [-0.0050541851864622355, 0.025226209451960364]
- unpaired_minus_state: 0.004142; seed+group 95% CI [-0.012320973418477854, 0.020579387911571705]
- paired_minus_unpaired: 0.006237; seed+group 95% CI [-0.013255204776884069, 0.024644437128293752]

## rollout_effector_euclidean_m
- paired_minus_state: 0.008622; seed+group 95% CI [-0.007412247949919679, 0.0242149988747323]
- unpaired_minus_state: 0.003602; seed+group 95% CI [-0.012590183187786808, 0.01954685378071417]
- paired_minus_unpaired: 0.005021; seed+group 95% CI [-0.014682394991974218, 0.024121430628683754]

Collection-policy limitation: unpaired data cover more distinct source starts than paired data; this is not an isolated loss-level pairing test.
The original state pool has 192 records but 191 unique source starts. New unpaired data preserve that pool and keep the duplicate within one physics condition.
Physical results use common nominal train-only linear decoders; they are decoded predictions, not successful control.
No result-dependent choice of seeds, checkpoints, or test episodes was made.

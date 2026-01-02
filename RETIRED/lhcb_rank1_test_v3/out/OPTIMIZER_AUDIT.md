# Optimizer Audit Report

## Verification: Λ >= 0

### wide_linear

- NLL_unc (A+B): 1325.060553
- NLL_con: 1331.001481
- Λ = 2*(NLL_con - NLL_unc) = 11.881855
- **Λ >= 0: ✓ PASS**

Optimizer statistics:
- Channel A: 240 evaluations, best NLL = 740.4979, NLL range = [740.4979, 817.7407]
- Channel B: 240 evaluations, best NLL = 584.5627, NLL range = [584.5627, 699.0430]
- Joint: 160 evaluations, best NLL = 1331.0015

### wide_quadratic

- NLL_unc (A+B): 1125.290352
- NLL_con: 1130.160321
- Λ = 2*(NLL_con - NLL_unc) = 9.739939
- **Λ >= 0: ✓ PASS**

Optimizer statistics:
- Channel A: 240 evaluations, best NLL = 620.9861, NLL range = [620.9861, 710.6137]
- Channel B: 240 evaluations, best NLL = 504.3043, NLL range = [504.3043, 4559.1339]
- Joint: 160 evaluations, best NLL = 1130.1603

### tight_linear

- NLL_unc (A+B): 867.197556
- NLL_con: 872.141618
- Λ = 2*(NLL_con - NLL_unc) = 9.888124
- **Λ >= 0: ✓ PASS**

Optimizer statistics:
- Channel A: 240 evaluations, best NLL = 488.8579, NLL range = [488.8579, 540.9718]
- Channel B: 240 evaluations, best NLL = 378.3397, NLL range = [378.3397, 446.3729]
- Joint: 160 evaluations, best NLL = 872.1416

### tight_quadratic

- NLL_unc (A+B): 742.908784
- NLL_con: 744.699646
- Λ = 2*(NLL_con - NLL_unc) = 3.581724
- **Λ >= 0: ✓ PASS**

Optimizer statistics:
- Channel A: 240 evaluations, best NLL = 398.5191, NLL range = [398.5191, 467.5360]
- Channel B: 240 evaluations, best NLL = 344.3897, NLL range = [344.3897, 1022.5894]
- Joint: 160 evaluations, best NLL = 744.6996


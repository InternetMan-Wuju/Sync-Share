"""Run only after inspecting smoke correctness. Sequential, fixed A/B design."""
import subprocess,json
from pathlib import Path
P=Path('/runexp')
def run(case,mode,name,*flags):
    subprocess.run(['python3','/runexp/scripts/client.py','--case',str(case),'--mode',mode,'--name',name,*flags],check=True)
# Validate all remaining inputs on the same hidden/routing before any timing.
for c in (0,1,2):run(c,'active_cold',f'validate_cloned_cold_{c}','--profile','--verify')
# Warm every path and input; excluded from paired E2E summary.
for c in range(3):
    for m in ('off','shadow','active_cold','active_warm'):run(c,m,f'warmup_r2_{c}_{m}')
# 5 blocks; every block contains all3 inputs x all4 modes in rotated/reversed order.
modes=['off','shadow','active_cold','active_warm']
for block in range(5):
    for c in range(3):
        order=modes[(block+c)%4:]+modes[:(block+c)%4]
        if block%2:order=order[::-1]
        for m in order:
            if m=='active_warm':
                # Untimed request guarantees this exact plan's replica is ready,
                # even when A/B rotation places warm before cold on a new input.
                run(c,'active_cold',f'prepare_warm_b{block}_c{c}')
            run(c,m,f'e2e_b{block}_c{c}_{m}')
# Diagnostics separated from E2E. Same fixed workload, no correctness hash overhead.
for block in range(5):
    order=modes[block%4:]+modes[:block%4]
    if block%2:order=order[::-1]
    for m in order:
        if m=='active_warm':run(0,'active_cold',f'prepare_profile_warm_b{block}')
        run(0,m,f'profile_b{block}_c0_{m}','--profile')
# Concurrent all16-rank live-input layer replay, 20 warmups / >=100ms blocks x5.
for c in range(3):run(c,'replay',f'replay_c{c}')
print('SUITE_COMPLETE',flush=True)

#!/usr/bin/env python3
"""Describe output repetition so random-ID acceptance is not mistaken for dialogue."""
import collections
import json
import sys
from pathlib import Path

root=Path(sys.argv[1])
rows=[]
for scene in ('decode','mixed','prefix'):
    for mode in ('off','on'):
        for path in sorted((root/scene/mode).glob('request_*/measurement/output_ids_final_or_partial.json')):
            ids=json.loads(path.read_text())
            histogram=collections.Counter(ids)
            period=None
            for step in range(1,min(65,len(ids))):
                if all(ids[i]==ids[i-step] for i in range(step,len(ids))):
                    period=step;break
            rows.append({'scene':scene,'mode':mode,'request':path.parent.parent.name,
                'output_count':len(ids),'distinct_ids':len(histogram),'full_sequence_period_up_to_64':period,
                'top5_ids':histogram.most_common(5),'first32_ids':ids[:32],
                'purpose':'Output shape audit; random token inputs are not real dialogue, ignore_eos=true.'})
out=root/'analysis';out.mkdir(exist_ok=True)
(out/'output_patterns.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
print('pattern audit:',len(rows),'saved request outputs')

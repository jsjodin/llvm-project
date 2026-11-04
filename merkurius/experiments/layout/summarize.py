# The median of each measurement over the runs, per layout, workload and
# size, with checks that every layout computed the same checksums.
import csv, statistics, sys
from collections import defaultdict
rows = list(csv.DictReader(open(sys.argv[1]), delimiter='\t'))
groups = defaultdict(list)
for r in rows:
    groups[(r['workload'], r['elements'], r['layout'])].append(r)
checks = defaultdict(set)
for r in rows:
    checks[(r['workload'], r['elements'])].add((r['check'], r['check_copy']))
fields = ['bytes_per_elem', 'build_ns', 'sum_ns', 'copy_ns', 'sum_copy_ns']
print('\t'.join(['workload', 'elements', 'layout'] + fields))
def order(k):
    return (k[0], int(k[1]) if k[1].isdigit() else 0, k[2])
for key in sorted(groups, key=order):
    vals = []
    for f in fields:
        xs = [float(r[f]) for r in groups[key] if r[f] not in ('', None)]
        vals.append('%.3f' % statistics.median(xs) if xs else '')
    print('\t'.join(list(key) + vals))
bad = [k for k, v in checks.items() if len(v) != 1]
if bad:
    print('checksums differ between layouts or runs:', bad, file=sys.stderr)
    sys.exit(1)

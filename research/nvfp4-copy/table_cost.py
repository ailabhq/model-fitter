"""Table error and bytes of Inkling's 40 NVFP4 expert tensors (layers 0-19), per candidate type.

Input: research/table-mix/results/tables/inkling-2-mini.tsv.gz from the research/table-mix branch
(columns: tensor, type, bytes, error A, error B, rows sampled, elements, has imatrix), e.g.
    git show origin/research/table-mix:research/table-mix/results/tables/inkling-2-mini.tsv.gz > ink.tsv.gz
"""
import collections
import gzip
import re
import sys

table = collections.defaultdict(dict)
for line in gzip.open(sys.argv[1], "rt"):
    r = line.rstrip("\n").split("\t")
    table[r[0]][r[1]] = (int(r[2]), float(r[3]), float(r[4]))

print("role     type     GiB     error A    error B")
for role in ("down", "gate_up"):
    names = [n for n in table if re.fullmatch(rf"blk\.(\d+)\.ffn_{role}_exps\.weight", n) and int(n.split(".")[1]) < 20]
    assert len(names) == 20
    for ty in ("nvfp4", "iq4_nl", "iq4_xs", "q4_0", "mxfp4", "q5_0", "q8_0", "iq3_s", "iq3_xxs"):
        if all(ty in table[n] for n in names):
            b, a, e = (sum(table[n][ty][i] for n in names) for i in range(3))
            print(f"{role:8s} {ty:7s} {b / 2**30:6.3f} {a:10.0f} {e:10.0f}")

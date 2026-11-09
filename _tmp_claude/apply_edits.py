import json, sys
nb_path, edits_path = sys.argv[1], sys.argv[2]
nb = json.load(open(nb_path, encoding="utf-8"))
edits = json.load(open(edits_path, encoding="utf-8"))
YC = "# YOUR CODE HERE\nraise NotImplementedError()"
TAG = "# YOUR CODE HERE (Daniela)\n"
for k, e in edits.items():
    cell = nb["cells"][int(k)]
    src = "".join(cell["source"])
    if isinstance(e, str):
        assert src.count(YC) == 1, (k, "YC")
        src = src.replace(YC, TAG + e)
    else:
        for old, new in e:
            assert src.count(old) == 1, (k, old[:50])
            src = src.replace(old, new)
    lines = src.split("\n")
    cell["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
left = [i for i, c in enumerate(nb["cells"]) if "NotImplementedError" in "".join(c["source"])]
assert not left, left
json.dump(nb, open(nb_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("ok, sin NotImplementedError")

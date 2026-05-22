import json

class C2CCoverage:
    """
    Exact paper implementation of c2c_cvg:

      c2c(ci, cj) = |ci ∩ cj| / max(|ci|, |cj|)
      simC(A, B)   = { ci ∈ A : ∃ cj ∈ B with c2c(ci,cj) ≥ th }
      c2c_cvg(A,B) = |simC(A,B)| / |A| * 100

    A = 'source' (first argument), B = 'target' (second argument).
    """

    def __init__(self, source, target, mode='array'):
        self.source = source  # A  (predicted or whichever you choose as numerator base)
        self.target = target  # B  (the other architecture)
        self.mode = mode

    # ---------- parsers (only used if mode='file') ----------
    def _parse_rsf(self, path):
        m = {}
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                p = line.split()
                if len(p) == 3 and p[0].lower() == 'contain':
                    m[p[2]] = p[1]   # entity -> cluster
        return m

    def _parse_json(self, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        mapping = {}
        def walk(node, group=None):
            if node.get("@type") == "group":
                g = node.get("name")
                for ch in (node.get("nested") or []):
                    walk(ch, g)
            elif node.get("@type") == "item":
                mapping[node.get("name")] = group
        for top in (data.get("structure") or []):
            walk(top, top.get("name"))
        return mapping

    # ---------- inputs -> aligned label arrays ----------
    def _prepare_labels(self):
        if self.mode == 'array':
            # supports (entities, labels) tuples on either side; else assumes aligned arrays
            def unpack(x):
                if isinstance(x, tuple) and len(x) == 2:
                    ents, labs = x
                    return list(map(str, ents)), list(labs)
                return None, list(x)

            eA, yA = unpack(self.source)
            eB, yB = unpack(self.target)

            if eA is not None and eB is not None:
                A = {e:l for e,l in zip(eA, yA)}
                B = {e:l for e,l in zip(eB, yB)}
                ents = [e for e in A if e in B]   # intersection (paper doesn’t pad)
                yA = [A[e] for e in ents]
                yB = [B[e] for e in ents]
                return yA, yB

            n = min(len(yA), len(yB))
            return yA[:n], yB[:n]

        elif self.mode == 'file':
            A = self._parse_json(self.source) if self.source.lower().endswith(".json") else self._parse_rsf(self.source)
            B = self._parse_json(self.target) if self.target.lower().endswith(".json") else self._parse_rsf(self.target)
            ents = [e for e in A if e in B]      # intersection (no padding)
            yA = [A[e] for e in ents]
            yB = [B[e] for e in ents]
            return yA, yB

        else:
            raise ValueError("mode must be 'array' or 'file'")

    # ---------- labels -> list[set(indices)] ----------
    def _clusters(self, labels):
        pos = {}
        for i, lab in enumerate(labels):
            pos.setdefault(lab, set()).add(i)
        return list(pos.values())

    # ---------- c2c and simC ----------
    @staticmethod
    def _c2c(a, b):
        return (len(a & b) / max(len(a), len(b))) if (a or b) else 0.0

    def _simC(self, A_clusters, B_clusters, threshold):
        # convert threshold percent to fraction if needed
        thr = threshold if threshold <= 1 else (threshold / 100.0)

        count = 0
        for ci in A_clusters:  # each cluster in A
            # check if this A-cluster overlaps enough with ANY cluster in B
            found_match = False
            for cj in B_clusters:
                if self._c2c(ci, cj) >= thr:
                    found_match = True
                    break
            if found_match:
                count += 1
        return count


    # ---------- final metric ----------
    def c2c_cvg(self, threshold=0.50):
        yA, yB = self._prepare_labels()
        A = self._clusters(yA)   # clusters of the FIRST architecture (denominator)
        B = self._clusters(yB)   # clusters of the SECOND architecture
        if not A:
            return 0.0
        simC = self._simC(A, B, threshold)
        return 100.0 * simC / len(A)

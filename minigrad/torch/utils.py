def topo_sort(root):
    visited = set()
    topo = []
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            if v._ctx:
                for child in v._ctx[2]:
                    build_topo(child)
            topo.append(v)
    build_topo(root)
    return topo
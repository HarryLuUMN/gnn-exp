from collections import deque

def subgraph_hoop_sampling(graphData, hubNode, hoopNum):
    """
    Perform k-hop subgraph sampling from a PyG-format graph dictionary.

    Args:
        graphData (dict): PyG-format graph dictionary containing:
            - x: [[...], [...], ...]     node features
            - edge_index: [[src...], [dst...]]
            - edge_attr:  [[...], [...], ...]  (optional)
            - y: [...] (optional)
            - batch: [...] (optional)
        hubNode (int): center node
        hoopNum (int): k-hop number

    Returns:
        dict: New PyG-format subgraph data with remapped indexing.
    """

    x = graphData["x"]
    edge_index = graphData["edge_index"]
    edge_attr = graphData.get("edge_attr")
    y = graphData.get("y")
    batch = graphData.get("batch")

    num_nodes = len(x)

    # --- 1. Build adjacency list ---
    adj = [[] for _ in range(num_nodes)]
    for s, t in zip(edge_index[0], edge_index[1]):
        adj[s].append(t)
        adj[t].append(s)  # assuming undirected graph

    # --- 2. BFS to get k-hop nodes ---
    visited = set([hubNode])
    queue = deque([(hubNode, 0)])

    while queue:
        node, dist = queue.popleft()
        if dist == hoopNum:
            continue

        for nei in adj[node]:
            if nei not in visited:
                visited.add(nei)
                queue.append((nei, dist + 1))

    # Final node list for subgraph
    sub_nodes = sorted(list(visited))

    # --- 3. Create old → new indexing map ---
    mapping = {old: new for new, old in enumerate(sub_nodes)}

    # --- 4. Filter edges ---
    new_edge_index = [[], []]
    new_edge_attr = []

    for i, (s, t) in enumerate(zip(edge_index[0], edge_index[1])):
        if s in mapping and t in mapping:
            new_edge_index[0].append(mapping[s])
            new_edge_index[1].append(mapping[t])

            if edge_attr is not None:
                new_edge_attr.append(edge_attr[i])

    # --- 5. Slice node features ---
    new_x = [x[old] for old in sub_nodes]

    # --- 6. Slice node labels if exist ---
    new_y = [y[old] for old in sub_nodes] if y is not None else None

    # --- 7. Slice batch if exist ---
    new_batch = [batch[old] for old in sub_nodes] if batch is not None else None

    # --- 8. Package result ---
    out = {
        "x": new_x,
        "edge_index": new_edge_index,
    }

    if edge_attr is not None:
        out["edge_attr"] = new_edge_attr
    if new_y is not None:
        out["y"] = new_y
    if new_batch is not None:
        out["batch"] = new_batch

    return out



def multiple_subgraph_hoop_sampling(graphData, hubNodes, hoopNum):
    """
    Perform k-hop subgraph sampling for multiple hub nodes and merge into
    a single PyG-format subgraph.

    Args:
        graphData (dict): PyG-format graph dict
        hubNodes (list[int]): center nodes for sampling
        hoopNum (int): k-hop number

    Returns:
        dict: merged PyG-format subgraph
    """

    x = graphData["x"]
    edge_index = graphData["edge_index"]
    edge_attr = graphData.get("edge_attr")
    y = graphData.get("y")
    batch = graphData.get("batch")

    num_nodes = len(x)

    # --- Build adjacency list ---
    adj = [[] for _ in range(num_nodes)]
    for s, t in zip(edge_index[0], edge_index[1]):
        adj[s].append(t)
        adj[t].append(s)

    # --- Union of all k-hop sampled nodes ---
    all_nodes = set()

    for hubNode in hubNodes:
        visited = set([hubNode])
        queue = deque([(hubNode, 0)])

        while queue:
            node, d = queue.popleft()
            if d == hoopNum:
                continue

            for nei in adj[node]:
                if nei not in visited:
                    visited.add(nei)
                    queue.append((nei, d + 1))

        all_nodes |= visited   # union

    # Sort final node list
    sub_nodes = sorted(list(all_nodes))

    # --- Create old → new index map ---
    mapping = {old: new for new, old in enumerate(sub_nodes)}

    # --- Filter edges ---
    new_edge_index = [[], []]
    new_edge_attr = []

    for i, (s, t) in enumerate(zip(edge_index[0], edge_index[1])):
        if s in mapping and t in mapping:
            new_edge_index[0].append(mapping[s])
            new_edge_index[1].append(mapping[t])
            if edge_attr is not None:
                new_edge_attr.append(edge_attr[i])

    # --- Slice node-level data ---
    new_x = [x[i] for i in sub_nodes]
    new_y = [y[i] for i in sub_nodes] if y is not None else None
    new_batch = [batch[i] for i in sub_nodes] if batch is not None else None

    # --- Build output ---
    out = {
        "x": new_x,
        "edge_index": new_edge_index,
    }

    if edge_attr is not None:
        out["edge_attr"] = new_edge_attr
    if new_y is not None:
        out["y"] = new_y
    if new_batch is not None:
        out["batch"] = new_batch

    return out

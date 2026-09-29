import pickle
from pathlib import Path


def save_nodes(
    nodes,
    strategy,
    path="vectorstore/nodes.pkl",
):
    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "strategy": strategy,
        "nodes": nodes,
    }

    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_nodes(path="vectorstore/nodes.pkl"):
    with open(path, "rb") as f:
        payload = pickle.load(f)

    return payload
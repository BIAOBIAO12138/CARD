import collections
from typing import Dict, List, Optional


class TrieNode:
    __slots__ = ("children", "is_end")

    def __init__(self):
        self.children: Dict[int, "TrieNode"] = {}
        self.is_end: bool = False


def build_prefix_tree(sequences: List[List[int]]) -> TrieNode:
    root = TrieNode()
    for seq in sequences:
        node = root
        for tok in seq:
            t = int(tok)
            if t not in node.children:
                node.children[t] = TrieNode()
            node = node.children[t]
        node.is_end = True
    return root


def make_prefix_allowed_tokens_fn(root: TrieNode, pad_token_id: int = 0):
    root_children = list(root.children.keys()) or [pad_token_id]

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        if input_ids.numel() == 0:
            return root_children

        prefix = [int(t) for t in input_ids.tolist() if int(t) != pad_token_id]
        if not prefix:
            return root_children

        node: Optional[TrieNode] = root
        for tok in prefix:
            child = node.children.get(tok)
            if child is None:
                return root_children
            node = child

        if node.children:
            return list(node.children.keys())
        return [pad_token_id]

    return prefix_allowed_tokens_fn

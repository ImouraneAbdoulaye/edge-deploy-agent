"""Build the datasheet retrieval index (text extraction + chunking + embeddings).

    python scripts/build_index.py
    python scripts/build_index.py --query "flash memory size" --chip STM32L476RG
"""

from __future__ import annotations

import argparse

from edge_deploy_agent.rag.index import DatasheetIndex, build_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", help="Only search an existing index")
    parser.add_argument("--chip")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    if args.query is None:
        n = build_index()
        print(f"Indexed {n} chunks.")
        return

    for hit in DatasheetIndex().search(args.query, args.chip, args.k):
        print(f"\n--- {hit.doc} p.{hit.page} (score {hit.score:.4f})\n{hit.text[:600]}")


if __name__ == "__main__":
    main()

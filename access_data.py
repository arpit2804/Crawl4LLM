import logging
import sys
import os
import json
from corpus_interface import ClueWeb22Api, UnifiedGetter

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if len(sys.argv) < 3:
        print("Usage: python access_data.py <cw22_root_path> <docid>")
        return

    cw22_root = sys.argv[1]
    docid = sys.argv[2]

    if not os.path.exists(cw22_root):
        print(f"Root path does not exist: {cw22_root}")
        return

    cw22 = UnifiedGetter(ClueWeb22Api(cw22_root), docid_pos=0)
    print(f"Initialized API with root: {cw22_root}")

    doc_content = cw22.get_doc(docid)
    if doc_content:
        print("\n[DOCUMENT CONTENT]\n")
        print(doc_content)
    else:
        print(f"Document not found: {docid}")

    outlinks = cw22.get_outlinks(docid)
    print("\n[OUTLINKS]\n")
    print(outlinks)

    for outlink in outlinks:
        doc = cw22.get_doc(outlink)
        if doc:
            print(f"\n[OUTLINK DOCUMENT FOUND] {outlink}\n")
            print(doc)
        else:
            print(f"\n[OUTLINK DOCUMENT NOT FOUND] {outlink}")


if __name__ == "__main__":
    main()
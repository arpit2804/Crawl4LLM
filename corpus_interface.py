import gzip
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)


class DocumentAnnotation(dict):
    _compare_key: str | None = None
    _order: str = "desc"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def set_compare_method(cls, key: str, order: Literal["desc", "asc"]) -> None:
        cls._compare_key = key
        if order not in ["desc", "asc"]:
            raise ValueError("Order must be either 'desc' or 'asc'")
        cls._order = order

    def __lt__(self, other) -> bool:
        if self._compare_key is None:
            raise ValueError("Compare key not set")
        return (self[self._compare_key] > other[self._compare_key]) if self._order == "desc" else (self[self._compare_key] < other[self._compare_key])


@dataclass
class Document:
    docid: str
    text: str | None = None
    annotations: DocumentAnnotation = field(default_factory=DocumentAnnotation)


class ClueWeb22Api:
    def __init__(self, cw22root_path) -> None:
        self.cw22root_path = cw22root_path

    def get_base_filename_by_id(self, cw22id: str, file_type: str = "html") -> str:
        id_parts = cw22id.split("-")
        language = id_parts[1][:2]
        segment = id_parts[1][:4]
        directory = id_parts[1]
        base_path = os.path.join(self.cw22root_path, file_type, language, segment, directory)
        base_filename = os.path.join(base_path, f"{directory}-{id_parts[2]}")
        return base_filename

    def get_json_record(self, cw22id: str, record_type: str) -> str:
        id_parts = cw22id.split("-")
        language = id_parts[1][:2]  # "en"
        segment = id_parts[1][:4]   # "en00"
        directory = id_parts[1]     # "en0000"

        # Construct full path to the single large JSON file
        json_path = os.path.join(
            self.cw22root_path,
            record_type,
            language,
            segment,
            directory,
            f"{directory}-00000.json.gz"
        )
        print(json_path)
        with gzip.open(json_path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    records = json.loads(line)
                    for record in records:
                        if record.get("ClueWeb22-ID") == cw22id:
                            return json.dumps(record)
                except json.JSONDecodeError:
                    continue

        raise FileNotFoundError(f"Document not found: {cw22id}")



    def get_clean_text(self, cw22id: str) -> str:
        return self.get_json_record(cw22id, "txt")

    def get_inlinks(self, cw22id: str) -> str:
        return self.get_json_record(cw22id, "inlink")

    def get_outlinks(self, cw22id: str) -> str:
        return self.get_json_record(cw22id, "outlink")


class UnifiedGetter:
    def __init__(self, cw22_api: ClueWeb22Api, docid_pos: int = 0) -> None:
        self.cw22_api = cw22_api
        self.docid_pos = docid_pos

    def get_doc(self, docid: str) -> Document | None:
        try:
            cw22_data = json.loads(self.cw22_api.get_clean_text(docid))
            return Document(docid=docid, text=cw22_data.get("Clean-Text"))
        except Exception as e:
            logger.debug(f"Failed to get doc: {docid}, error: {e}")
            return None

    def get_outlinks(self, docid: str) -> list[str]:
        try:
            obj = json.loads(self.cw22_api.get_outlinks(docid))
            if isinstance(obj, dict):
                outlinks = obj.get("outlinks", [])
            elif isinstance(obj, list):
                outlinks = obj
            else:
                outlinks = []

            return [
                x[self.docid_pos]
                for x in outlinks
                if x and x[self.docid_pos].startswith("clueweb22-en0")
            ]
        except Exception as e:
            logger.info(f"Failed to get outlinks for {docid}, error: {e}")
            return []


    def get_inlinks(self, docid: str) -> list[str]:
        try:
            obj = json.loads(self.cw22_api.get_inlinks(docid))
            anchors = obj.get("anchors", []) if isinstance(obj, dict) else obj
            return [
                x[self.docid_pos]
                for x in anchors
                if x and x[self.docid_pos].startswith("clueweb22-en0")
            ]
        except Exception as e:
            logger.debug(f"Failed to get inlinks for {docid}, error: {e}")
            return []


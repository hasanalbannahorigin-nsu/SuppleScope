"""Build the local iDISK2.0 SQLite database used by the research agent and patient-risk step.

1. Download the iDISK2.0 CSV files (github.com/houyurain/iDISK2.0, folder Data/) to e.g. D:\\CSE499\\iDISK2.0\\Data
   (check the data licence first; do not commit the data to the repository).
2. See how the loader reads the columns:   python scripts/load_idisk.py inspect D:\\CSE499\\iDISK2.0\\Data
3. Build the database:                     python scripts/load_idisk.py load D:\\CSE499\\iDISK2.0\\Data
   (writes IDISK_DB_PATH, default backend/data/knowledge/idisk.sqlite)
4. Try a lookup:                           python scripts/load_idisk.py lookup "fenugreek"
If a column is guessed wrong, write a JSON file such as
  {"dsi.csv": {"name_col": "Preferred_Name", "synonym_cols": ["Synonyms"]}}
and pass it with --map mapping.json. iDISK 1.x RRF folders (MRCONSO.RRF ...) are read too.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.services.knowledge.idisk import IdiskStore, build_database, discover  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["inspect", "load", "lookup"])
    ap.add_argument("target", help="folder (inspect/load) or ingredient name (lookup)")
    ap.add_argument("--db", default=get_settings().idisk_db_path)
    ap.add_argument("--map", default=None)
    args = ap.parse_args()
    overrides = json.loads(Path(args.map).read_text(encoding="utf-8")) if args.map else None
    if args.command == "inspect":
        for m in discover(Path(args.target), overrides):
            print(f"\n{Path(m.path).name}: {m.kind}")
            print(f"  headers: {m.headers}")
            if m.kind == "entities":
                print(f"  type={m.entity_type} id={m.id_col} name={m.name_col} synonyms={m.synonym_cols} text={m.text_cols}")
            else:
                print(f"  start={m.start_col} relation={m.rel_col} end={m.end_col}")
    elif args.command == "load":
        counts = build_database(Path(args.target), Path(args.db), overrides)
        print(f"built {args.db}: {counts}")
    else:
        store = IdiskStore(Path(args.db))
        m = store.match([args.target])
        if not m:
            print("no match")
            return
        print(f"{m.entity_id} {m.name} ({m.entity_type}) score {m.score} via '{m.matched_name}'")
        for f in store.facts(m.entity_id)[:30]:
            print(f"  [{f.kind}] {f.text}")


if __name__ == "__main__":
    main()

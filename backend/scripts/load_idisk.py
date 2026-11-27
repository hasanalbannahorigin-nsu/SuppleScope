"""Build the local iDISK2.0 SQLite database used by the research agent and patient-risk step.

1. The iDISK2.0 repository (github.com/houyurain/iDISK2.0, Apache-2.0) holds only README files. Download the
   two Google Drive folders linked in its Data/Entity/README.md and Data/Relation/README.md, and unzip them, e.g.
   to D:\\CSE499\\iDISK2.0\\Data\\Entity (DSI.csv, DSP.csv, D.csv, Dis.csv, SS.csv) and
   D:\\CSE499\\iDISK2.0\\Data\\Relation (dsp_dsi.csv, dsi_dis.csv, dsi_d.csv, dsi_ss.csv).
   Do not commit the data to the repository.
2. See how the loader reads the columns:   python scripts/load_idisk.py inspect D:\\CSE499\\iDISK2.0\\Data
   (subfolders are searched; an empty or zipped folder is explained)
3. Build the database:                     python scripts/load_idisk.py load D:\\CSE499\\iDISK2.0\\Data
   (writes IDISK_DB_PATH, default backend/data/knowledge/idisk.sqlite)
4. Try a lookup:                           python scripts/load_idisk.py lookup "fenugreek"
   (iDISK2.0 names herbs by their scientific name; the Bangla-Latin dictionary adds it, as in the agent)
If a column is guessed wrong, write a JSON file such as
  {"DSI.csv": {"name_col": "Name", "synonym_cols": []}}
and pass it with --map mapping.json. iDISK 1.x RRF folders (MRCONSO.RRF ...) are read too.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.services.knowledge.idisk import IdiskStore, build_database, discover, folder_report, rrf_folder  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["inspect", "load", "lookup"])
    ap.add_argument("target", help="folder (inspect/load) or ingredient name (lookup)")
    ap.add_argument("--db", default=get_settings().idisk_db_path)
    ap.add_argument("--map", default=None)
    args = ap.parse_args()
    overrides = json.loads(Path(args.map).read_text(encoding="utf-8")) if args.map else None
    if args.command in ("inspect", "load"):
        folder = Path(args.target)
        if not discover(folder, overrides) and rrf_folder(folder) is None:
            print("No iDISK data files found.")
            print("\n".join(folder_report(folder)))
            sys.exit(1)
    if args.command == "inspect":
        if rrf_folder(Path(args.target)) is not None:
            print(f"iDISK 1.x RRF files in {rrf_folder(Path(args.target))}")
        for m in discover(Path(args.target), overrides):
            print(f"\n{Path(m.path).relative_to(Path(args.target))}: {m.kind}")
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
        from app.services.knowledge.normalise import default_normaliser
        nz = default_normaliser().normalise(args.target)  # adds the scientific name iDISK2.0 uses for herbs
        m = store.match([args.target, *(nz.search_names if nz.entry_id else [])])
        if not m:
            print("no match")
            return
        print(f"{m.entity_id} {m.name} ({m.entity_type}) score {m.score} via '{m.matched_name}'")
        for f in store.facts(m.entity_id)[:30]:
            print(f"  [{f.kind}] {f.text}")


if __name__ == "__main__":
    main()

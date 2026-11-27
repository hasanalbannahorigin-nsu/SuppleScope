"""iDISK2.0 (integrated Dietary Supplement Knowledge base, Hou et al., JMIR 2025) in a local SQLite file.

iDISK2.0 is published as CSV files (dsi.csv ingredients, dsp.csv products, disease.csv, drug.csv,
symptom.csv, relationships.csv) at github.com/houyurain/iDISK2.0. Their exact column names are not
documented in the README, so the loader guesses the id / name / synonym / relation columns from the
headers and prints its guess (`python scripts/load_idisk.py inspect <folder>`); a JSON mapping file
can override any guess. The older iDISK 1.x RRF release (MRCONSO/MRREL/MRSTY/MRSAT, pipe-delimited)
is also read when those files are present.

The data are not stored in the repository. Check the data licence before sharing the database: the
1.x files were withdrawn in April 2025 because the licence of a third-party source (Natural Medicines)
expired, and iDISK2.0 integrates several sources.

SQLite was chosen over Neo4j (used by the paper) because the agent needs one-hop lookups only:
ingredient -> drugs it interacts with, adverse effects, diseases, products. That keeps the stack free
and file-based on a Windows laptop.
"""
from __future__ import annotations

import csv
import json
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

from app.services.knowledge.normalise import name_key

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

TYPE_BY_STEM = {"dsi": "DSI", "sdsi": "DSI", "dsp": "DSP", "disease": "Disease", "dis": "Disease", "drug": "Drug",
                "spd": "Drug", "symptom": "Symptom", "ss": "Symptom", "soc": "SOC", "tc": "TC"}
ID_COLS = ("id:id", ":id", "id", "cui", "entity_id", "node_id", "concept_id", "dsi_id", "dsp_id", "disease_id", "drug_id",
           "symptom_id", "idisk_id", "dsi_cui", "dsp_cui")
NAME_COLS = ("name", "preferred_name", "pref_name", "preferred_term", "term", "str", "title", "ingredient_name",
             "product_name", "dsi_name", "dsp_name", "disease_name", "drug_name", "symptom_name", "common_name")
SYN_HINTS = ("synonym", "alias", "other_name", "common_name", "scientific_name", "latin_name", "names")
START_COLS = (":start_id", "start_id", "start", "source", "source_id", "head", "head_id", "subject", "subject_id", "from",
              "cui1", "src", "node1", "entity1")
END_COLS = (":end_id", "end_id", "end", "target", "target_id", "tail", "tail_id", "object", "object_id", "to", "cui2",
            "dst", "node2", "entity2")
REL_COLS = (":type", "type", "relation", "relation_type", "relationship", "rel", "predicate", "edge_type", "label")
TEXT_HINTS = ("safety", "warning", "caution", "adverse", "interaction", "mechanism", "background", "description",
              "effect", "dose", "dosage", "pregnan", "toxicity", "contraindication")

RELATION_KIND = [
    (re.compile(r"interact", re.I), "interaction"),
    (re.compile(r"adverse|side.?effect|reaction", re.I), "adverse_effect"),
    (re.compile(r"contraindicat|caution|avoid", re.I), "caution"),
    (re.compile(r"effective|treat|indicat|use[sd]?_for|for_disease|alleviat|improve", re.I), "effective_for"),
    (re.compile(r"ingredient|contain", re.I), "has_ingredient"),
    (re.compile(r"therapeutic|class", re.I), "therapeutic_class"),
]


def relation_kind(relation: str, other_type: str | None = None) -> str:
    for pat, kind in RELATION_KIND:
        if pat.search(relation or ""):
            return kind
    if other_type == "Drug":
        return "interaction"
    return "other"


def _norm(h: str) -> str:
    return re.sub(r"\s+", "_", (h or "").strip().lower())


def _pick(headers: list[str], wanted: tuple[str, ...], *, contains: bool = False) -> str | None:
    norm = {_norm(h): h for h in headers}
    for w in wanted:
        if w in norm:
            return norm[w]
    if contains:
        for w in wanted:
            for n, h in norm.items():
                if w in n:
                    return h
    return None


def _sniff(path: Path) -> csv.Dialect | str:
    sample = path.open(encoding="utf-8-sig", errors="replace").read(20000)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t|;")
    except csv.Error:
        return "excel"


@dataclass
class FileMap:
    path: str
    kind: str  # entities | relations
    entity_type: str | None = None
    id_col: str | None = None
    name_col: str | None = None
    synonym_cols: list[str] = field(default_factory=list)
    text_cols: list[str] = field(default_factory=list)
    start_col: str | None = None
    end_col: str | None = None
    rel_col: str | None = None
    headers: list[str] = field(default_factory=list)


def guess_map(path: Path) -> FileMap:
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as fh:
        reader = csv.reader(fh, _sniff(path))
        headers = next(reader, [])
    stem = path.stem.lower()
    start, end = _pick(headers, START_COLS), _pick(headers, END_COLS)
    if "relation" in stem or "edge" in stem or (start and end):
        return FileMap(path=str(path), kind="relations", headers=headers,
                       start_col=start or _pick(headers, START_COLS, contains=True),
                       end_col=end or _pick(headers, END_COLS, contains=True),
                       rel_col=_pick(headers, REL_COLS) or _pick(headers, REL_COLS, contains=True))
    etype = TYPE_BY_STEM.get(stem) or next((v for k, v in TYPE_BY_STEM.items() if stem.startswith(k)), stem.upper())
    id_col = _pick(headers, ID_COLS) or _pick(headers, ("_id", "cui"), contains=True) or (headers[0] if headers else None)
    name_col = _pick(headers, NAME_COLS) or _pick(headers, ("name", "term"), contains=True)
    syn = [h for h in headers if h not in (id_col, name_col) and any(x in _norm(h) for x in SYN_HINTS)]
    text = [h for h in headers if h not in (id_col, name_col, *syn) and any(x in _norm(h) for x in TEXT_HINTS)]
    return FileMap(path=str(path), kind="entities", entity_type=etype, id_col=id_col, name_col=name_col,
                   synonym_cols=syn, text_cols=text, headers=headers)


LFS_POINTER = b"version https://git-lfs.github.com/spec/"


def _is_lfs_pointer(path: Path) -> bool:
    """A Git LFS placeholder (a few lines of text) instead of the real data file."""
    try:
        with path.open("rb") as fh:
            return fh.read(len(LFS_POINTER)) == LFS_POINTER
    except OSError:
        return False


def data_files(folder: Path) -> list[Path]:
    """CSV/TSV files in the folder and its subfolders, without Git LFS placeholders."""
    files = sorted(folder.rglob("*.csv")) + sorted(folder.rglob("*.tsv"))
    return [p for p in files if not _is_lfs_pointer(p)]


def rrf_folder(folder: Path) -> Path | None:
    """Folder that holds MRCONSO.RRF (iDISK 1.x), at the top or in a subfolder."""
    if (folder / "MRCONSO.RRF").exists():
        return folder
    hit = next(iter(sorted(folder.rglob("MRCONSO.RRF"))), None)
    return hit.parent if hit else None


def folder_report(folder: Path, limit: int = 25) -> list[str]:
    """Plain-language lines on why a folder gave no data files, and what to do."""
    if not folder.exists():
        return [f"The folder {folder} does not exist. Check the path (the clone may have made another folder name)."]
    files = [p for p in sorted(folder.rglob("*")) if p.is_file() and ".git" not in p.parts]
    if not files:
        return [f"The folder {folder} has no files."]
    lines = []
    zips = [p for p in files if p.suffix.lower() in (".zip", ".gz", ".7z", ".rar", ".tar")]
    pointers = [p for p in files if p.suffix.lower() in (".csv", ".tsv", ".rrf") and _is_lfs_pointer(p)]
    if zips:
        lines.append("Compressed files found; unpack them, then run inspect on the unpacked folder:")
        lines += [f"  Expand-Archive \"{z}\" -DestinationPath \"{z.with_suffix('')}\"" if z.suffix.lower() == ".zip"
                  else f"  {z} (unpack with 7-Zip)" for z in zips[:10]]
    if pointers:
        lines.append(f"{len(pointers)} data file(s) are Git LFS placeholders, not the data. In the iDISK2.0 folder run:")
        lines += ["  git lfs install", "  git lfs pull"]
    if not lines:
        lines.append("No CSV, TSV or RRF data files were found. The data may be published outside the repository "
                     "(see its README for a download link).")
    lines.append(f"Files in {folder} (first {min(limit, len(files))} of {len(files)}):")
    lines += [f"  {p.stat().st_size:>12,d}  {p.relative_to(folder)}" for p in files[:limit]]
    return lines


def discover(folder: Path, overrides: dict | None = None) -> list[FileMap]:
    maps = []
    for p in data_files(folder):
        m = guess_map(p)
        for k, v in (overrides or {}).get(p.name, {}).items():
            setattr(m, k, v)
        maps.append(m)
    return maps


SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (id TEXT PRIMARY KEY, type TEXT, name TEXT, attrs TEXT);
CREATE TABLE IF NOT EXISTS names (entity_id TEXT, name TEXT, key TEXT);
CREATE TABLE IF NOT EXISTS relations (subject TEXT, relation TEXT, object TEXT, attrs TEXT);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
CREATE INDEX IF NOT EXISTS ix_names_key ON names(key);
CREATE INDEX IF NOT EXISTS ix_rel_s ON relations(subject);
CREATE INDEX IF NOT EXISTS ix_rel_o ON relations(object);
"""


def _split_names(value: str) -> list[str]:
    value = (value or "").strip()
    if value.startswith("[") and value.endswith("]"):
        try:
            return [str(x).strip() for x in json.loads(value) if str(x).strip()]
        except ValueError:
            value = value.strip("[]")
    return [x.strip(" '\"") for x in re.split(r"\s*[|;]\s*", value) if x.strip(" '\"")]


def build_database(folder: str | Path, db_path: str | Path, overrides: dict | None = None) -> dict:
    """Load every CSV (iDISK2.0) or the RRF files (iDISK 1.x) of `folder` into SQLite. Returns counts."""
    folder, db_path = Path(folder), Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    counts = {"entities": 0, "names": 0, "relations": 0}
    rrf = rrf_folder(folder)
    if rrf is not None:
        _load_rrf(con, rrf, counts)
        con.execute("INSERT INTO meta VALUES ('format', 'iDISK 1.x RRF')")
    else:
        maps = discover(folder, overrides)
        for m in [x for x in maps if x.kind == "entities"]:
            _load_entities(con, m, counts)
        for m in [x for x in maps if x.kind == "relations"]:
            _load_relations(con, m, counts)
        con.execute("INSERT INTO meta VALUES ('format', 'iDISK2.0 CSV')")
        con.execute("INSERT INTO meta VALUES ('mapping', ?)", (json.dumps([m.__dict__ for m in maps], ensure_ascii=False),))
    con.execute("INSERT INTO meta VALUES ('source_folder', ?)", (str(folder),))
    con.commit()
    con.close()
    return counts


def _load_entities(con: sqlite3.Connection, m: FileMap, counts: dict) -> None:
    if not m.id_col:
        return
    path = Path(m.path)
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh, dialect=_sniff(path)):
            eid = (row.get(m.id_col) or "").strip()
            if not eid:
                continue
            name = (row.get(m.name_col) or "").strip() if m.name_col else ""
            attrs = {k: v for k, v in row.items() if k and v and k not in (m.id_col, m.name_col) and len(v) < 20000}
            con.execute("INSERT OR REPLACE INTO entities VALUES (?, ?, ?, ?)",
                        (eid, m.entity_type, name or eid, json.dumps(attrs, ensure_ascii=False)))
            counts["entities"] += 1
            names = [name] + [n for c in m.synonym_cols for n in _split_names(row.get(c) or "")]
            for n in dict.fromkeys(x for x in names if x):
                con.execute("INSERT INTO names VALUES (?, ?, ?)", (eid, n, name_key(n)))
                counts["names"] += 1


def _load_relations(con: sqlite3.Connection, m: FileMap, counts: dict) -> None:
    if not (m.start_col and m.end_col):
        return
    path = Path(m.path)
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh, dialect=_sniff(path)):
            s, o = (row.get(m.start_col) or "").strip(), (row.get(m.end_col) or "").strip()
            if not s or not o:
                continue
            rel = (row.get(m.rel_col) or "related_to").strip() if m.rel_col else "related_to"
            attrs = {k: v for k, v in row.items() if k and v and k not in (m.start_col, m.end_col, m.rel_col)}
            con.execute("INSERT INTO relations VALUES (?, ?, ?, ?)", (s, rel, o, json.dumps(attrs, ensure_ascii=False)))
            counts["relations"] += 1


def _rrf_rows(path: Path):
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("|")
            if parts and parts[-1] == "":
                parts = parts[:-1]
            yield parts


def _load_rrf(con: sqlite3.Connection, folder: Path, counts: dict) -> None:
    """iDISK 1.x: MRSTY (CUI|STY), MRCONSO (CUI|AUI|STR|TTY|SAB|SCODE|ISPREF), MRREL (RUI|CUI1|REL|CUI2|SAB),
    MRSAT (ATUI|UI|STYPE|ATN|ATV|SAB)."""
    types = {r[0]: r[1] for r in _rrf_rows(folder / "MRSTY.RRF") if len(r) >= 2} if (folder / "MRSTY.RRF").exists() else {}
    pref: dict[str, str] = {}
    for r in _rrf_rows(folder / "MRCONSO.RRF"):
        if len(r) < 3:
            continue
        cui, name = r[0], r[2]
        if cui not in pref or (len(r) > 6 and r[6] == "Y"):
            pref[cui] = name
        con.execute("INSERT INTO names VALUES (?, ?, ?)", (cui, name, name_key(name)))
        counts["names"] += 1
    attrs: dict[str, dict] = {}
    if (folder / "MRSAT.RRF").exists():
        for r in _rrf_rows(folder / "MRSAT.RRF"):
            if len(r) >= 5 and r[2] == "DSCUI":
                attrs.setdefault(r[1], {})[r[3]] = r[4]
    for cui, name in pref.items():
        stype = {"SDSI": "DSI", "DIS": "Disease", "SPD": "Drug", "SS": "Symptom"}.get(types.get(cui, ""), types.get(cui))
        con.execute("INSERT OR REPLACE INTO entities VALUES (?, ?, ?, ?)",
                    (cui, stype, name, json.dumps(attrs.get(cui, {}), ensure_ascii=False)))
        counts["entities"] += 1
    if (folder / "MRREL.RRF").exists():
        for r in _rrf_rows(folder / "MRREL.RRF"):
            if len(r) >= 4:
                con.execute("INSERT INTO relations VALUES (?, ?, ?, ?)", (r[1], r[2], r[3], json.dumps({"rui": r[0]})))
                counts["relations"] += 1


# ---- lookups ------------------------------------------------------------------------------------------------

@dataclass
class IdiskMatch:
    entity_id: str
    name: str
    entity_type: str | None
    score: float
    matched_name: str


@dataclass
class IdiskFact:
    kind: str  # interaction | adverse_effect | caution | effective_for | has_ingredient | therapeutic_class | other | attribute
    relation: str
    value: str
    other_id: str | None
    other_type: str | None
    text: str  # sentence used as the quoted passage


class IdiskStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.con = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, check_same_thread=False)
        self._names: list[tuple[str, str, str]] | None = None

    @classmethod
    def open_default(cls) -> IdiskStore | None:
        from app.config import ROOT_DIR, get_settings

        p = Path(get_settings().idisk_db_path)
        for c in ([p] if p.is_absolute() else [Path.cwd() / p, ROOT_DIR / "backend" / p]):
            if c.exists():
                return cls(c)
        return None

    def _ingredient_names(self) -> list[tuple[str, str, str]]:
        if self._names is None:
            self._names = self.con.execute(
                "SELECT n.key, n.entity_id, n.name FROM names n JOIN entities e ON e.id = n.entity_id "
                "WHERE e.type IN ('DSI', 'SDSI') OR e.type IS NULL").fetchall()
        return self._names

    def match(self, names: list[str], threshold: float = 90.0) -> IdiskMatch | None:
        """Best ingredient (DSI) for any of `names`: exact key first, then RapidFuzz."""
        names = [n for n in names if n and n.strip()]
        for n in names:
            row = self.con.execute(
                "SELECT n.entity_id, e.name, e.type, n.name FROM names n JOIN entities e ON e.id = n.entity_id "
                "WHERE n.key = ? ORDER BY CASE WHEN e.type IN ('DSI','SDSI') THEN 0 ELSE 1 END LIMIT 1",
                (name_key(n),)).fetchone()
            if row:
                return IdiskMatch(row[0], row[1], row[2], 100.0, row[3])
        try:
            from rapidfuzz import fuzz, process
        except ImportError:
            return None
        pool = self._ingredient_names()
        keys = [k for k, _, _ in pool]
        best = None
        for n in names:
            k = name_key(n)
            if len(k) < 4:
                continue
            hit = process.extractOne(k, keys, scorer=fuzz.ratio, score_cutoff=threshold)
            if hit and (best is None or hit[1] > best[1]):
                best = hit
        if not best:
            return None
        _, eid, matched = pool[best[2]]
        row = self.con.execute("SELECT name, type FROM entities WHERE id = ?", (eid,)).fetchone()
        return IdiskMatch(eid, row[0] if row else matched, row[1] if row else None, float(best[1]), matched)

    def facts(self, entity_id: str, limit: int = 80) -> list[IdiskFact]:
        name_row = self.con.execute("SELECT name, attrs FROM entities WHERE id = ?", (entity_id,)).fetchone()
        subject = name_row[0] if name_row else entity_id
        out: list[IdiskFact] = []
        rows = self.con.execute(
            "SELECT r.relation, r.object, e.name, e.type, r.attrs FROM relations r LEFT JOIN entities e ON e.id = r.object "
            "WHERE r.subject = ? UNION ALL "
            "SELECT r.relation, r.subject, e.name, e.type, r.attrs FROM relations r LEFT JOIN entities e ON e.id = r.subject "
            "WHERE r.object = ? LIMIT ?", (entity_id, entity_id, limit)).fetchall()
        for rel, other, other_name, other_type, attrs in rows:
            kind = relation_kind(rel, other_type)
            if kind == "has_ingredient":
                continue  # product -> ingredient edges are not facts about the ingredient
            value = other_name or other
            extra = ""
            try:
                a = json.loads(attrs or "{}")
                extra = "; ".join(f"{k}: {v}" for k, v in a.items() if isinstance(v, str) and len(v) < 300 and k != "rui")
            except ValueError:
                pass
            text = f"iDISK2.0: {subject} {rel} {value}" + (f" ({extra})" if extra else "") + "."
            out.append(IdiskFact(kind=kind, relation=rel, value=value, other_id=other, other_type=other_type, text=text))
        if name_row and name_row[1]:
            try:
                attrs = json.loads(name_row[1])
            except ValueError:
                attrs = {}
            for k, v in attrs.items():
                if isinstance(v, str) and v.strip() and any(h in _norm(k) for h in TEXT_HINTS):
                    out.append(IdiskFact(kind="attribute", relation=k, value=v.strip()[:1500], other_id=None,
                                         other_type=None, text=f"iDISK2.0 {k} of {subject}: {v.strip()[:1500]}"))
        return out

    def stats(self) -> dict:
        q = lambda sql: self.con.execute(sql).fetchone()[0]  # noqa: E731
        return {"entities": q("SELECT COUNT(*) FROM entities"), "relations": q("SELECT COUNT(*) FROM relations"),
                "names": q("SELECT COUNT(*) FROM names"),
                "format": (self.con.execute("SELECT v FROM meta WHERE k='format'").fetchone() or [None])[0]}

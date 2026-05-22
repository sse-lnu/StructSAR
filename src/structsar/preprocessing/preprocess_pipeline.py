"""Clean and validate architecture-recovery preprocessing outputs.

This script replaces the notebook-style preprocessing with a repeatable CLI.
By default it reads the current ``processed`` CSV files, normalizes labels,
writes a clean copy under ``Clean_Data``, and exports normalized ground-truth
labels under ``GT``.  It also contains raw SARIF/Depends extraction helpers so
new Java, C, C++, and Python systems can be rebuilt from ``*-file.pkl`` or
``*-file.json`` inputs when needed.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import pickle
import re
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


DATASETS: dict[str, dict[str, Any]] = {
    "ant": {"raw_dir": "Ant", "language": "java", "gt": "ant_gt.txt", "gt_type": "regex", "exclude_test_files": True},
    "argouml": {"raw_dir": "ArgoUML", "language": "java", "gt": "argouml_gt.txt", "gt_type": "regex"},
    "archstudio": {"raw_dir": "AS4", "language": "java", "gt": "archstudio_gt.json", "gt_type": "json"},
    "bash": {"raw_dir": "bash", "language": "c", "gt": "bash_gt.json", "gt_type": "json", "collapse_multi_by_path": True},
    "chromium": {"raw_dir": "chromium", "language": "cpp", "gt": "chromium_gt.json", "gt_type": "json"},
    "commons": {"raw_dir": "Common-Imaging", "language": "java", "gt": "CI_gt.txt", "gt_type": "regex", "exclude_test_files": True},
    "hadoop": {"raw_dir": "Hadoop", "language": "java", "gt": "hadoop_gt.json", "gt_type": "json", "collapse_multi_by_path": True},
    "hdc": {"raw_dir": "distributed_camera", "language": "c", "gt": "hdc_gt.json", "gt_type": "json"},
    "hdf": {"raw_dir": "drivers_framework", "language": "c", "gt": "hdf_gt.json", "gt_type": "json"},
    "hfd": {"raw_dir": "drivers_framework", "language": "c", "gt": "hdf_gt.json", "gt_type": "json"},
    "jabref": {"raw_dir": "jabref", "language": "java", "gt": "jabref_labels.txt", "gt_type": "regex"},
    "libxml": {"raw_dir": "libxml", "language": "c", "gt": "libxml_gt.json", "gt_type": "json"},
    "lucene": {"raw_dir": "Lucene", "language": "java", "gt": "lucene_gt.txt", "gt_type": "regex", "exclude_test_files": True},
    "oodt": {"raw_dir": "oodt", "language": "java", "gt": "oodt_gt.json", "gt_type": "json", "collapse_multi_by_path": True},
    "pandas": {
        "raw_dir": "pandas",
        "language": "python",
        "gt": "pandas_gt.json",
        "gt_type": "json",
        "prefer_raw_rebuild": True,
        "include_gt_only_files": True,
    },
    "sweetHome": {"raw_dir": "SweetHome_3D", "language": "java", "gt": "sh_gt.txt", "gt_type": "regex", "exclude_test_files": True},
    "teammates": {"raw_dir": "teammates", "language": "java", "gt": "teammates_labels.txt", "gt_type": "regex"},
}

LIST_COLUMNS = {
    "Module_List",
    "Modules",
    "Source_Modules",
    "Target_Modules",
    "Source_Module_List",
    "Target_Module_List",
}


@dataclass
class LabelIndex:
    labels_by_entity: dict[str, list[str]]
    regex_rules: list[tuple[str, re.Pattern[str], str]]
    root_packages: list[str]
    collapse_multi_by_path: bool
    raw_count: int
    source: Path | None


def pickle_load(path: Path) -> Any:
    with path.open("rb") as handle:
        try:
            return pickle.load(handle)
        except Exception:
            handle.seek(0)
            return pickle.load(handle, encoding="latin1")


def load_model_file(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return pickle_load(path)


def source_raw_root(data_root: Path) -> Path:
    for candidate in (data_root / "raw", data_root / "old_data"):
        if candidate.exists():
            return candidate
    return data_root / "raw"


def source_processed_dir(data_root: Path) -> Path:
    for candidate in (
        data_root / "processed",
        data_root / "old_data" / "processed",
        data_root / "Clean_Data" / "processed",
    ):
        if candidate.exists():
            return candidate
    return data_root / "processed"


def clean_path(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).replace("\\", "/").strip()
    parts = text.split("/", 1)
    if len(parts) == 2 and "." in parts[0] and parts[0].replace(".", "/") in parts[1]:
        return parts[1]
    return text


def is_test_path(value: Any) -> bool:
    text = clean_path(value).lower()
    parts = [part for part in text.split("/") if part]
    return "test" in parts or "tests" in parts


def filter_dataset_scope(dataset: str, df: pd.DataFrame) -> pd.DataFrame:
    if not DATASETS.get(dataset, {}).get("exclude_test_files") or "File" not in df:
        return df
    return df[~df["File"].map(is_test_path)].copy()


def normalize_module(value: Any) -> str:
    return str(value).strip().lower()


def normalize_modules(values: Any) -> list[str]:
    if values is None or (isinstance(values, float) and pd.isna(values)):
        return []
    if isinstance(values, str):
        text = values.strip()
        if not text or text.lower() in {"nan", "none", "__none__", "unmapped"}:
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                values = parsed
            else:
                values = [parsed]
        except Exception:
            values = [text]
    elif not isinstance(values, (list, tuple, set)):
        values = [values]

    modules: list[str] = []
    for item in values:
        if item is None or (isinstance(item, float) and pd.isna(item)):
            continue
        module = normalize_module(item)
        if module and module not in {"nan", "none", "__none__", "unmapped"} and module not in modules:
            modules.append(module)
    return modules


def filter_to_root_packages(df: pd.DataFrame, root_packages: list[str] | None) -> pd.DataFrame:
    if not root_packages or df.empty:
        return df
    dot_roots = [root.strip().replace("/", ".").rstrip(".") for root in root_packages]
    slash_roots = [root.replace(".", "/") for root in dot_roots]

    def in_root(value: Any) -> bool:
        text = clean_path(value)
        dotted = text.replace("/", ".").removesuffix(".java")
        for root in dot_roots:
            candidates = [root]
            parts = root.split(".", 1)
            if len(parts) == 2 and parts[0] in {"main", "test", "client"}:
                candidates.append(parts[1])
            if any(dotted == candidate or dotted.startswith(candidate + ".") for candidate in candidates):
                return True
        return False

    mask = pd.Series(False, index=df.index)
    if "Entity" in df:
        mask = mask | df["Entity"].map(in_root)
    if "File" in df:
        def file_in_root(value: Any) -> bool:
            text = clean_path(value)
            for root in slash_roots:
                candidates = [root]
                parts = root.split("/", 1)
                if len(parts) == 2 and parts[0] in {"main", "test", "client"}:
                    candidates.extend([
                        f"src/{parts[0]}/java/{parts[1]}",
                        parts[1],
                    ])
                if any(text == candidate or text.startswith(candidate + "/") for candidate in candidates):
                    return True
            return False

        mask = mask | df["File"].map(file_in_root)
    return df[mask].copy()


def modules_to_csv_value(values: Iterable[str]) -> str:
    return json.dumps(list(values), ensure_ascii=False)


def has_multi_label_gt(labels: LabelIndex) -> bool:
    if labels.collapse_multi_by_path:
        return False
    return any(len(normalize_modules(modules)) > 1 for modules in labels.labels_by_entity.values())


def module_match_tokens(value: str) -> list[str]:
    text = re.sub(r"^[0-9]+[a-z]?__", "", value.lower())
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return [token for token in text.split() if token and token not in {"org", "apache", "src", "main", "java"}]


def choose_best_path_module(file_name: Any, entity: Any, modules: list[str]) -> str:
    if not modules:
        return ""
    if len(modules) == 1:
        return modules[0]

    file_text = clean_path(file_name).lower()
    entity_text = str(entity or "").lower()
    haystack = f"{file_text} {file_text.replace('/', '.')} {entity_text}"
    haystack_tokens = set(module_match_tokens(haystack))

    def score(module: str) -> tuple[int, int, int, str]:
        normalized = normalize_module(module)
        module_path = normalized.replace(".", "/")
        module_dot = normalized.replace("/", ".")
        tokens = module_match_tokens(normalized)
        direct = 0
        if module_path and module_path in file_text:
            direct += 200
        if module_dot and module_dot in entity_text:
            direct += 120
        token_hits = sum(1 for token in tokens if token in haystack_tokens or token in file_text or token in entity_text)
        contiguous = 0
        if tokens:
            joined_slash = "/".join(tokens)
            joined_dot = ".".join(tokens)
            joined_plain = "".join(tokens)
            if joined_slash in file_text or joined_dot in entity_text or joined_plain in haystack.replace("/", "").replace(".", ""):
                contiguous = 50
        specificity = sum(len(token) for token in tokens)
        return (direct + token_hits * 20 + contiguous, token_hits, specificity, normalized)

    return max(modules, key=score)


def parse_regex_gt(path: Path) -> LabelIndex:
    rules: list[tuple[str, re.Pattern[str], str]] = []
    root_packages: list[str] = []
    section: str | None = None
    seen_mapping_section = False
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            lowered = line.lower()
            if "root-package" in lowered:
                section = "root"
            elif "mapping" in lowered or "<regular_expression>" in lowered:
                section = "mapping"
                seen_mapping_section = True
            elif any(token in lowered for token in ("modules", "relations", "name", "jar", "metrics")):
                section = None
            continue
        if section == "root":
            root = line.split()[0].strip().strip("/")
            if root:
                root_packages.append(root.replace("/", "."))
            continue
        if seen_mapping_section and section != "mapping":
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        module, pattern = parts[0], parts[1].strip()
        if module.lower() in {"name", "jar", "modules", "relations", "mapping"}:
            continue
        try:
            rules.append((normalize_module(module), re.compile(pattern), pattern))
        except re.error:
            continue
    if not root_packages:
        literal_prefixes: list[list[str]] = []
        for _, _, pattern in rules:
            prefix = re.split(r"(?<!\\)[(*[?|+]", pattern, maxsplit=1)[0]
            prefix = prefix.replace(r"\.", ".").rstrip(".")
            parts = [part for part in prefix.split(".") if part]
            if parts:
                literal_prefixes.append(parts)
        if literal_prefixes:
            common = list(literal_prefixes[0])
            for parts in literal_prefixes[1:]:
                keep = 0
                for left, right in zip(common, parts):
                    if left != right:
                        break
                    keep += 1
                common = common[:keep]
                if not common:
                    break
            if common:
                root_packages.append(".".join(common))
            else:
                root_packages.extend(".".join(parts[:3]) for parts in literal_prefixes if len(parts) >= 3)
    return LabelIndex(
        labels_by_entity={},
        regex_rules=rules,
        root_packages=sorted(set(root_packages)),
        collapse_multi_by_path=False,
        raw_count=len(rules),
        source=path,
    )


def parse_json_gt(path: Path) -> LabelIndex:
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels: dict[str, list[str]] = defaultdict(list)
    root_packages = [
        str(root).strip().replace("/", ".").rstrip(".")
        for root in payload.get("root_packages", [])
        if str(root).strip()
    ] if isinstance(payload, dict) else []

    def visit_group(group: dict[str, Any], inherited_module: str | None = None) -> None:
        module = group.get("name") or inherited_module or "root"
        for child in group.get("nested", []) or []:
            if not isinstance(child, dict):
                continue
            name = child.get("name")
            nested = child.get("nested")
            if nested:
                visit_group(child, module)
            elif name and module:
                normalized = clean_path(name)
                mod = normalize_module(module)
                if mod not in labels[normalized]:
                    labels[normalized].append(mod)

    for item in payload.get("structure", []) if isinstance(payload, dict) else []:
        if isinstance(item, dict):
            visit_group(item)

    return LabelIndex(dict(labels), [], root_packages, False, sum(len(v) for v in labels.values()), path)


def parse_rsf_gt(path: Path) -> LabelIndex:
    labels: dict[str, list[str]] = defaultdict(list)
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        relation, module, entity = parts[0], parts[1], " ".join(parts[2:])
        if relation.lower() != "contain":
            continue
        normalized_entity = clean_path(entity)
        normalized_module = normalize_module(module)
        if normalized_module not in labels[normalized_entity]:
            labels[normalized_entity].append(normalized_module)
    return LabelIndex(dict(labels), [], [], False, sum(len(v) for v in labels.values()), path)


def load_raw_labels(data_root: Path, dataset: str) -> LabelIndex:
    config = DATASETS[dataset]
    raw_root = source_raw_root(data_root)
    gt_path = raw_root / config["raw_dir"] / config["gt"]
    if not gt_path.exists() and config["gt_type"] == "json":
        rsf_candidates = sorted((raw_root / config["raw_dir"]).glob("*_gt.rsf"))
        if rsf_candidates:
            return parse_rsf_gt(rsf_candidates[0])
    if not gt_path.exists():
        return LabelIndex({}, [], [], False, 0, None)
    if config["gt_type"] == "regex":
        return parse_regex_gt(gt_path)
    if config["gt_type"] == "rsf":
        return parse_rsf_gt(gt_path)
    return parse_json_gt(gt_path)


def load_labels(data_root: Path, dataset: str) -> LabelIndex:
    config = DATASETS[dataset]
    gt_json_path = data_root / "GT" / f"{dataset}_gt.json"
    if gt_json_path.exists():
        labels = parse_json_gt(gt_json_path)
        if not labels.root_packages:
            raw_labels = load_raw_labels(data_root, dataset)
            labels.root_packages.extend(raw_labels.root_packages)
        labels.collapse_multi_by_path = bool(config.get("collapse_multi_by_path"))
        return labels
    labels = load_raw_labels(data_root, dataset)
    labels.collapse_multi_by_path = bool(config.get("collapse_multi_by_path"))
    return labels


def labels_for_row(row: pd.Series, labels: LabelIndex) -> list[str]:
    entity = clean_path(row.get("Entity", ""))
    file_name = clean_path(row.get("File", ""))
    if labels.labels_by_entity:
        return labels.labels_by_entity.get(entity) or labels.labels_by_entity.get(file_name) or []
    for candidate in (entity, file_name):
        if not candidate:
            continue
        matches = [module for module, pattern, _ in labels.regex_rules if pattern.search(candidate)]
        if matches:
            return list(dict.fromkeys(matches))
    return []


def infer_modules(row: pd.Series, labels: LabelIndex) -> list[str]:
    for column in ("Module_List", "Modules", "Module"):
        if column in row and normalize_modules(row[column]):
            modules = normalize_modules(row[column])
            if labels.collapse_multi_by_path and len(modules) > 1:
                return [choose_best_path_module(row.get("File", ""), row.get("Entity", ""), modules)]
            return modules
    modules = labels_for_row(row, labels)
    if labels.collapse_multi_by_path and len(modules) > 1:
        return [choose_best_path_module(row.get("File", ""), row.get("Entity", ""), modules)]
    return modules


def normalize_data_frame(df: pd.DataFrame, labels: LabelIndex) -> tuple[pd.DataFrame, dict[str, int]]:
    original_rows = len(df)
    df = df.copy()
    if "File" in df:
        df["File"] = df["File"].map(clean_path)
    if "Entity" not in df and "File" in df:
        df["Entity"] = df["File"]

    multi_label_gt = has_multi_label_gt(labels)
    df["_module_values"] = [infer_modules(row, labels) for _, row in df.iterrows()]
    df = df[df["_module_values"].map(bool)].copy()
    if df.empty:
        stats = {
            "data_rows_in": original_rows,
            "data_rows_out": 0,
            "data_rows_dropped_unlabeled": original_rows,
            "unique_files_out": 0,
            "unique_entities_out": 0,
            "multi_label_rows": 0,
            "multi_label_gt": has_multi_label_gt(labels),
        }
        return df.drop(columns=["_module_values"], errors="ignore"), stats
    df["Module"] = df["_module_values"].map(lambda modules: modules[0])
    if multi_label_gt:
        df["Module_List"] = df["_module_values"].map(modules_to_csv_value)
        df["Modules"] = df["Module_List"]
        df["is_duplicated"] = df["_module_values"].map(lambda modules: len(modules) > 1)
        if "Duplicated" in df.columns:
            df = df.drop(columns=["Duplicated"])
    else:
        df = df.drop(columns=[col for col in ("Module_List", "Modules", "is_duplicated", "Duplicated") if col in df.columns])

    subset = [col for col in ("File", "Entity", "Member_ID", "Member_Name", "Member_Type") if col in df.columns]
    if subset:
        df = df.drop_duplicates(subset=subset).reset_index(drop=True)

    stats = {
        "data_rows_in": original_rows,
        "data_rows_out": len(df),
        "data_rows_dropped_unlabeled": original_rows - len(df),
        "unique_files_out": int(df["File"].nunique()) if "File" in df else 0,
        "unique_entities_out": int(df["Entity"].nunique()) if "Entity" in df else 0,
        "multi_label_rows": int(df["_module_values"].map(lambda modules: len(modules) > 1).sum()),
        "multi_label_gt": multi_label_gt,
    }
    df = df.drop(columns=["_module_values"])
    return df, stats


def file_module_lookup(data_df: pd.DataFrame) -> dict[str, list[str]]:
    lookup: dict[str, list[str]] = {}
    for file_name, group in data_df.groupby("File"):
        modules: list[str] = []
        values = group["Module_List"] if "Module_List" in group else group["Module"]
        for value in values:
            for module in normalize_modules(value):
                if module not in modules:
                    modules.append(module)
        lookup[str(file_name)] = modules
    return lookup


def normalize_dependency_frame(df: pd.DataFrame, data_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    original_rows = len(df)
    df = df.copy()
    if df.empty:
        return df, {"dep_rows_in": 0, "dep_rows_out": 0, "dep_rows_dropped_unlabeled": 0}

    for column in ("Source_File", "Target_File"):
        if column in df:
            df[column] = df[column].map(clean_path)

    valid_files = set(data_df["File"].astype(str))
    df = df[
        df["Source_File"].astype(str).isin(valid_files)
        & df["Target_File"].astype(str).isin(valid_files)
    ].copy()

    lookup = file_module_lookup(data_df)
    source_lists = df["Source_File"].astype(str).map(lookup).map(lambda x: x or [])
    target_lists = df["Target_File"].astype(str).map(lookup).map(lambda x: x or [])
    df = df[source_lists.map(bool) & target_lists.map(bool)].copy()
    source_lists = source_lists.loc[df.index]
    target_lists = target_lists.loc[df.index]

    df["Source_Module"] = source_lists.map(lambda modules: modules[0])
    df["Target_Module"] = target_lists.map(lambda modules: modules[0])
    if "Module_List" in data_df.columns:
        df["Source_Module_List"] = source_lists.map(modules_to_csv_value)
        df["Target_Module_List"] = target_lists.map(modules_to_csv_value)
        df["Source_Modules"] = df["Source_Module_List"]
        df["Target_Modules"] = df["Target_Module_List"]
        df["Source_is_duplicated"] = source_lists.map(lambda modules: len(modules) > 1)
        df["Target_is_duplicated"] = target_lists.map(lambda modules: len(modules) > 1)
    else:
        df = df.drop(columns=[
            col for col in (
                "Source_Module_List", "Target_Module_List", "Source_Modules",
                "Target_Modules", "Source_is_duplicated", "Target_is_duplicated"
            )
            if col in df.columns
        ])
    df = df.drop_duplicates().reset_index(drop=True)

    stats = {
        "dep_rows_in": original_rows,
        "dep_rows_out": len(df),
        "dep_rows_dropped_unlabeled": original_rows - len(df),
    }
    return df, stats


def export_gt(dataset: str, labels: LabelIndex, out_dir: Path, data_df: pd.DataFrame) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    json_groups: dict[str, list[str]] = defaultdict(list)
    matched_keys = set(data_df["Entity"].map(clean_path)) | set(data_df["File"].map(clean_path))

    if labels.labels_by_entity:
        for entity, modules in labels.labels_by_entity.items():
            for module in modules:
                if entity not in json_groups[module]:
                    json_groups[module].append(entity)
            rows.append({
                "Entity": entity,
                "Module_List": modules_to_csv_value(modules),
                "Module": modules[0] if modules else "",
                "is_duplicated": len(modules) > 1,
                "Matched_In_Data": entity in matched_keys,
            })
    else:
        matched = data_df[["File", "Entity", "Module"]].drop_duplicates()
        for row in matched.itertuples(index=False):
            entity = clean_path(getattr(row, "Entity"))
            file_name = clean_path(getattr(row, "File"))
            module = normalize_module(getattr(row, "Module"))
            item = entity or file_name
            if item and item not in json_groups[module]:
                json_groups[module].append(item)
        for module, _, pattern in labels.regex_rules:
            rows.append({"Module": module, "Pattern": pattern})

    write_csv_atomic(pd.DataFrame(rows), out_dir / f"{dataset}_gt.csv")
    gt_json = {
        "@schemaVersion": "1.0",
        "name": dataset,
        "root_packages": labels.root_packages,
        "collapse_multi_by_path": labels.collapse_multi_by_path,
        "structure": [
            {
                "@type": "group",
                "name": module,
                "nested": [
                    {"@type": "item", "name": entity}
                    for entity in sorted(entities)
                ],
            }
            for module, entities in sorted(json_groups.items())
        ],
    }
    write_text_atomic(out_dir / f"{dataset}_gt.json", json.dumps(gt_json, indent=2, ensure_ascii=False))
    return {"gt_rows_out": len(rows), "gt_raw_items": labels.raw_count}


def replace_with_retry(tmp_path: Path, final_path: Path, attempts: int = 5) -> None:
    last_error: PermissionError | None = None
    for _ in range(attempts):
        try:
            os.replace(tmp_path, final_path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.5)
    raise last_error if last_error else PermissionError(final_path)


def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    df.to_csv(tmp_path, index=False)
    replace_with_retry(tmp_path, path)


def write_text_atomic(path: Path, text: str) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    replace_with_retry(tmp_path, path)


def write_report(rows: list[dict[str, Any]], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("dataset")
    write_csv_atomic(df, report_dir / "preprocess_validation_report.csv")


def find_model_file(raw_dir: Path, dataset: str) -> Path | None:
    candidates = sorted(raw_dir.glob("*-file.pkl")) + sorted(raw_dir.glob("*-file.json"))
    if not candidates:
        candidates = sorted(raw_dir.glob(f"{dataset}*.pkl")) + sorted(raw_dir.glob(f"{dataset}*.json"))
    return candidates[0] if candidates else None


def java_entity(member_name: Any, member_type: Any) -> tuple[str, str]:
    value = str(member_name).split("(", 1)[0].replace("$", ".").replace("/", ".")
    value = re.sub(r"\.java$", "", value)
    entity = value.rsplit(".", 1)[0] if str(member_type) in {"function", "var"} else value
    short = value.split(".")[-1]
    return entity, short


def non_java_member(file_name: str, member_name: Any, member_type: Any) -> tuple[str, str]:
    if member_type == "file":
        return file_name, Path(file_name).stem
    return file_name, str(member_name).split("(")[0].split(".")[-1]


def extract_entities(model: dict[str, Any], language: str) -> pd.DataFrame:
    entities = model.get("variables", [])
    cells = model.get("cells", [])
    nodes = pd.DataFrame(entities)
    nodes["ID"] = nodes.index
    nodes = nodes.rename(columns={0: "File"})
    nodes["File"] = nodes["File"].map(clean_path)
    file_to_id = dict(zip(nodes["File"], nodes["ID"]))

    rows: list[dict[str, Any]] = []
    for file_name, file_id in file_to_id.items():
        if language == "java":
            entity = re.sub(r"\.java$", "", file_name.replace("/", ".").replace("$", "."))
            member_name = Path(file_name).stem
        else:
            entity = file_name
            member_name = Path(file_name).stem
        rows.append({
            "File": file_name,
            "ID": file_id,
            "Member_Name": member_name,
            "Member_Type": "file",
            "Entity": entity,
        })

    for cell in cells:
        for detail in cell.get("details", []) or []:
            for role in ("src", "dest"):
                ent = detail.get(role, {}) or {}
                file_name = clean_path(ent.get("file"))
                member_name = ent.get("object")
                member_type = ent.get("type")
                if file_name not in file_to_id or pd.isna(member_name):
                    continue
                if language == "java":
                    entity, clean_member = java_entity(member_name, member_type)
                    if clean_member.lower() == "java":
                        clean_member = Path(file_name).stem
                else:
                    entity, clean_member = non_java_member(file_name, member_name, member_type)
                rows.append({
                    "File": file_name,
                    "ID": file_to_id[file_name],
                    "Member_Name": clean_member,
                    "Member_Type": member_type,
                    "Entity": entity,
                })
    df = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    if not df.empty:
        df["Member_ID"] = (
            df.apply(lambda row: f"{row.File}::{row.Member_Type}::{row.Member_Name}", axis=1)
            .astype("category")
            .cat.codes
        )
    return df


def extract_dependencies(model: dict[str, Any], data_df: pd.DataFrame) -> pd.DataFrame:
    file_to_id = dict(zip(data_df["File"], data_df["ID"]))
    rows: list[dict[str, Any]] = []
    for cell in model.get("cells", []) or []:
        for detail in cell.get("details", []) or []:
            src = detail.get("src", {}) or {}
            dst = detail.get("dest", {}) or {}
            src_file = clean_path(src.get("file"))
            dst_file = clean_path(dst.get("file"))
            if src_file not in file_to_id or dst_file not in file_to_id or src_file == dst_file:
                continue
            src_obj, dst_obj = src.get("object"), dst.get("object")
            dep_type = detail.get("type")
            same_as_file = (
                str(src_obj or "").replace(".java", "").replace("/", ".")
                == src_file.replace(".java", "").replace("/", ".")
                or str(dst_obj or "").replace(".java", "").replace("/", ".")
                == dst_file.replace(".java", "").replace("/", ".")
            )
            member_level = bool(src_obj and dst_obj and not same_as_file)
            rows.append({
                "Source_ID": cell.get("src") if member_level else file_to_id[src_file],
                "Target_ID": cell.get("dest") if member_level else file_to_id[dst_file],
                "Source_File": src_file,
                "Target_File": dst_file,
                "Source_Member": "" if not member_level else str(src_obj).split(".")[-1],
                "Target_Member": "" if not member_level else str(dst_obj).split(".")[-1],
                "Source_Member_Type": src.get("type"),
                "Target_Member_Type": dst.get("type"),
                "Dependency_Type": dep_type,
                "Dependency_Count": (cell.get("values", {}) or {}).get(dep_type, 1),
                "Is_Member_Level": member_level,
            })
    df = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    if not df.empty and "Dependency_Type" in df:
        df = df[~df["Dependency_Type"].astype(str).str.contains("possible", case=False, na=False)]
    return df


def append_gt_only_file_rows(df: pd.DataFrame, labels: LabelIndex, language: str) -> pd.DataFrame:
    if not labels.labels_by_entity:
        return df

    df = df.copy()
    existing = set()
    if "File" in df:
        existing.update(df["File"].astype(str).map(clean_path))
    if "Entity" in df:
        existing.update(df["Entity"].astype(str).map(clean_path))

    next_id = int(pd.to_numeric(df.get("ID", pd.Series([-1])), errors="coerce").max()) + 1 if not df.empty else 0
    next_member_id = int(pd.to_numeric(df.get("Member_ID", pd.Series([-1])), errors="coerce").max()) + 1 if not df.empty else 0
    rows: list[dict[str, Any]] = []

    for item in sorted(labels.labels_by_entity):
        file_name = clean_path(item)
        if file_name in existing:
            continue
        if language == "java":
            entity = re.sub(r"\.java$", "", file_name.replace("/", ".").replace("$", "."))
        else:
            entity = file_name
        rows.append({
            "File": file_name,
            "ID": next_id,
            "Member_Name": Path(file_name).stem,
            "Member_Type": "file",
            "Entity": entity,
            "Member_ID": next_member_id,
        })
        next_id += 1
        next_member_id += 1

    if not rows:
        return df
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


def process_raw_dataset(data_root: Path, dataset: str, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int], dict[str, int]]:
    config = DATASETS[dataset]
    raw_dir = source_raw_root(data_root) / config["raw_dir"]
    model_file = find_model_file(raw_dir, dataset)
    if model_file is None:
        raise FileNotFoundError(f"No model file found for {dataset} in {raw_dir}")
    model = load_model_file(model_file)
    labels = load_labels(data_root, dataset)
    raw_df = filter_to_root_packages(extract_entities(model, config["language"]), labels.root_packages)
    raw_df = filter_dataset_scope(dataset, raw_df)
    if config.get("include_gt_only_files"):
        raw_df = append_gt_only_file_rows(raw_df, labels, config["language"])
    data_df, data_stats = normalize_data_frame(raw_df, labels)
    deps_df, dep_stats = normalize_dependency_frame(extract_dependencies(model, data_df), data_df)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_df.to_csv(out_dir / f"{dataset}.csv", index=False)
    deps_df.to_csv(out_dir / f"{dataset}_deps.csv", index=False)
    return data_df, deps_df, data_stats, dep_stats


def copy_raw_tree(data_root: Path, clean_root: Path, mode: str) -> None:
    if mode == "none":
        return
    src = source_raw_root(data_root)
    dst = clean_root / "raw"
    if dst.exists():
        return
    if mode == "copy":
        shutil.copytree(src, dst)
    elif mode == "link":
        shutil.copytree(src, dst, copy_function=os.link)
    else:
        raise ValueError(f"Unknown copy mode: {mode}")


def clean_existing(data_root: Path, copy_raw: str) -> None:
    processed_dir = source_processed_dir(data_root)
    clean_root = data_root / "Clean_Data"
    clean_processed = clean_root / "processed"
    gt_dir = data_root / "GT"
    report_dir = clean_root / "reports"
    clean_processed.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    copy_raw_tree(data_root, clean_root, copy_raw)

    report_rows: list[dict[str, Any]] = []
    for data_path in sorted(processed_dir.glob("*.csv")):
        if data_path.name.endswith("_deps.csv"):
            continue
        dataset = data_path.stem
        if dataset not in DATASETS:
            continue
        deps_path = processed_dir / f"{dataset}_deps.csv"
        labels = load_labels(data_root, dataset)
        if DATASETS[dataset].get("prefer_raw_rebuild"):
            clean_df, _clean_deps, data_stats, dep_stats = process_raw_dataset(data_root, dataset, clean_processed)
        else:
            data_df = pd.read_csv(data_path)
            data_df = filter_to_root_packages(data_df, labels.root_packages)
            data_df = filter_dataset_scope(dataset, data_df)
            clean_df, data_stats = normalize_data_frame(data_df, labels)
            clean_df.to_csv(clean_processed / data_path.name, index=False)

            dep_stats = {"dep_rows_in": 0, "dep_rows_out": 0, "dep_rows_dropped_unlabeled": 0}
            if deps_path.exists():
                deps_df = pd.read_csv(deps_path)
                clean_deps, dep_stats = normalize_dependency_frame(deps_df, clean_df)
                clean_deps.to_csv(clean_processed / deps_path.name, index=False)

        gt_stats = export_gt(dataset, labels, gt_dir, clean_df)
        report_rows.append({
            "dataset": dataset,
            "language": DATASETS[dataset]["language"],
            "gt_source": str(labels.source) if labels.source else "",
            **data_stats,
            **dep_stats,
            **gt_stats,
        })

    write_report(report_rows, report_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--mode", choices=["clean-existing", "rebuild-raw"], default="clean-existing")
    parser.add_argument("--dataset", choices=sorted(DATASETS), help="Dataset to rebuild in rebuild-raw mode.")
    parser.add_argument("--copy-raw", choices=["none", "link", "copy"], default="none")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    if args.mode == "clean-existing":
        clean_existing(data_root, args.copy_raw)
    else:
        if not args.dataset:
            raise SystemExit("--dataset is required for --mode rebuild-raw")
        out_dir = data_root / "Clean_Data" / "processed"
        process_raw_dataset(data_root, args.dataset, out_dir)


if __name__ == "__main__":
    main()

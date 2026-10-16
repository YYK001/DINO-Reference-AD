from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from DINOv3.relation_reliability.datasets import Record


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DOMAINS = ("same", "background", "illumination", "view")
OFFICIAL_TRAIN_COUNT = 521
OFFICIAL_TEST_COUNTS = {
    "same": 689,
    "background": 305,
    "illumination": 273,
    "view": 372,
}


@dataclass(frozen=True)
class TestLocator:
    path: str
    relative_path: str
    domain: str
    sample_id: str


def find_aebad_s_root(root: str | Path) -> Path:
    supplied = Path(root).resolve()
    candidates = (
        supplied,
        supplied / "AeBAD_S",
        supplied / "AeBAD" / "AeBAD_S",
    )
    for candidate in candidates:
        if (candidate / "train" / "good").is_dir() and (candidate / "test").is_dir():
            return candidate
    raise FileNotFoundError(f"AeBAD-S train/test root not found under {supplied}")


def _images(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    return sorted(
        (
            path for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_SUFFIXES
            and not path.name.startswith("._")
            and "__MACOSX" not in path.parts
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def _test_domain(test_root: Path, image: Path) -> str:
    relative = image.relative_to(test_root)
    if len(relative.parts) != 3:
        raise RuntimeError(f"AeBAD-S test path must be test/<defect>/<domain>/<image>: {image}")
    domain = relative.parts[1]
    if domain not in DOMAINS:
        raise RuntimeError(f"unsupported AeBAD-S domain {domain!r}: {image}")
    return domain


def validate_official_counts(train_count: int, domain_counts: dict[str, int]) -> None:
    if int(train_count) != OFFICIAL_TRAIN_COUNT:
        raise RuntimeError(f"AeBAD-S train count mismatch: {train_count} != {OFFICIAL_TRAIN_COUNT}")
    if domain_counts != OFFICIAL_TEST_COUNTS:
        raise RuntimeError(
            "AeBAD-S domain counts mismatch: "
            f"actual={domain_counts}, expected={OFFICIAL_TEST_COUNTS}"
        )


def discover_phase_a(
    root: str | Path,
    *,
    formal: bool = True,
) -> tuple[Path, list[Record], list[TestLocator]]:
    dataset_root = find_aebad_s_root(root)
    train_images = _images(dataset_root / "train" / "good")
    train_records = [
        Record(
            path=str(path), label=0, mask_path=None,
            category="AeBAD-S", dataset="aebad_s", domain="source",
            shift="train_good", role="train", mask_capability="not_evaluation",
        )
        for path in train_images
    ]

    test_root = dataset_root / "test"
    test_images = _images(test_root)
    locators = []
    for index, image in enumerate(test_images):
        relative = image.relative_to(dataset_root).as_posix()
        locators.append(TestLocator(
            path=str(image),
            relative_path=relative,
            domain=_test_domain(test_root, image),
            sample_id=f"test_{index:04d}",
        ))

    if formal:
        domain_counts = {
            domain: sum(locator.domain == domain for locator in locators)
            for domain in DOMAINS
        }
        validate_official_counts(len(train_records), domain_counts)
    return dataset_root, train_records, locators


def load_phase_b_records(
    dataset_root: str | Path,
    locators: Sequence[TestLocator],
    *,
    formal: bool = True,
) -> list[Record]:
    root = Path(dataset_root).resolve()
    output = []
    for locator in locators:
        relative = Path(locator.relative_path)
        defect = relative.parts[1]
        label = int(defect != "good")
        mask = None
        if label:
            image = Path(locator.path)
            candidate = root / "ground_truth" / defect / locator.domain / f"{image.stem}.png"
            if formal and not candidate.is_file():
                raise FileNotFoundError(candidate)
            mask = str(candidate)
        output.append(Record(
            path=locator.path, label=label, mask_path=mask,
            category="AeBAD-S", dataset="aebad_s", domain=locator.domain,
            shift=defect, role="evaluation", mask_capability="pixel",
        ))
    if formal:
        logical_keys = set()
        for record in output:
            key = (record.shift.casefold(), record.domain.casefold(), Path(record.path).stem.casefold())
            if key in logical_keys:
                raise RuntimeError(f"duplicate AeBAD-S logical sample key: {key}")
            logical_keys.add(key)
        for domain in DOMAINS:
            labels = {record.label for record in output if record.domain == domain}
            if labels != {0, 1}:
                raise RuntimeError(f"AeBAD-S/{domain} evaluation labels are incomplete")
    return output

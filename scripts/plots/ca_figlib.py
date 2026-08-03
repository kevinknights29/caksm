"""Shared plumbing for the CA integrator figure scripts.

Holds the artifact paths, the solver-transcript parser, the participant-count
cost model, and the drawing and output furniture. One script per figure lives
beside this module and holds only what is specific to that figure.

Two conventions the data invites the opposite of: a timing is a distribution, so
the median is the mark and the observed range is the bar; and a stopped arm is
drawn with a distinct texture rather than dropped. A figure whose artifact is
missing or incomplete records itself as blocked, naming what it waits on.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ca_figstyle as ca

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA = ROOT / "data"
DOCS = ROOT / "docs"
PDF_DIR = DOCS / "thesis" / "figures"

WEAK = DATA / "ca-integrator-weak"
EXACT = DATA / "ca-integrator-exact-depth"
STRONG = DATA / "ca-integrator-strong"
VERTICAL = DATA / "ca-integrator-mpk-vertical"
CERTIFICATE = DATA / "ca-integrator-certificate"
PARTICIPANTS = DATA / "ca-participant-calibration-quiet"
LARGEST = DATA / "ca-integrator-largest-common"
REFEREE_DOC = DOCS / "ca_referee_baseline.md"

DPI = 300


# Transcript parsing

# srun prefixes every line with the task rank. Strip it before matching so a
# one-node and a two-node transcript parse identically.
RANK_PREFIX = re.compile(r"^\s*\d+:\s?")


def read_lines(path: Path) -> list[str]:
    """Read a transcript, stripping the srun rank prefix from every line.

    Args:
        path: Transcript to read.

    Returns:
        The lines, rank prefix removed and newline stripped.
    """
    return [
        RANK_PREFIX.sub("", line.rstrip("\n"))
        for line in path.read_text(errors="replace").splitlines()
    ]


def _search(lines: list[str], pattern: str) -> re.Match | None:
    compiled = re.compile(pattern)
    for line in lines:
        found = compiled.search(line)
        if found:
            return found
    return None


def _number(match: re.Match | None, group: int = 1, cast=float):
    if match is None:
        return None
    try:
        return cast(match.group(group))
    except (IndexError, ValueError):
        return None


@dataclass
class Run:
    """One solver transcript, reduced to the quantities the figures plot."""

    path: Path
    option: str = ""
    n: int = 0
    width: int = 0
    steps: int = 0
    world_gpus: int = 1
    nodes: int = 1
    arm: str = "as-measured"
    run_status: str = ""
    recordable: bool = True
    validation: str = ""
    unconverged: int = 0
    avg_m: float = 0.0
    max_residual: float = 0.0
    max_kappa: float = 0.0
    min_certified_s: int = 0
    fallback_blocks: int = 0
    repeats: int = 1
    cycle_ms: float | None = None
    cycle_min_ms: float | None = None
    cycle_max_ms: float | None = None
    collectives_per_step: float = 0.0
    collective_split: tuple[int, int, int, int] = (0, 0, 0, 0)
    payload_bytes_per_step: float = 0.0
    halos_per_step: float = 0.0
    halo_kib_per_step: float = 0.0
    build_halos_per_step: float | None = None
    build_halo_kib_per_step: float | None = None
    build_depth_avg: float | None = None
    build_depth_min: int | None = None
    build_depth_max: int | None = None
    build_depth_hist: dict[int, int] = field(default_factory=dict)
    transition_halos_per_step: float | None = None
    transition_halo_kib_per_step: float | None = None
    transition_depth_hist: dict[int, int] = field(default_factory=dict)
    collective_floor_ms: float = 0.0
    halo_floor_ms: float = 0.0
    host_syncs_per_step: float | None = None
    host_sync_ms: float | None = None
    host_sync_lower_bound_ms: float | None = None

    @property
    def stopped(self) -> bool:
        """Whether this run is a diagnostic rather than a performance result."""
        return (
            self.unconverged > 0
            or self.validation == "FAIL"
            or self.run_status in {"stopped", "acceptance-fail"}
        )


def parse_run(path: Path) -> Run | None:
    """Parse one solver transcript into the quantities the figures plot.

    Args:
        path: Transcript to parse.

    Returns:
        The parsed run, or None if the file carries no solver header.
    """
    lines = read_lines(path)
    run = Run(path=path)

    header = _search(
        lines,
        r"option[:=] ?(\w+) \| basis=\w+ \| orth=\w+ \| n=(\d+) \| N=\d+ "
        r"\| steps=(\d+) \| tol=[\d.e+-]+ \| m_max=\d+ \| s=(\d+)",
    )
    if header is None:
        return None
    run.option = header.group(1)
    run.n = int(header.group(2))
    run.steps = int(header.group(3))
    run.width = int(header.group(4))

    arm = _search(lines, r"arm=([\w-]+)")
    if arm is not None:
        run.arm = arm.group(1)

    explicit_status = _search(lines, r"RUN_STATUS status=([\w-]+)")
    if explicit_status is not None:
        run.run_status = explicit_status.group(1)

    topology = _search(lines, r"decomposition=(\d+) slabs across (\d+) node")
    if topology is not None:
        run.world_gpus = int(topology.group(1))
        run.nodes = int(topology.group(2))

    run.recordable = _search(lines, r"recordable=no") is None
    validation = _search(lines, r"validation: (\w+)")
    run.validation = validation.group(1) if validation else ""

    krylov = _search(lines, r"Krylov m: min=\d+ avg=([\d.]+) max=\d+ \| unconverged=(\d+)")
    if krylov is not None:
        run.avg_m = float(krylov.group(1))
        run.unconverged = int(krylov.group(2))

    run.max_residual = _number(
        _search(lines, r"max residual: ([\d.e+-]+)")) or 0.0
    run.max_kappa = _number(
        _search(lines, r"max block kappa: ([\d.e+-]+)")) or 0.0
    certified = _search(
        lines, r"certified width: requested=\d+ min=(\d+) \| fallback blocks=(\d+)")
    if certified is not None:
        run.min_certified_s = int(certified.group(1))
        run.fallback_blocks = int(certified.group(2))

    collectives = _search(
        lines, r"collectives: ([\d.]+)/step .*payload=([\d.]+) bytes/step")
    if collectives is not None:
        run.collectives_per_step = float(collectives.group(1))
        run.payload_bytes_per_step = float(collectives.group(2))
        split = _search(
            lines,
            r"norm=(\d+) projection=(\d+) gram=(\d+)(?: agreement=(\d+))?")
        if split is not None:
            run.collective_split = (
                int(split.group(1)), int(split.group(2)), int(split.group(3)),
                int(split.group(4) or 0))
    else:
        single = _search(
            lines,
            r"reduction operations: ([\d.]+)/step \| payload: ([\d.]+) bytes/step")
        if single is not None:
            run.collectives_per_step = float(single.group(1))
            run.payload_bytes_per_step = float(single.group(2))

    halo = _search(
        lines,
        r"operator communication: ([\d.]+) halos/step \| ([\d.]+) KiB/step")
    if halo is not None:
        run.halos_per_step = float(halo.group(1))
        run.halo_kib_per_step = float(halo.group(2))

    halo_detail = _search(
        lines,
        r"operator communication detail: "
        r"build_halos_per_step=([\d.]+) build_kib_per_step=([\d.]+) "
        r"build_depth_avg=([\d.]+) build_depth_min=(\d+) build_depth_max=(\d+) "
        r"transition_halos_per_step=([\d.]+) "
        r"transition_kib_per_step=([\d.]+) transition_depth=(\d+)")
    if halo_detail is not None:
        run.build_halos_per_step = float(halo_detail.group(1))
        run.build_halo_kib_per_step = float(halo_detail.group(2))
        run.build_depth_avg = float(halo_detail.group(3))
        run.build_depth_min = int(halo_detail.group(4))
        run.build_depth_max = int(halo_detail.group(5))
        run.transition_halos_per_step = float(halo_detail.group(6))
        run.transition_halo_kib_per_step = float(halo_detail.group(7))

    halo_histogram = _search(
        lines,
        r"operator communication depth histogram: units=total_exchanges "
        r"build_depth_hist=([^ ]+) transition_depth_hist=([^ ]+)")
    if halo_histogram is not None:
        def parse_histogram(text: str) -> dict[int, int]:
            """Turn a "depth:count,depth:count" field into a dict."""
            result: dict[int, int] = {}
            for entry in text.split(","):
                depth, count = entry.split(":", maxsplit=1)
                result[int(depth)] = int(count)
            return result

        run.build_depth_hist = parse_histogram(halo_histogram.group(1))
        run.transition_depth_hist = parse_histogram(halo_histogram.group(2))

    floors = _search(
        lines,
        r"calibrated floors: collectives=([\d.]+) ms/step "
        r"halo-bandwidth=([\d.]+) ms/step")
    if floors is not None:
        run.collective_floor_ms = float(floors.group(1))
        run.halo_floor_ms = float(floors.group(2))

    # The measured cost is what the decomposition uses. The probe-derived lower
    # bound is parsed too, but it is not the floor: an Nsight capture put the
    # measured cost about 24 times above it, because the probe times an empty
    # round-trip while a real read waits for the queued stream.
    host = _search(
        lines,
        r"host synchronizations: ([\d.]+)/step \| measured=([\d.]+) ms/step "
        r"\(max over ranks\) \| empty round-trip=[\d.]+ us "
        r"\| lower bound=([\d.]+) ms/step")
    if host is not None:
        run.host_syncs_per_step = float(host.group(1))
        run.host_sync_ms = float(host.group(2))
        run.host_sync_lower_bound_ms = float(host.group(3))
    else:
        # Transcripts written before the account was corrected.
        legacy = _search(
            lines,
            r"host synchronizations: ([\d.]+)/step \| probed round-trip=[\d.]+ us "
            r"\| floor=([\d.]+) ms/step \| observed stall=([\d.]+) ms/step")
        if legacy is not None:
            run.host_syncs_per_step = float(legacy.group(1))
            run.host_sync_lower_bound_ms = float(legacy.group(2))
            run.host_sync_ms = float(legacy.group(3))
        else:
            counted = _search(lines, r"host synchronizations: ([\d.]+)/step")
            if counted is not None:
                run.host_syncs_per_step = float(counted.group(1))

    timing = _search(
        lines,
        r"solve median: ([\d.]+) ms \| cycle: ([\d.]+) ms/step "
        r"\| distribution: \[([\d.]+), ([\d.]+)\] ms \((\d+) runs\)")
    if timing is not None:
        run.cycle_ms = float(timing.group(2))
        run.repeats = int(timing.group(5))
        run.cycle_min_ms = float(timing.group(3)) / run.steps
        run.cycle_max_ms = float(timing.group(4)) / run.steps
    return run


def load_runs(directory: Path) -> list[Run]:
    """Parse every transcript in a directory.

    Args:
        directory: Artifact directory holding ``*.txt`` transcripts.

    Returns:
        The parsed runs, sorted by filename; empty if the directory is absent.
    """
    if not directory.is_dir():
        return []
    runs = []
    for path in sorted(directory.glob("*.txt")):
        run = parse_run(path)
        if run is not None:
            runs.append(run)
    return runs


# The communication model


@dataclass
class Latency:
    """One measured collective or halo cost, with its distribution."""

    payload_bytes: float
    median_us: float
    min_us: float
    max_us: float
    repeats: int


class ParticipantModel:
    """The immutable participant-count calibration, read as a cost model.

    Not the same thing as the node-tier constant the solver prints in its
    calibrated floors line: that is one number per rung, while this is a measured
    cost per participant count and per payload, which is what the crossover turns
    out to depend on.

    A payload the calibration did not measure is interpolated between the
    measured points on a log-log payload axis, and every such value is marked
    interpolated where the figure reports it. Interpolating a calibration is not
    the same as fitting a curve and drawing it as a prediction.
    """

    def __init__(self) -> None:
        self.collectives: dict[int, dict[str, Latency]] = {}
        self.halos: dict[int, dict[tuple[int, int], Latency]] = {}
        self.sources: list[Path] = []
        for participants, path in (
            (2, PARTICIPANTS / "node_2participants.csv"),
            (4, PARTICIPANTS / "node_4participants.csv"),
        ):
            if not path.exists():
                continue
            self.sources.append(path)
            self.collectives[participants] = {}
            self.halos[participants] = {}
            with path.open() as handle:
                for record in csv.DictReader(handle):
                    if record["contended"] != "0":
                        continue
                    entry = Latency(
                        float(record["payload_bytes"]),
                        float(record["global_median_s"]) * 1e6,
                        float(record["global_min_s"]) * 1e6,
                        float(record["global_max_s"]) * 1e6,
                        int(record["repeats"]))
                    if record["operation"] == "allreduce":
                        self.collectives[participants][record["label"]] = entry
                    elif record["operation"] == "halo":
                        key = (int(record["n"]), int(record["s"]))
                        self.halos[participants][key] = entry

    @property
    def available(self) -> bool:
        """Whether any calibration table was found on disk."""
        return bool(self.collectives)

    @staticmethod
    def _interpolate(points: list[Latency], payload: float) -> float:
        ordered = sorted(points, key=lambda p: p.payload_bytes)
        if len(ordered) == 1:
            return ordered[0].median_us
        for index in range(len(ordered) - 1):
            low, high = ordered[index], ordered[index + 1]
            if low.payload_bytes <= payload <= high.payload_bytes:
                if high.payload_bytes == low.payload_bytes:
                    return low.median_us
                weight = (
                    (math.log(payload) - math.log(low.payload_bytes))
                    / (math.log(high.payload_bytes) - math.log(low.payload_bytes)))
                return low.median_us + weight * (high.median_us - low.median_us)
        nearest = min(ordered, key=lambda p: abs(p.payload_bytes - payload))
        return nearest.median_us

    def collective_us(
        self, participants: int, kind: str, width: int, target: int
    ) -> tuple[float, bool]:
        """Price one collective from the calibration.

        Args:
            participants: Number of participating devices.
            kind: Collective name: "norm", "projection" or "gram".
            width: Block width the collective carries.
            target: Krylov target dimension, used to size a projection.

        Returns:
            The median latency in microseconds, and whether it was measured at
            this exact payload rather than interpolated between measured ones.
        """
        table = self.collectives.get(participants)
        if not table:
            return 0.0, False
        if kind == "norm":
            return table["norm"].median_us, True
        label = f"{kind}_s{width}"
        if label in table:
            return table[label].median_us, True
        payload = (
            8.0 * width * width if kind == "gram"
            else 8.0 * target * width)
        points = [v for k, v in table.items() if k.startswith(f"{kind}_")]
        if not points:
            return 0.0, False
        return self._interpolate(points, payload), False

    def collective_payload_us(
        self, participants: int, payload_bytes: float
    ) -> tuple[float, bool]:
        """Price a small untyped all-reduce such as the int32 agreement tuple."""
        table = self.collectives.get(participants)
        if not table:
            return 0.0, False
        exact = [
            point for point in table.values()
            if point.payload_bytes == payload_bytes
        ]
        if exact:
            return exact[0].median_us, True
        points = [
            point for label, point in table.items()
            if label != "bandwidth"
        ]
        if not points:
            return 0.0, False
        return self._interpolate(points, payload_bytes), False

    def halo_us(
        self, participants: int, n: int, depth: int
    ) -> tuple[float, bool]:
        """Price one halo exchange from the calibration.

        Args:
            participants: Number of participating devices.
            n: Grid size, which sets the face area.
            depth: Number of planes exchanged.

        Returns:
            The median latency in microseconds, and whether it was measured at
            this exact payload rather than interpolated between measured ones.
        """
        table = self.halos.get(participants)
        if not table:
            return 0.0, False
        if (n, depth) in table:
            return table[(n, depth)].median_us, True
        points = [v for (grid, _), v in table.items() if grid == n]
        if not points:
            return 0.0, False
        payload = 8.0 * depth * n * n
        return self._interpolate(points, payload), False


def modeled_floors(
    model: ParticipantModel, run: Run, target: int = 25
) -> dict[str, float | bool]:
    """The predicted communication floor of one run, term by term.

    Counts come from the transcript and costs from the calibration, so nothing
    here is fitted. The halo depth is the recurrence depth the arm actually
    exchanged, which is the block width on the as-measured arm and one less on
    the exact-depth arm.
    """
    participants = run.world_gpus
    norm_us, _ = model.collective_us(participants, "norm", run.width, target)
    projection_us, projection_exact = model.collective_us(
        participants, "projection", run.width, target)
    gram_us, gram_exact = model.collective_us(
        participants, "gram", run.width, target)
    agreement_us, agreement_exact = model.collective_payload_us(
        participants, 4.0 * 4.0)
    steps = float(run.steps or 1)
    norm_ops, projection_ops, gram_ops, agreement_ops = run.collective_split
    reductions_ms = (
        norm_ops * norm_us + projection_ops * projection_us
        + gram_ops * gram_us + agreement_ops * agreement_us
    ) / steps / 1000.0
    halo_exact = True
    halo_account_exact = bool(
        run.build_depth_hist or run.transition_depth_hist)
    halo_cost_us = 0.0
    if halo_account_exact:
        for depth, count in (
            list(run.build_depth_hist.items())
            + list(run.transition_depth_hist.items())
        ):
            if depth <= 0 or count == 0:
                continue
            cost_us, exact = model.halo_us(participants, run.n, depth)
            halo_cost_us += count * cost_us
            halo_exact = halo_exact and exact
        halos_ms = halo_cost_us / steps / 1000.0
        halo_depth = "histogram"
    else:
        # Legacy artifacts predate the split account. Keep their old estimate
        # available for diagnostics, but callers must not treat it as exact.
        depth = run.width - 1 if run.arm == "exact-depth" else run.width
        depth = max(depth, 0)
        halo_us, halo_exact = (
            model.halo_us(participants, run.n, depth)
            if depth > 0 else (0.0, True))
        halos_ms = run.halos_per_step * halo_us / 1000.0
        halo_depth = depth
    return {
        "reductions_ms": reductions_ms,
        "halos_ms": halos_ms,
        # The probe-derived lower bound, not the measured stall. The stall is a
        # real measurement but it is not a separable term: it is the host
        # waiting for queued device work, so it overlaps the local work, the
        # collectives and the halos it sits behind. Only the irreducible
        # round-trip latency is disjoint from them and can enter a stacked
        # decomposition or a predicted floor. The measured stall is carried
        # alongside, as an annotation, never as a summand.
        "host_sync_ms": run.host_sync_lower_bound_ms or 0.0,
        "host_sync_measured_ms": run.host_sync_ms or 0.0,
        "exact_calibration": bool(
            projection_exact and gram_exact and agreement_exact and halo_exact),
        "exact_halo_account": halo_account_exact,
        "halo_depth": halo_depth,
    }



def load_certificate() -> tuple[list[dict], list[dict], dict, str | None]:
    """Load and gate the certificate sweep for the figures that draw it.

    Shared by the basis-conditioning and certified-width figures, which are two
    views of one artifact and must agree on what counts as a usable sweep.

    Args:
        None.

    Returns:
        The prediction rows, the evolving-solver rows, those rows keyed by
        (option, basis, n, block_width), and None. On a failed gate, three empty
        containers and the reason the figure is blocked.
    """
    predicted_path = CERTIFICATE / "predicted_width.csv"
    measured_path = CERTIFICATE / "measured_width.csv"
    if not predicted_path.exists() or not measured_path.exists():
        return [], [], {}, (
            "the certificate artifact must carry both the a-priori prediction "
            "and evolving-solver measurements; rerun "
            "scripts/regime/ca_certificate_sweep.sh on an idle Synge V100")

    with predicted_path.open() as handle:
        predicted_all = list(csv.DictReader(handle))
    with measured_path.open() as handle:
        measured_all = list(csv.DictReader(handle))
    predicted_fields = {
        "option", "basis", "n", "recurrence_degree", "block_width",
        "predicted_block_width_max", "basis_block_width_max",
        "predicted_monomial_block_width_max",
        "predicted_newton_block_width_max",
        "predicted_chebyshev_block_width_max", "block_basis_kappa",
        "block_monomial_kappa", "block_newton_kappa",
        "block_chebyshev_kappa", "limit", "contended",
    }
    measured_fields = {
        "option", "basis", "n", "block_width",
        "min_certified_block_width", "fallback_blocks",
        "first_fallback_step", "max_block_kappa", "avg_m", "unconverged",
        "contended",
    }
    if (
        not predicted_all or not measured_all
        or not predicted_fields <= set(predicted_all[0])
        or not measured_fields <= set(measured_all[0])
    ):
        return [], [], {}, (
            "the certificate CSVs predate the complete prediction/measurement "
            "block-width schema")

    if (
        len(predicted_all) != 72 or len(measured_all) != 72
        or any(r["contended"] != "0" for r in predicted_all)
        or any(r["contended"] != "0" for r in measured_all)
    ):
        return [], [], {}, (
            "this figure requires exactly 72 idle prediction rows and 72 "
            "matching evolving-solver rows")
    predicted = predicted_all
    measured = measured_all
    required_bases = {"monomial", "newton", "chebyshev"}
    if (
        not predicted or not measured
        or {r["basis"] for r in predicted} != required_bases
        or {r["basis"] for r in measured} != required_bases
    ):
        return [], [], {}, (
            "idle measurements for monomial, Newton, and Chebyshev are all "
            "required; predicted-only or two-basis artifacts are diagnostic")

    expected = {
        (option, basis, n, width)
        for option in ("basket", "rainbow")
        for basis in required_bases
        for n in (31, 61)
        for width in range(1, 7)
    }
    try:
        predicted_keys = [
            (
                r["option"], r["basis"], int(r["n"]),
                int(r["block_width"]),
            )
            for r in predicted
        ]
        measured_keys = [
            (
                r["option"], r["basis"], int(r["n"]),
                int(r["block_width"]),
            )
            for r in measured
        ]
    except ValueError:
        return [], [], {}, (
            "the certificate artifact contains a malformed block-width key")
    if (
        set(predicted_keys) != expected or len(predicted_keys) != len(expected)
        or set(measured_keys) != expected or len(measured_keys) != len(expected)
    ):
        return [], [], {}, (
            "the idle certificate artifact is a trimmed or partial sweep; "
            "this figure requires both options, all three bases, n=31/61, "
            "and block columns 1..6")

    # Catch a row that mixes the two width conventions before it is plotted: a
    # recurrence degree pasted into a block-width column would silently shift
    # every series by one column.
    try:
        for record in predicted:
            basis = record["basis"]
            block_width = int(record["block_width"])
            predicted_max = int(record["predicted_block_width_max"])
            basis_max = int(record["basis_block_width_max"])
            if (
                int(record["recurrence_degree"]) != block_width
                or not 1 <= predicted_max <= 6
                or not 1 <= basis_max <= 6
                or predicted_max != int(
                    record[f"predicted_{basis}_block_width_max"])
                or not math.isclose(
                    float(record["block_basis_kappa"]),
                    float(record[f"block_{basis}_kappa"]),
                    rel_tol=1e-12, abs_tol=0.0)
            ):
                raise ValueError
        for record in measured:
            block_width = int(record["block_width"])
            certified = int(record["min_certified_block_width"])
            if not 1 <= certified <= block_width <= 6:
                raise ValueError
    except (KeyError, ValueError):
        return [], [], {}, (
            "the certificate rows mix recurrence-degree and block-column "
            "conventions or contain an invalid basis-specific field")

    measured_by_key = {
        (r["option"], r["basis"], int(r["n"]), int(r["block_width"])): r
        for r in measured
    }
    missing = [
        r for r in predicted
        if (r["option"], r["basis"], int(r["n"]), int(r["block_width"]))
        not in measured_by_key
    ]
    if missing:
        return [], [], {}, (
            f"{len(missing)} idle prediction point(s) have no matching "
            "evolving-solver measurement")

    # The a-priori maximum is a property of the operator and the basis, so it
    # must not depend on which width was requested when it was recorded.
    prediction_series = {
        (option, basis, n): {
            int(r["predicted_block_width_max"])
            for r in predicted
            if r["option"] == option and r["basis"] == basis
            and int(r["n"]) == n
        }
        for option in ("basket", "rainbow")
        for basis in required_bases
        for n in (31, 61)
    }
    if any(len(values) != 1 for values in prediction_series.values()):
        return [], [], {}, (
            "the a-priori maximum changes across requested block widths for "
            "the same option, basis, and grid")


    return predicted, measured, measured_by_key, None

# Outputs


class Figure:
    """One figure's identity and the files it owns.

    Every script builds exactly one of these, then either writes it or records
    it as blocked. Routing both through the same object is what keeps a newly
    blocked figure from leaving last week's image behind as though it were
    current.

    The name is what the figure is about, not where it sits in a list. An
    ordinal in the filename ages the moment a figure is inserted or dropped,
    and it tells a reader who opens the PNG nothing at all.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.stem = f"ca_{name}"

    @property
    def png(self) -> Path:
        """The 300 dpi raster this figure writes."""
        return HERE / f"{self.stem}.png"

    @property
    def pdf(self) -> Path:
        """The vector copy this figure writes for the thesis."""
        return PDF_DIR / f"{self.stem}.pdf"

    @property
    def sidecar(self) -> Path:
        """The marker written in place of an image when blocked."""
        return HERE / f"{self.stem}.blocked.txt"

    def write(self, fig, rows: list[dict]) -> None:
        """Write a 300 dpi PNG and a vector PDF for the thesis.

        Args:
            fig: The finished figure.
            rows: The plotted values, one dict per mark. Not written to disk;
                each script derives its printed summary from these, and a
                reader who wants the numbers re-runs the script.
        """
        self.sidecar.unlink(missing_ok=True)
        fig.savefig(self.png, dpi=DPI, bbox_inches="tight")
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {self.png.relative_to(ROOT)}")

    def blocked(self, reason: str) -> str:
        """Record the figure as blocked, naming the artifact it waits on."""
        # A newly blocked figure must not leave a stale accepted image for the
        # packaging script to pick up beside its blocked sidecar.
        for stale in (self.png, self.pdf, HERE / f"{self.stem}.pdf"):
            stale.unlink(missing_ok=True)
        with self.sidecar.open("w") as handle:
            handle.write(f"figure={self.name}\n")
            handle.write("status=blocked\n")
            handle.write(f"reason={reason}\n")
        print(f"  BLOCKED {self.name}: {reason}")
        return reason


def run(figure: Figure, draw) -> int:
    """Standard entry point for a ca_*.py figure script.

    The palette is validated before anything is drawn, so a re-stepped ramp that
    lost its separation stops the run instead of producing an unreadable figure.
    """
    ca.validate_palette()
    print(f"{figure.name}:")
    draw()
    return 0


# Drawing furniture


HEADLINE_SIZE = 12.5
PANEL_SIZE = 9.5


def title(fig, headline: str) -> None:
    """Set the figure's single headline.

    The headline states the finding rather than restating the axes, so a reader
    who reads only the top line leaves holding the claim. Provenance stays out
    of the image and lives in the LaTeX caption instead, which is where a
    published figure carries it.

    Args:
        fig: Figure to title.
        headline: One line naming the result.
    """
    fig.suptitle(headline, fontsize=HEADLINE_SIZE, color=ca.C_INK)


def panel_title(ax, text: str, subtitle: str = "") -> None:
    """Set a panel heading, optionally with a subordinate second line.

    Args:
        ax: Axes to title.
        text: The panel's name.
        subtitle: Optional second line, set smaller and lighter.
    """
    if subtitle:
        ax.set_title(
            f"{text}\n{subtitle}", fontsize=PANEL_SIZE, color=ca.C_INK,
            linespacing=1.4)
    else:
        ax.set_title(text, fontsize=PANEL_SIZE, color=ca.C_INK)


def legend(ax, *, loc: str = "best", ncol: int = 1, **kwargs):
    """Draw a legend in the one style every figure in this set uses.

    Args:
        ax: Axes to attach the legend to.
        loc: Matplotlib location string.
        ncol: Number of columns.
        **kwargs: Passed through to ``Axes.legend``, for explicit handles
            and labels.

    Returns:
        The created legend.
    """
    return ax.legend(
        fontsize=8.0, frameon=False, loc=loc, ncol=ncol,
        labelcolor=ca.C_INK, handlelength=2.6, borderaxespad=0.4, **kwargs)


def label_series(ax, x, y, text: str, color: str, *, dy: float = 9.0,
                 dx: float = 0.0, ha: str = "center",
                 size: float = 9.0) -> None:
    """Name a series on the series itself.

    Cheaper for the reader than a legend, which asks them to hold a color in
    memory and match it across the figure. Only worth it where the label has
    clear space; where series converge, a legend is the better tool.

    Args:
        ax: Axes to draw on.
        x: Data x of the anchor point.
        y: Data y of the anchor point.
        text: The series name.
        color: The series color, so label and data read as one thing.
        dy: Vertical offset in points from the anchor.
        dx: Horizontal offset in points from the anchor.
        ha: Horizontal alignment.
        size: Font size in points.
    """
    ax.annotate(
        text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
        ha=ha, va="bottom" if dy >= 0 else "top", fontsize=size,
        color=color, fontweight="medium", zorder=6)


def label_value(ax, x, y, text: str, *, dy: float = 6.0, dx: float = 0.0,
                color: str | None = None, size: float = 7.6,
                ha: str = "center", weight: str = "normal") -> None:
    """Put a number on the mark it belongs to.

    A reader comparing two bars that differ by a tenth of a millisecond is
    guessing, and a number that needs the script re-run to recover is not on the
    figure.

    Args:
        ax: Axes to draw on.
        x: Data x of the mark.
        y: Data y of the mark.
        text: The formatted number.
        dy: Vertical offset in points from the mark.
        dx: Horizontal offset in points from the mark.
        color: Text color; defaults to the subordinate gray.
        size: Font size in points.
        ha: Horizontal alignment.
        weight: Font weight.
    """
    ax.annotate(
        text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
        ha=ha, va="bottom" if dy >= 0 else "top", fontsize=size,
        color=color or ca.C_MUTED, fontweight=weight, zorder=6,
        annotation_clip=False)


def style_axes(ax, grid_axis: str = "both") -> None:
    """Apply the shared axes styling: light grid, no top or right spine.

    Args:
        ax: Axes to style.
        grid_axis: Which axis carries grid lines: "both", "x" or "y".
    """
    ax.grid(alpha=0.35, lw=0.5, color=ca.C_GRID, axis=grid_axis)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(ca.C_GRID)
    ax.tick_params(colors=ca.C_INK, labelsize=8.5)
    ax.xaxis.label.set_color(ca.C_INK)
    ax.yaxis.label.set_color(ca.C_INK)
    ax.title.set_color(ca.C_INK)


def plot_distribution(ax, x, run: Run, color: str, marker: str, label=None):
    """Draw one run as a median mark with its observed range.

    Args:
        ax: Axes to draw on.
        x: Horizontal position for the mark.
        run: The parsed run supplying the median and range.
        color: Series color.
        marker: Matplotlib marker for the median.
        label: Legend label, or None to keep the mark out of the legend.
    """
    if run.cycle_ms is None:
        return
    low = run.cycle_ms - (run.cycle_min_ms or run.cycle_ms)
    high = (run.cycle_max_ms or run.cycle_ms) - run.cycle_ms
    ax.errorbar(
        x, run.cycle_ms, yerr=[[max(low, 0.0)], [max(high, 0.0)]],
        color=color, marker=marker, ms=4.5, lw=1.2, capsize=2.5,
        elinewidth=0.9, label=label, zorder=4,
    )


def draw_series(ax, points: list[Run], color: str, marker: str,
                linestyle: str, label: str, seen_stopped: set) -> None:
    """Draw one arm's trajectory against GPU count, stopped points included.

    A segment touching a stopped point takes the stopped texture, so the reader
    can follow where the arm went without the segment reading as a measured
    trend.

    Args:
        ax: Axes to draw on.
        points: Runs making up this arm, in any order.
        color: Series color.
        marker: Matplotlib marker for measured points.
        linestyle: Line style for measured segments.
        label: Legend label for the arm.
        seen_stopped: Set of axes already carrying a "stopped" legend entry,
            mutated so the entry appears once per panel.
    """
    if not points:
        return
    ordered = sorted(points, key=lambda r: r.world_gpus)
    labeled = False
    for first, second in zip(ordered, ordered[1:]):
        crossing = first.stopped or second.stopped
        ax.plot(
            [first.world_gpus, second.world_gpus],
            [first.cycle_ms, second.cycle_ms],
            color=ca.C_STOPPED if crossing else color,
            ls=(0, (1, 2)) if crossing else linestyle,
            lw=1.0 if crossing else 1.3, zorder=2)
    for point in ordered:
        if point.stopped:
            key = id(ax)
            ax.plot(
                [point.world_gpus], [point.cycle_ms], marker="x", ms=6,
                lw=0, color=ca.C_STOPPED, zorder=4,
                label="stopped" if key not in seen_stopped else None)
            seen_stopped.add(key)
        else:
            plot_distribution(
                ax, point.world_gpus, point, color, marker,
                label=None if labeled else label)
            labeled = True
    if not labeled:
        # Every point stopped: keep the arm's identity in the legend anyway.
        ax.plot([], [], color=color, ls=linestyle, marker=marker, lw=1.3,
                ms=4.5, label=label)


# GPU count alone cannot separate the two distinct two-GPU rungs, and the whole
# point of the strong-scaling ladder is that one node and two nodes differ.
STRONG_TOPOLOGIES = [
    ((1, 1), "1 GPU"),
    ((2, 1), "2 GPU\n1 node"),
    ((2, 2), "2 GPU\n2 nodes"),
    ((4, 2), "4 GPU\n2 nodes"),
]
STRONG_TOPOLOGY_X = {
    topology: float(index)
    for index, (topology, _) in enumerate(STRONG_TOPOLOGIES)
}


def draw_topology_series(
    ax, points: list[Run], color: str, marker: str,
    linestyle: str, label: str, seen_stopped: set
) -> None:
    """Draw one arm against topology, keeping the two two-GPU rungs apart.

    Args:
        ax: Axes to draw on.
        points: Runs making up this arm, in any order.
        color: Series color.
        marker: Matplotlib marker for measured points.
        linestyle: Line style for measured segments.
        label: Legend label for the arm.
        seen_stopped: Set of axes already carrying a "stopped" legend entry,
            mutated so the entry appears once per panel.
    """
    if not points:
        return
    ordered = sorted(
        points, key=lambda run: STRONG_TOPOLOGY_X[(run.world_gpus, run.nodes)])
    labeled = False
    for first, second in zip(ordered, ordered[1:]):
        crossing = first.stopped or second.stopped
        ax.plot(
            [
                STRONG_TOPOLOGY_X[(first.world_gpus, first.nodes)],
                STRONG_TOPOLOGY_X[(second.world_gpus, second.nodes)],
            ],
            [first.cycle_ms, second.cycle_ms],
            color=ca.C_STOPPED if crossing else color,
            ls=(0, (1, 2)) if crossing else linestyle,
            lw=1.0 if crossing else 1.3, zorder=2)
    for point in ordered:
        x = STRONG_TOPOLOGY_X[(point.world_gpus, point.nodes)]
        if point.stopped:
            key = id(ax)
            ax.plot(
                [x], [point.cycle_ms], marker="x", ms=6, lw=0,
                color=ca.C_STOPPED, zorder=4,
                label="stopped" if key not in seen_stopped else None)
            seen_stopped.add(key)
        else:
            plot_distribution(
                ax, x, point, color, marker,
                label=None if labeled else label)
            labeled = True
    if not labeled:
        ax.plot([], [], color=color, ls=linestyle, marker=marker, lw=1.3,
                ms=4.5, label=label)

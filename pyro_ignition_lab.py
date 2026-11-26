"""热解与着火模拟 · Python 3.10+ · 图形界面使用标准库 Tkinter。"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import os
import math
import queue
import random
import sys
import tempfile
import threading
from dataclasses import asdict, dataclass, replace
from pathlib import Path

R = 8.314462618
KELVIN = 273.15
ROOM = 25.0
SAMPLE_CAPACITY = 1500.0  # J / (kg initial sample K)
GAS_CAPACITY = 3200.0
FURNACE_TO_SOLID = 95.0  # W / (kg initial sample K)
FURNACE_TO_GAS = 260.0
SOLID_TO_GAS = 30.0
PYROLYSIS_HEAT = 260_000.0  # J / kg decomposed solid, absorbed
OXIDATION_HEAT = 5_500_000.0  # J / kg volatile, released
OXIDATION_A = 2.0e5  # 1/s, hypothetical
OXIDATION_E = 76_000.0  # J/mol, hypothetical
VENT_TIME = 18.0


@dataclass(frozen=True)
class Parameters:
    ramp: float = 95.0  # °C/min
    furnace_max: float = 590.0  # °C
    duration: float = 480.0  # s
    residue: float = 22.0  # percent of decomposed solid mass
    pyro_energy: float = 127.0  # kJ/mol
    log_prefactor: float = 8.0  # log10(A / 1 s^-1)
    ignition_rise: float = 40.0  # K above furnace
    ignition_heat: float = 6.0  # kW / kg initial sample
    persistence: float = 2.0  # s


FIELDS = (
    ("ramp", "升温速率", "°C/min", 1, 300),
    ("furnace_max", "最高炉温", "°C", 26, 850),
    ("duration", "模拟时长", "s", 10, 3600),
    ("residue", "残炭产率", "%", 0, 95),
    ("pyro_energy", "热解活化能", "kJ/mol", 30, 300),
    ("log_prefactor", "热解 log₁₀(A / s⁻¹)", "", 2, 14),
    ("ignition_rise", "气相相对炉温的温升", "K", 1, 350),
    ("ignition_heat", "气相放热率下限", "kW/kg₀", 0.1, 1000),
    ("persistence", "判据持续时间", "s", 0.5, 30),
)


@dataclass(frozen=True)
class Point:
    time: float
    furnace: float
    solid: float
    gas: float
    mass: float
    conversion: float
    gas_release: float
    vented: float
    oxidized: float
    volatile: float


@dataclass(frozen=True)
class Result:
    points: list[Point]
    ignition: float | None
    onset: float | None
    peak_release: float
    peak_gas: float
    max_balance_error: float


def validate(params: Parameters) -> None:
    for key, name, unit, minimum, maximum in FIELDS:
        value = getattr(params, key)
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"{name}请输入 {minimum:g}–{maximum:g} {unit}".rstrip())


def simulate(params: Parameters, oxygen_ratio: float, step: float = 0.1) -> Result:
    """Compare an air-like (1.0) atmosphere with no oxidation (0.0)."""
    validate(params)
    if not math.isfinite(oxygen_ratio) or not 0 <= oxygen_ratio <= 1:
        raise ValueError("相对空气氧含量须为 0–1。")
    if not 0.01 <= step <= 0.2:
        raise ValueError("计算步长须为 0.01–0.2 秒。")

    def furnace(t: float) -> float:
        return KELVIN + min(params.furnace_max, ROOM + params.ramp * t / 60)

    def heat_rates(solid: float, gas: float, ambient: float) -> tuple[float, float]:
        exchange = SOLID_TO_GAS * (gas - solid)
        return (
            (FURNACE_TO_SOLID * (ambient - solid) + exchange) / SAMPLE_CAPACITY,
            (FURNACE_TO_GAS * (ambient - gas) - exchange) / GAS_CAPACITY,
        )

    def transfer(ts: float, tg: float, ambient: float, interval: float) -> tuple[float, float]:
        ds, dg = heat_rates(ts, tg, ambient)
        ms, mg = heat_rates(ts + interval * ds / 2, tg + interval * dg / 2, ambient)
        return ts + interval * ms, tg + interval * mg

    def record(time: float, heat_kw: float) -> Point:
        return Point(
            time, furnace(time) - KELVIN, ts - KELVIN, tg - KELVIN,
            100 * (active + char), 100 * (1 - active), heat_kw,
            100 * vented, 100 * oxidized, 100 * volatile,
        )

    steps = math.ceil(params.duration / step)
    dt = params.duration / steps
    sample_every = max(1, math.ceil(steps / 2100))
    pyro_a = 10.0 ** params.log_prefactor
    pyro_e = params.pyro_energy * 1000
    yield_char = params.residue / 100
    ts = tg = ROOM + KELVIN
    active = 1.0
    char = volatile = vented = oxidized = 0.0
    peak_release = balance_error = 0.0
    peak_gas = ROOM
    onset = ignition = qualifying_since = None
    points = [record(0.0, 0.0)]

    for index in range(1, steps + 1):
        time = index * dt
        ts, tg = transfer(ts, tg, furnace(time - 0.75 * dt), dt / 2)

        k_pyro = pyro_a * math.exp(-pyro_e / (R * ts))
        reacted = active * -math.expm1(-k_pyro * dt)
        active -= reacted
        new_char = reacted * yield_char
        char += new_char
        produced = reacted - new_char

        k_ox = oxygen_ratio * OXIDATION_A * math.exp(-OXIDATION_E / (R * tg))
        k_vent = 1 / VENT_TIME
        k_total = k_ox + k_vent
        # Exact inventory balance for constant volatile production during this step.
        lost_fraction = -math.expm1(-k_total * dt)
        generated_remaining = produced * lost_fraction / (k_total * dt)
        lost = volatile * lost_fraction + max(0.0, produced - generated_remaining)
        volatile = volatile * (1 - lost_fraction) + generated_remaining
        burned = lost * k_ox / k_total
        oxidized += burned
        vented += lost - burned
        heat_kw = burned * OXIDATION_HEAT / (1000 * dt)

        ts -= reacted * PYROLYSIS_HEAT / SAMPLE_CAPACITY
        tg += burned * OXIDATION_HEAT / GAS_CAPACITY
        ts, tg = transfer(ts, tg, furnace(time - 0.25 * dt), dt / 2)
        if not (1 < ts < 3000 and 1 < tg < 3000):
            raise ValueError("模型温度超出计算范围。请检查参数是否适用于当前模型。")
        balance_error = max(balance_error, abs(active + char + volatile + vented + oxidized - 1))
        peak_release = max(peak_release, heat_kw)
        peak_gas = max(peak_gas, tg - KELVIN)
        if onset is None and active <= 0.95:
            onset = time

        if ignition is None:
            passes = tg - furnace(time) >= params.ignition_rise and heat_kw >= params.ignition_heat
            if passes:
                if qualifying_since is None:
                    qualifying_since = time
                if time - qualifying_since + 1e-9 >= params.persistence:
                    ignition = qualifying_since
            else:
                qualifying_since = None

        if index % sample_every == 0 or index == steps:
            points.append(record(time, heat_kw))

    return Result(points, ignition, onset, peak_release, peak_gas, balance_error)


def export_csv(path: str | Path, air: Result, nitrogen: Result) -> None:
    columns = (
        ("solid_temperature_C", "solid"), ("gas_temperature_C", "gas"),
        ("remaining_solid_mass_pct", "mass"), ("pyrolysis_conversion_pct", "conversion"),
        ("oxidation_heat_kW_per_kg0", "gas_release"), ("volatile_inventory_pct", "volatile"),
        ("oxidized_volatile_pct", "oxidized"), ("vented_volatile_pct", "vented"),
    )
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["time_s", "furnace_temperature_C"] + [
            atmosphere + "_" + label for atmosphere in ("air", "nitrogen") for label, _ in columns
        ])
        for a, n in zip(air.points, nitrogen.points, strict=True):
            values = [a.time, a.furnace] + [getattr(point, field) for point in (a, n) for _, field in columns]
            writer.writerow([f"{value:.7f}" for value in values])


# Inert thermogravimetry calibration
FIT_KEYS = ("pyro_energy", "log_prefactor", "residue")
FIT_BOUNDS = {key: (lo, hi) for key, _, _, lo, hi in FIELDS if key in FIT_KEYS}
GA = "GA"
SCE = "SCE-UA"
METHODS = (GA, SCE)


@dataclass(frozen=True)
class TGCurve:
    name: str
    time: tuple[float, ...]
    temperature: tuple[float, ...]
    mass: tuple[float, ...]
    inverse_rt: tuple[float, ...]
    dt: tuple[float, ...]

    @classmethod
    def build(cls, name, times, temperatures, masses):
        if len(times) != len(temperatures) or len(times) != len(masses) or len(times) < 6:
            raise ValueError("每条 TG 曲线至少需要 6 行完整数据。")
        if not all(math.isfinite(value) for series in (times, temperatures, masses) for value in series):
            raise ValueError("时间、温度和质量须为有限数值。")
        dt = tuple(b - a for a, b in zip(times, times[1:]))
        if any(interval <= 0 or not math.isfinite(interval) for interval in dt):
            raise ValueError("time_s 须严格递增。")
        if min(temperatures) <= -KELVIN:
            raise ValueError("样品温度须高于绝对零度。")
        if masses[0] <= 0 or min(masses) < 0:
            raise ValueError("初始质量必须大于零，其他质量不得为负。")
        values = tuple((mass / masses[0]) * 100 for mass in masses)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("质量归一化后存在非有限数值。")
        return cls(name, tuple(times), tuple(temperatures), values,
                   tuple(1 / (R * (temp + KELVIN)) for temp in temperatures), dt)


def read_tg(path):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as file:
        sample = file.read(4096)
        file.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(file, dialect=dialect)
        names = {name.strip().lower(): name for name in reader.fieldnames or ()}
        required = ("time_s", "sample_temperature_c", "mass_pct")
        if not all(key in names for key in required):
            raise ValueError("CSV 表头须包含 time_s,sample_temperature_C,mass_pct。")
        times, temperatures, masses = [], [], []
        for row in reader:
            if not any(value for value in row.values()):
                continue
            try:
                time, temperature, mass = (float(row[names[key]]) for key in required)
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path.name} 第 {reader.line_num} 行含空值或非数字。") from error
            times.append(time)
            temperatures.append(temperature)
            masses.append(mass)
            if len(times) > 12000:
                raise ValueError("单条 TG 曲线最多支持 12,000 行。")
    return TGCurve.build(path.name, times, temperatures, masses)


def predict_tg(curve, values):
    energy, log_a, char = values
    prefactor = 10.0 ** log_a
    energy *= 1000
    exp = math.exp
    previous = prefactor * exp(-energy * curve.inverse_rt[0])
    cumulative = 0.0
    remaining = [100.0]
    for interval, inverse in zip(curve.dt, curve.inverse_rt[1:]):
        current = prefactor * exp(-energy * inverse)
        cumulative += (previous + current) * (interval / 2)
        remaining.append(char + (100 - char) * exp(-cumulative))
        previous = current
    return remaining


def sample_tg():
    curves = []
    for ramp in (12, 40):
        duration = 60 * 550 / ramp
        times = [index * duration / 135 for index in range(136)]
        temperatures = [30 + (ramp / 60) * time for time in times]
        curve = TGCurve.build(f"合成示例 · {ramp} °C/min", times, temperatures, [100.0] * len(times))
        mass = predict_tg(curve, (129.0, 8.2, 24.0))
        curves.append(TGCurve.build(curve.name, times, temperatures, mass))
    return tuple(curves)


@dataclass(frozen=True)
class FitResult:
    method: str
    values: tuple[float, ...]
    rmse: float
    evaluations: int
    history: tuple[tuple[int, float], ...]


class FitCancelled(Exception):
    pass


def fit_tg(curves, method, budget=2000, seed=13, cancelled=lambda: False, report=lambda n: None):
    if not curves:
        raise ValueError("请选择 TG 数据。")

    def objective(parameters):
        squared = 0.0
        for curve in curves:
            prediction = predict_tg(curve, parameters)
            squared += sum((actual - fitted) ** 2 for actual, fitted in zip(curve.mass, prediction)) / len(prediction)
        return math.sqrt(squared / len(curves))

    return optimize_parameters(objective, tuple(FIT_BOUNDS[key] for key in FIT_KEYS),
                               method, budget, seed, cancelled, report)


def optimize_parameters(objective, bounds, method, budget, seed, cancelled, report):
    if method not in METHODS or not bounds or not 100 <= budget <= 4000 or budget < 5 * (2 * len(bounds) + 1):
        raise ValueError("请选择算法、拟合参数及足够的评价次数。")
    dimension = len(bounds)
    rng = random.Random(seed)
    size = 5 * (2 * dimension + 1)
    evaluations = 0
    best = (float("inf"), None)
    history = []

    def decode(point):
        return tuple(low + component * (high - low) for component, (low, high) in zip(point, bounds))

    def uniform():
        return tuple(rng.random() for _ in range(dimension))

    def inside(point):
        return all(0 <= value <= 1 for value in point)

    def evaluate(point):
        nonlocal evaluations, best
        if cancelled():
            raise FitCancelled()
        candidate = (objective(decode(point)), point)
        evaluations += 1
        if candidate[0] < best[0]:
            best = candidate
            history.append((evaluations, best[0]))
        if evaluations % 40 == 0 or evaluations == budget:
            report(evaluations)
        return candidate

    # Both algorithms start with the same stratified population for a given seed.
    strata = [rng.sample(range(size), k=size) for _ in range(dimension)]
    population = sorted(evaluate(tuple((bins[index] + rng.random()) / size for bins in strata))
                        for index in range(size))

    if method == GA:
        def parent():
            return min(rng.sample(population, 3))[1]

        while evaluations < budget:
            next_population = population[:3]
            scale = 0.075 * (1 - evaluations / budget) ** 2 + 0.0004
            while len(next_population) < size and evaluations < budget:
                left, right = parent(), parent()
                candidate = []
                blend = rng.uniform(-0.25, 1.25)
                for a, b in zip(left, right):
                    value = blend * a + (1 - blend) * b
                    if rng.random() < 0.38:
                        value += rng.gauss(0, scale)
                    candidate.append(max(0.0, min(1.0, value)))
                next_population.append(evaluate(tuple(candidate)))
            population = sorted(next_population)
    else:
        complexes_count = 5
        complex_size = size // complexes_count
        while evaluations < budget:
            complexes = [population[index::complexes_count] for index in range(complexes_count)]
            for group in complexes:
                for _ in range(complex_size):
                    if evaluations >= budget:
                        break
                    group.sort()
                    indices = list(range(len(group)))
                    subset = []
                    for _ in range(dimension + 1):
                        chosen = rng.choices(indices, weights=[len(group) - i for i in indices])[0]
                        indices.remove(chosen)
                        subset.append(chosen)
                    subset.sort()
                    worst_index = subset[-1]
                    worst = group[worst_index]
                    centroid = tuple(sum(group[index][1][axis] for index in subset[:-1]) / dimension
                                     for axis in range(dimension))
                    reflection = tuple(2 * center - value for center, value in zip(centroid, worst[1]))
                    candidate = evaluate(reflection if inside(reflection) else uniform())
                    if candidate[0] >= worst[0] and evaluations < budget:
                        contraction = tuple((center + value) / 2 for center, value in zip(centroid, worst[1]))
                        candidate = evaluate(contraction)
                        if candidate[0] >= worst[0] and evaluations < budget:
                            candidate = evaluate(uniform())
                    group[worst_index] = candidate
            population = sorted(candidate for group in complexes for candidate in group)

    if not history:
        raise ValueError("搜索范围内未找到可计算的参数。请检查工况、参数范围和时间步长。")
    if history[-1][0] != evaluations:
        history.append((evaluations, best[0]))
    return FitResult(method, decode(best[1]), best[0], evaluations, tuple(history))


class CalibrationWindow:
    def __init__(self, lab):
        self.lab, self.tk, self.ttk = lab, lab.tk, lab.ttk
        self.curves, self.results = (), ()
        self.queue = queue.Queue()
        self.stop = threading.Event()
        self.busy = False
        tk, ttk = self.tk, self.ttk
        self.window = tk.Toplevel(lab.root)
        self.window.title("TG 参数拟合 · GA / SCE-UA")
        self.window.geometry("1050x740")
        self.window.minsize(760, 580)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        body = ttk.Frame(self.window, padding=14)
        body.pack(fill="both", expand=True)
        lab.text(body, "TG 参数拟合", bg=BG, size=18, bold=True).pack(anchor="w")
        lab.text(body, "氮气 TG · 最小化实验质量与模型质量之间的均方根误差（百分点）",
                 bg=BG, fg=MUTED, wraplength=940, justify="left").pack(anchor="w", pady=(2, 8))
        row = ttk.Frame(body)
        row.pack(fill="x")
        self.load_button = ttk.Button(row, text="导入 TG CSV", command=self.load)
        self.load_button.pack(side="left", padx=(0, 6))
        self.demo_button = ttk.Button(row, text="加载合成示例", command=self.demo)
        self.demo_button.pack(side="left", padx=(0, 10))
        self.curve_summary = tk.StringVar(value="尚未导入曲线")
        lab.text(row, value=self.curve_summary, bg=BG, fg=MUTED).pack(side="left", fill="x")
        lab.text(body, "CSV 列名  time_s, sample_temperature_C, mass_pct  ·  每条曲线以首行质量归一为 100%",
                 bg=BG, fg=MUTED, size=9, wraplength=940, justify="left").pack(anchor="w", pady=4)
        settings = ttk.Frame(body)
        settings.pack(fill="x", pady=5)
        self.method = tk.StringVar(value="同时比较")
        self.budget = tk.StringVar(value="2000")
        self.seed = tk.StringVar(value="13")
        ttk.Combobox(settings, textvariable=self.method, values=("同时比较", GA, SCE),
                     width=10, state="readonly").pack(side="left", padx=(0, 9))
        for name, variable in (("每种算法评价次数", self.budget), ("种子", self.seed)):
            lab.text(settings, name, bg=BG, size=9).pack(side="left", padx=(0, 4))
            ttk.Entry(settings, textvariable=variable, width=7).pack(side="left", padx=(0, 8))
        self.fit_button = ttk.Button(settings, text="开始拟合", command=self.start, style="Primary.TButton")
        self.fit_button.pack(side="left", padx=(0, 6))
        self.stop_button = ttk.Button(settings, text="停止", command=self.stop.set, state="disabled")
        self.stop_button.pack(side="left")
        self.status = tk.StringVar(value="可导入多个升温程序的氮气 TG 数据。")
        lab.text(body, value=self.status, bg=BG, fg=MUTED, wraplength=940, justify="left").pack(anchor="w", pady=3)

        columns = ("method", "error", "energy", "loga", "residue", "eval")
        labels = ("算法", "RMSE (百分点)", "E (kJ/mol)", "log₁₀(A / s⁻¹)", "残炭产率 (%)", "评价次数")
        self.table = ttk.Treeview(body, columns=columns, show="headings", selectmode="browse", height=2)
        for key, label in zip(columns, labels):
            self.table.heading(key, text=label)
            self.table.column(key, width=120, minwidth=80, anchor="center")
        self.table.pack(fill="x", pady=(3, 6))
        self.table.bind("<<TreeviewSelect>>", self.set_controls)
        toolbar = ttk.Frame(body)
        toolbar.pack(fill="x")
        self.plot_mode = tk.StringVar(value="fit")
        for label, mode in (("TG 拟合曲线", "fit"), ("误差收敛", "history")):
            ttk.Radiobutton(toolbar, text=label, variable=self.plot_mode, value=mode, command=self.draw).pack(
                side="left", padx=(0, 7))
        self.selection = ttk.Combobox(toolbar, state="readonly", width=27)
        self.selection.pack(side="left")
        self.selection.bind("<<ComboboxSelected>>", lambda _event: self.draw())
        self.plot = tk.Canvas(body, bg=PANEL, height=250, highlightthickness=0)
        self.plot.pack(fill="both", expand=True, pady=(4, 7))
        self.plot.bind("<Configure>", lambda _event: self.draw())
        actions = ttk.Frame(body)
        actions.pack(fill="x")
        self.apply_button = ttk.Button(actions, text="应用选中参数到模拟", command=self.apply)
        self.apply_button.pack(side="left", padx=(0, 6))
        self.export_button = ttk.Button(actions, text="导出拟合对比 CSV", command=self.save)
        self.export_button.pack(side="left")
        lab.text(body, "拟合范围  E 30–300 kJ/mol · log₁₀A 2–14 · 残炭产率 0–95%",
                 bg=BG, fg=MUTED, size=9).pack(anchor="w", pady=(4, 0))
        self.set_controls()
        self.window.after(80, self.poll)

    def close(self):
        self.stop.set()
        self.window.destroy()

    def set_controls(self, _event=None):
        self.load_button.configure(state="disabled" if self.busy else "normal")
        self.demo_button.configure(state="disabled" if self.busy else "normal")
        self.fit_button.configure(state="normal" if self.curves and not self.busy else "disabled")
        self.stop_button.configure(state="normal" if self.busy else "disabled")
        self.apply_button.configure(state="normal" if self.results and self.table.selection() else "disabled")
        self.export_button.configure(state="normal" if self.results else "disabled")

    def set_curves(self, curves):
        self.curves = curves
        self.results = ()
        self.table.delete(*self.table.get_children())
        self.curve_summary.set(f"{len(curves)} 条曲线 · {sum(len(curve.time) for curve in curves):,} 点")
        self.selection.configure(values=[curve.name for curve in curves])
        self.selection.current(0)
        self.status.set("数据已载入。选择算法后开始拟合。")
        self.set_controls()
        self.draw()

    def load(self):
        paths = self.lab.dialog.askopenfilenames(parent=self.window, filetypes=[("TG CSV", "*.csv"), ("所有文件", "*")])
        if paths:
            try:
                curves = tuple(read_tg(path) for path in paths)
                if sum(len(curve.time) for curve in curves) > 24000:
                    raise ValueError("合计最多支持 24,000 个采样点。")
                self.set_curves(curves)
            except (OSError, UnicodeError, ValueError, csv.Error) as error:
                self.status.set(f"导入失败 · {error}")

    def demo(self):
        self.set_curves(sample_tg())

    def start(self):
        if self.busy or not self.curves:
            return
        try:
            budget, seed = int(self.budget.get()), int(self.seed.get())
            if not 100 <= budget <= 4000:
                raise ValueError()
        except ValueError:
            self.status.set("请输入 100–4,000 次评价和一个整数种子。")
            return
        self.busy = True
        self.stop.clear()
        self.results = ()
        self.table.delete(*self.table.get_children())
        self.set_controls()
        self.draw()
        selected = self.method.get()
        methods = METHODS if selected == "同时比较" else (selected,)
        curves = self.curves
        self.status.set(f"正在拟合 · {methods[0]}")

        def worker():
            try:
                outputs = []
                for method in methods:
                    outputs.append(fit_tg(curves, method, budget, seed, self.stop.is_set,
                        lambda count: self.queue.put(("status", f"{method} · {count:,} / {budget:,} 次评价"))))
                self.queue.put(("done", tuple(outputs)))
            except FitCancelled:
                self.queue.put(("error", "拟合已停止。"))
            except (ValueError, ArithmeticError) as error:
                self.queue.put(("error", f"拟合失败 · {error}"))

        threading.Thread(target=worker, daemon=True).start()

    def poll(self):
        try:
            while True:
                event, value = self.queue.get_nowait()
                if event == "done":
                    self.finish(value)
                else:
                    self.status.set(value)
                    if event == "error":
                        self.busy = False
                        self.set_controls()
        except queue.Empty:
            pass
        self.window.after(80, self.poll)

    def finish(self, results):
        self.busy = False
        self.results = results
        for result in results:
            energy, log_a, residue = result.values
            self.table.insert("", "end", iid=result.method, values=(
                result.method, f"{result.rmse:.5g}", f"{energy:.4f}", f"{log_a:.5f}", f"{residue:.4f}", result.evaluations))
        best = min(results, key=lambda result: result.rmse)
        self.table.selection_set(best.method)
        self.status.set(f"拟合完成 · 本次最低 RMSE {best.rmse:.5g} 个百分点（{best.method}）")
        self.set_controls()
        self.draw()

    def apply(self):
        selection = self.table.selection()
        if selection:
            result = next(item for item in self.results if item.method == selection[0])
            for key, value in zip(FIT_KEYS, result.values):
                self.lab.inputs[key].set(f"{value:.10g}")
            self.lab.run()

    def save(self):
        if not self.results:
            return
        path = self.lab.dialog.asksaveasfilename(parent=self.window, defaultextension=".csv",
            initialfile="tg_fit_comparison.csv", filetypes=[("CSV", "*.csv")])
        if path:
            try:
                with open(path, "w", encoding="utf-8-sig", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["dataset", "time_s", "sample_temperature_C", "observed_mass_pct"] +
                                    [result.method + "_fitted_mass_pct" for result in self.results])
                    for curve in self.curves:
                        name = curve.name
                        if name.lstrip().startswith(("=", "+", "-", "@")):
                            name = "'" + name
                        predictions = [predict_tg(curve, result.values) for result in self.results]
                        for time, temp, actual, *fitted in zip(curve.time, curve.temperature, curve.mass, *predictions):
                            writer.writerow([name] + [f"{value:.7f}" for value in (time, temp, actual, *fitted)])
            except OSError as error:
                self.status.set(f"导出失败 · {error}")
            else:
                self.status.set(f"拟合对比已导出 · {Path(path).name}")

    def draw(self):
        canvas = self.plot
        canvas.delete("all")
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width < 260 or height < 110:
            return
        if not self.curves:
            canvas.create_text(width / 2, height / 2, text="导入氮气 TG 数据或加载合成示例", fill=MUTED)
            return
        palette = {GA: AIR, SCE: NITROGEN}
        history_mode = self.plot_mode.get() == "history"
        index = max(0, self.selection.current())
        curve = self.curves[index]
        if history_mode:
            if not self.results:
                canvas.create_text(width / 2, height / 2, text="运行拟合后显示各次评价的最低误差", fill=MUTED)
                return
            series = [(result.method, palette[result.method],
                       [(x, math.log10(max(y, 1e-9))) for x, y in result.history]) for result in self.results]
            xname, yname = "函数评价次数", "RMSE (百分点，对数轴)"
        else:
            x = curve.time
            series = [("实验", FURNACE, list(zip(x, curve.mass)))]
            series.extend((result.method, palette[result.method], list(zip(x, predict_tg(curve, result.values))))
                          for result in self.results)
            xname, yname = "时间 (s)", "残余质量 (%)"
        xs = [x for _, _, coords in series for x, _ in coords]
        ys = [y for _, _, coords in series for _, y in coords]
        xlow = 1 if history_mode else min(xs)
        xhigh = max(max(xs), xlow + 1e-9)
        ylow = math.floor(min(ys)) if history_mode else min(0, min(ys))
        yhigh = max(ylow + 1, math.ceil(max(ys)) if history_mode else max(100, max(ys)))
        left, right, top, bottom = 68, width - 18, 34, height - 36
        xx = lambda x: left + (right - left) * (x - xlow) / (xhigh - xlow)
        yy = lambda y: bottom - (bottom - top) * (y - ylow) / (yhigh - ylow)
        canvas.create_text(8, 12, text=yname, anchor="w", fill=INK)
        for step in range(6):
            x = xlow + (xhigh - xlow) * step / 5
            y = ylow + (yhigh - ylow) * step / 5
            canvas.create_line(xx(x), top, xx(x), bottom, fill=LINE)
            canvas.create_line(left, yy(y), right, yy(y), fill=LINE)
            canvas.create_text(left - 7, yy(y), text=f"{10**y:.2g}" if history_mode else f"{y:.0f}", anchor="e", fill=MUTED)
            canvas.create_text(xx(x), bottom + 10, text=f"{x:.0f}", fill=MUTED)
        canvas.create_text((left + right) / 2, height - 9, text=xname, fill=MUTED)
        for i, (name, color, coords) in enumerate(series):
            points = []
            for index, (x, y) in enumerate(coords):
                if history_mode and index:
                    points.extend((xx(x), yy(coords[index - 1][1])))
                points.extend((xx(x), yy(y)))
            if len(points) >= 4:
                canvas.create_line(*points, fill=color, width=2, dash=(3, 4) if name == "实验" else ())
            x = right - 65 * (len(series) - i)
            canvas.create_line(x, 15, x + 16, 15, fill=color, width=2)
            canvas.create_text(x + 20, 15, text=name, anchor="w", fill=MUTED)


MODEL_TEXT = """模型与判据
─────────
固体和挥发物气相各用一个平均温度。炉温从 25 °C 线性升高，达到上限后保温。

热解率  rₚ = A exp(−E/RTₛ) · 未反应固体比例
生成的挥发物在模型中可流出，也可发生等效氧化。空气分支使用恒定相对氧浓度，氮气分支的氧化速率为零。两分支使用相同的炉温程序。

固相考虑炉体换热、气固换热与热解吸热。气相考虑炉体换热、气固换热与挥发物氧化放热。运行结束时的残余固体包括未反应固体和残炭。

热解起点记为转化率首次达到 5% 的时间。模型着火定义为气相温升（相对炉温）和氧化放热率同时越过输入阈值，并持续指定时间。图中标记的是满足持续条件的时间段起点。

内置参数为合成示例。气相模型采用固定等效参数，着火时间是模型判据结果。用于实际材料或实验工况时，需要热分析和独立的着火/放热实验校准。

单位 kg₀ 指每千克初始样品质量。CSV 中记录每个采样时刻的炉温、相温度、残余固体、转化率、计算步平均放热率和挥发物质量分配。

TG 校准 · GA 与 SCE-UA
导入起始于未热解样品的氮气 TG CSV，使用实测样品温度及时间。每条曲线以首行质量归一为 100%。模型在实测温度下积分速率，曲线之间等权，用质量 RMSE 比较拟合效果。不同升温程序可联合拟合，拟合误差用于比较本次计算，参数仍需独立实验验证。

动力学背景参考
NIST SP 1018-1，第 7.2 节
Vyazovkin 等，Thermochimica Acta 520 (2011), 1–19
DOI 10.1016/j.tca.2011.03.034"""


BG, PANEL, INK, MUTED = "#eff3f8", "#ffffff", "#1c2d43", "#526780"
HEADER, LINE, ACCENT = "#142945", "#dce5f1", "#215dae"
AIR, NITROGEN, FURNACE = "#c74b27", "#087b76", "#697a90"


@dataclass(frozen=True)
class SlabParameters:
    thickness: float = 8.0       # mm
    density: float = 550.0        # kg/m³
    conductivity: float = 0.19    # W/(m K)
    heat_capacity: float = 1550.0  # J/(kg K)
    flux: float = 16.0           # kW/m², absorbed at the exposed surface
    front_h: float = 10.0         # W/(m² K)
    back_h: float = 18.0
    emissivity: float = 0.85
    ambient: float = 25.0         # °C, also the initial temperature
    duration: float = 360.0       # s
    cells: int = 40
    step: float = 0.2             # s


SLAB_FIELDS = (
    ("thickness", "厚度", "mm", 0.1, 200),
    ("density", "初始密度", "kg/m³", 20, 10000),
    ("conductivity", "导热系数", "W/(m·K)", 0.01, 400),
    ("heat_capacity", "比热容", "J/(kg·K)", 300, 6000),
    ("flux", "表面吸收热通量", "kW/m²", 0, 75),
    ("front_h", "正面换热系数", "W/(m²·K)", 0, 500),
    ("back_h", "背面换热系数", "W/(m²·K)", 0, 500),
    ("emissivity", "两面发射率", "", 0, 1),
    ("ambient", "环境和初始温度", "°C", -40, 150),
    ("duration", "模拟时长", "s", 1, 3600),
    ("cells", "网格数", "", 4, 160),
    ("step", "时间步长", "s", 0.05, 1),
)

STEFAN_BOLTZMANN = 5.670374419e-8


@dataclass(frozen=True)
class SlabFrame:
    time: float
    front: float
    back: float
    temperature: tuple[float, ...]
    residual: tuple[float, ...]
    remaining_mass: float   # % of initial sample mass
    loss_rate: float       # g/(m² s), step average
    composition: tuple[float, ...] = ()  # condensed components followed by generated gas (% initial)


@dataclass(frozen=True)
class SlabResult:
    depth_mm: tuple[float, ...]  # includes both surfaces
    frames: tuple[SlabFrame, ...]
    peak_loss: float
    mass_error: float
    energy_error: float
    component_names: tuple[str, ...] = ()  # named condensed species; gas is the final composition column


@dataclass(frozen=True)
class SolidProperties:
    density: float
    heat_capacity: float
    conductivity: float


@dataclass(frozen=True)
class ReactionStep:
    energy: float       # kJ/mol
    log_a: float        # log10(A / s^-1)
    solid_yield: float  # % of the mass consumed by this step
    heat: float         # kJ/kg consumed, positive if endothermic


@dataclass(frozen=True)
class SlabMechanism:
    mode: str = "parallel"
    first: ReactionStep = ReactionStep(120, 8, 30, 120)
    second: ReactionStep = ReactionStep(146, 9, 15, 150)
    intermediate: SolidProperties = SolidProperties(430, 1450, .14)
    char: SolidProperties = SolidProperties(280, 1100, .10)


@dataclass(frozen=True)
class Component:
    name: str
    initial: float  # fraction of initial bulk sample mass


@dataclass(frozen=True)
class NetworkReaction:
    source: str
    target: str     # a condensed component or "gases"
    prefactor: float  # Z in 1/s
    energy: float     # kJ/mol
    yield_pct: float  # condensed product per consumed mass
    heat: float       # effective kJ/kg of reactant consumed, endothermic positive


@dataclass(frozen=True)
class ReactionNetwork:
    components: tuple[Component, ...]
    reactions: tuple[NetworkReaction, ...]

    @property
    def names(self):
        return tuple(item.name for item in self.components)

    def validate(self):
        if not self.components or not self.reactions:
            raise ValueError("至少需要一个凝聚相组分和一条反应。")
        seen = set()
        for row, item in enumerate(self.components, 1):
            name = item.name
            if (not isinstance(name, str) or not name or name != name.strip() or not name.isprintable()
                    or name.casefold() == "gases" or name in seen):
                raise ValueError(f"组分第 {row} 行名称为空、重复、含无效字符或使用了保留名 gases。")
            seen.add(name)
            if not 0 <= item.initial <= 1:
                raise ValueError(f"组分第 {row} 行初始质量分数须为 0–1。")
        total = math.fsum(item.initial for item in self.components)
        if abs(total - 1) > 1e-9:
            raise ValueError(f"初始质量分数之和为 {total:.10g}，应为 1。")
        for i, r in enumerate(self.reactions, 1):
            if r.source not in seen or (r.target != "gases" and r.target not in seen):
                raise ValueError(f"反应第 {i} 行 From 或 To 名称未在组分表定义。")
            if r.source == r.target:
                raise ValueError(f"反应第 {i} 行的 From 与 To 相同。")
            if not (math.isfinite(r.prefactor) and r.prefactor > 0 and math.isfinite(r.energy)
                    and r.energy >= 0 and math.isfinite(r.heat) and 0 <= r.yield_pct <= 100):
                raise ValueError(f"反应第 {i} 行需满足 Z > 0、E ≥ 0、凝聚相产率 0–100%，反应热须为有限数值。")
            if r.target == "gases" and r.yield_pct != 0:
                raise ValueError(f"反应第 {i} 行产物为 gases，凝聚相产率应为 0%。")


class ReactionGrid:
    """First-order competing channels with frozen rates during each reaction half-step."""
    def __init__(self, network, count):
        network.validate()
        indices = {name: i for i, name in enumerate(network.names)}
        groups = {}
        for r in network.reactions:
            groups.setdefault(indices[r.source], []).append(
                (indices.get(r.target), r.prefactor, r.energy * (1000 / R), r.yield_pct / 100, r.heat * 1000))
        self.groups = tuple(groups.items())
        total = math.fsum(item.initial for item in network.components)
        initial = [item.initial / total for item in network.components]
        self.mass = [initial.copy() for _ in range(count)]
        # Track the portion of original stock separately from regenerated intermediates.
        self.virgin = [initial.copy() for _ in range(count)]

    def advance(self, cell, temperature, dt):
        mass, virgin = self.mass[cell], self.virgin[cell]
        produced = [0.0] * len(mass)
        gas = heat = original_consumed = 0.0
        for source, edges in self.groups:
            if not mass[source]:
                continue
            rates = [z * math.exp(-e / temperature) for _, z, e, _, _ in edges]
            rate = math.fsum(rates)
            if not rate:
                continue
            fraction = -math.expm1(-rate * dt)
            consumed = mass[source] * fraction
            mass[source] -= consumed
            original = virgin[source] * fraction
            virgin[source] -= original
            original_consumed += original
            for (target, _, _, retained, enthalpy), k in zip(edges, rates):
                amount = consumed * (k / rate)
                if target is not None:
                    produced[target] += amount * retained
                gas += amount * (1 - retained)
                heat += amount * enthalpy
        # A product enters its next reaction in the following half-step, regardless of row order.
        for i, value in enumerate(produced):
            mass[i] += value
        return gas, heat, original_consumed


REACTION_FIELDS = (
    ("energy", "活化能 (kJ/mol)", 30, 300),
    ("log_a", "log₁₀(A / s⁻¹)", 2, 14),
    ("solid_yield", "固体产率 (%)", 0, 100),
    ("heat", "反应热 (kJ/kg)", 0, 1500),
)
PROPERTY_FIELDS = (
    ("density", "密度 (kg/m³)", 20, 10000),
    ("heat_capacity", "比热容 (J/(kg·K))", 300, 6000),
    ("conductivity", "导热系数 (W/(m·K))", 0.01, 400),
)


def validate_mechanism(mechanism):
    if isinstance(mechanism, ReactionNetwork):
        mechanism.validate()
        return
    if mechanism.mode not in ("parallel", "series"):
        raise ValueError("请选择并联或串联反应。")
    materials = ((mechanism.char, "残炭"),)
    if mechanism.mode == "series":
        materials += ((mechanism.intermediate, "中间产物"),)
    for fields, groups in (
        (REACTION_FIELDS, ((mechanism.first, "反应 1"), (mechanism.second, "反应 2"))),
        (PROPERTY_FIELDS, materials),
    ):
        for item, name in groups:
            for key, label, low, high in fields:
                value = getattr(item, key)
                if not math.isfinite(value) or not low <= value <= high:
                    raise ValueError(f"{name}的{label}须在 {low:g}–{high:g}。")


def tridiagonal_solve(diagonal, rhs, lower, upper):
    """Solve in place. Lower and upper entries exclude the nonexistent end terms."""
    for i, below in enumerate(lower, start=1):
        factor = below / diagonal[i - 1]
        diagonal[i] -= factor * upper[i - 1]
        rhs[i] -= factor * rhs[i - 1]
    rhs[-1] /= diagonal[-1]
    for i in range(len(rhs) - 2, -1, -1):
        rhs[i] = (rhs[i] - upper[i] * rhs[i + 1]) / diagonal[i]
    return rhs


def reaction_amounts(parallel, raw, intermediate, k1, k2, dt, yield_one):
    if parallel:
        rate = k1 + k2
        fraction = -math.expm1(-rate * dt) / rate if rate else dt
        return raw * (k1 * fraction), raw * (k2 * fraction), 0.0
    first = raw * -math.expm1(-k1 * dt)
    generated = first * yield_one
    gap = abs(k1 - k2) * dt
    integral = -math.expm1(-gap) / gap if gap else 1.0
    retained = min(generated, raw * yield_one * (k1 * dt)
                   * math.exp(-min(k1, k2) * dt) * integral)
    used = -math.expm1(-k2 * dt)
    return first, intermediate * used + (generated - retained), intermediate * (1 - used) + retained


def slab_simulate_multistep(p, mechanism, cancelled, profiles=True):
    validate_slab(p)
    validate_mechanism(mechanism)
    count = int(p.cells)
    steps = math.ceil(p.duration / p.step)
    dt = p.duration / steps
    dz = p.thickness / (1000 * count)
    cell_mass = p.density * dz
    ambient = p.ambient + KELVIN
    depths = (0.0, *(p.thickness * (i + .5) / count for i in range(count)), p.thickness)
    every = max(1, math.ceil(steps / 1500))
    parallel = mechanism.mode == "parallel"
    temp = [ambient] * count
    front = back = ambient
    raw, mid, char = [1.0] * count, [0.0] * count, [0.0] * count
    r1, r2 = mechanism.first, mechanism.second
    a1, a2 = 10 ** r1.log_a, 10 ** r2.log_a
    e1, e2 = r1.energy * 1000 / R, r2.energy * 1000 / R
    h1, h2 = 1000 * r1.heat, 1000 * r2.heat
    y1, y2 = r1.solid_yield / 100, r2.solid_yield / 100
    materials = (SolidProperties(p.density, p.heat_capacity, p.conductivity),
                 mechanism.intermediate, mechanism.char)
    specific_volumes = tuple(p.density / material.density for material in materials)
    cp = tuple(material.heat_capacity for material in materials)
    k = tuple(material.conductivity for material in materials)
    sigma = p.emissivity * STEFAN_BOLTZMANN
    max_energy = max_mass = total_lost = peak_loss = 0.0

    def thermal_properties(u, v, w):
        v0, v1, v2 = (u * specific_volumes[0], v * specific_volumes[1], w * specific_volumes[2])
        total = v0 + v1 + v2
        void = max(0.0, 1 - total)
        # The static void contributes effective fixed nitrogen properties (rho*cp, k).
        capacity = cell_mass * (u * cp[0] + v * cp[1] + w * cp[2]) + dz * void * 1200
        conductivity = (v0 * k[0] + v1 * k[1] + v2 * k[2] + void * .026) / (total + void)
        return capacity, conductivity

    def react_half():
        lost = 0.0
        for i, temperature in enumerate(temp):
            first, second, next_mid = reaction_amounts(
                parallel, raw[i], mid[i], a1 * math.exp(-e1 / temperature),
                a2 * math.exp(-e2 / temperature), dt / 2, y1)
            old_capacity, _ = thermal_properties(raw[i], mid[i], char[i])
            raw[i] = max(0, raw[i] - first - (second if parallel else 0))
            mid[i] = next_mid
            char[i] += (first * y1 if parallel else 0.0) + second * y2
            lost += first * (1 - y1) + second * (1 - y2)
            new_capacity, _ = thermal_properties(raw[i], mid[i], char[i])
            temp[i] -= cell_mass * (first * h1 + second * h2) / ((old_capacity + new_capacity) / 2)
        if any(not 1 < value < 3000 for value in temp):
            raise ValueError("温度超出模型范围，请减小步长并检查反应热和热物性。")
        return lost * cell_mass

    def boundary(flux, h, surface, conductivity):
        h += sigma * (surface + ambient) * (surface ** 2 + ambient ** 2)
        resistance = dz / (2 * conductivity)
        factor = 1 + h * resistance
        return flux / factor, h / factor, resistance

    def make_frame(time, loss_rate):
        residue = tuple(100 * (a + b + c) for a, b, c in zip(raw, mid, char))
        composition = (100 * sum(raw) / count, 100 * sum(mid) / count,
                       100 * sum(char) / count, 100 * total_lost / (count * cell_mass))
        return SlabFrame(time, front - KELVIN, back - KELVIN,
                         (front - KELVIN, *(value - KELVIN for value in temp), back - KELVIN) if profiles else (),
                         residue if profiles else (), sum(residue) / count, loss_rate, composition)

    frames = [make_frame(0, 0)]
    for index in range(1, steps + 1):
        if cancelled():
            raise FitCancelled()
        lost = react_half()
        capacity, conductivity = zip(*(thermal_properties(*parts) for parts in zip(raw, mid, char)))
        conductance = [2 / (dz / a + dz / b) for a, b in zip(conductivity, conductivity[1:])]
        source_f, coeff_f, resistance_f = boundary(p.flux * 1000, p.front_h, front, conductivity[0])
        source_b, coeff_b, resistance_b = boundary(0, p.back_h, back, conductivity[-1])
        left, right = [coeff_f, *conductance], [*conductance, coeff_b]
        ratio = [dt / value for value in capacity]
        diagonal = [1 + scale * (a + b) for scale, a, b in zip(ratio, left, right)]
        rhs = temp.copy()
        rhs[0] += ratio[0] * (source_f + coeff_f * ambient)
        rhs[-1] += ratio[-1] * (source_b + coeff_b * ambient)
        before = temp
        temp = tridiagonal_solve(diagonal, rhs,
                                 [-g * scale for g, scale in zip(conductance, ratio[1:])],
                                 [-g * scale for g, scale in zip(conductance, ratio)])
        qf = source_f + coeff_f * (ambient - temp[0])
        qb = source_b + coeff_b * (ambient - temp[-1])
        conduction_balance = sum(c * (b - a) for c, a, b in zip(capacity, before, temp))
        max_energy = max(max_energy, abs(conduction_balance - dt * (qf + qb)))
        lost += react_half()
        total_lost += lost
        front = temp[0] + (source_f + coeff_f * (ambient - temp[0])) * resistance_f
        back = temp[-1] + (source_b + coeff_b * (ambient - temp[-1])) * resistance_b
        loss_rate = 1000 * lost / dt
        peak_loss = max(peak_loss, loss_rate)
        solid = (sum(raw) + sum(mid) + sum(char)) / count
        max_mass = max(max_mass, abs(solid + total_lost / (count * cell_mass) - 1))
        if index % every == 0 or index == steps:
            frames.append(make_frame(index * dt, loss_rate))
    return SlabResult(depths, tuple(frames), peak_loss, max_mass, max_energy)


def validate_slab(parameters):
    for key, name, unit, low, high in SLAB_FIELDS:
        value = getattr(parameters, key)
        if not math.isfinite(value) or not low <= value <= high or (key == "cells" and int(value) != value):
            raise ValueError(f"{name}须为 {low:g}–{high:g} {unit}".rstrip() +
                             (" 的整数。" if key == "cells" else "。"))


def slab_simulate(parameters, kinetics, cancelled=lambda: False, *, profiles=True):
    """One-dimensional inert slab with fixed geometry and constant thermal properties.

    Finite-volume backward Euler diffusion with reaction half-steps. Boundary
    transfer uses a half-cell resistance and a lagged radiation coefficient.
    kinetics is (E in kJ/mol, log10 of A in s^-1, char yield in percent),
    or None for a pure-conduction comparison.
    """
    if isinstance(kinetics, SlabMechanism):
        return slab_simulate_multistep(parameters, kinetics, cancelled, profiles)
    validate_slab(parameters)
    p = parameters
    count = int(p.cells)
    steps = math.ceil(p.duration / p.step)
    dt = p.duration / steps
    dx = (p.thickness / 1000) / count
    depth = (0.0, *(p.thickness * (i + 0.5) / count for i in range(count)), p.thickness)
    ambient = p.ambient + KELVIN
    capacity = p.density * p.heat_capacity * dx
    conduction = p.conductivity / dx
    resistance = 0.5 / conduction
    share = dt * conduction / capacity
    applied = p.flux * 1000
    radiation = p.emissivity * STEFAN_BOLTZMANN
    sample_every = max(1, math.ceil(steps / 1500))
    initial_mass = p.density * dx * count
    temperatures = [ambient] * count
    reacted = [0.0] * count
    front = back = ambient
    peak_loss = vented = sum_reaction_heat = sum_boundary_heat = 0.0
    max_mass_error = max_energy_error = 0.0
    network = ReactionGrid(kinetics, count) if isinstance(kinetics, ReactionNetwork) else None
    single = kinetics is not None and network is None
    energy, log_prefactor, yield_percent = kinetics if single else (0.0, 0.0, 0.0)
    if single:
        for key, value in zip(FIT_KEYS, kinetics, strict=True):
            low, high = FIT_BOUNDS[key]
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError("热解参数超出主窗口输入范围。")
    prefactor = 10 ** log_prefactor if single else 0.0
    activation = energy * 1000 / R
    char = yield_percent / 100
    half_dt = dt / 2
    cooling = PYROLYSIS_HEAT / p.heat_capacity

    def react_half():
        total = 0.0
        gas = heat = 0.0
        if network is not None:
            for i, temp in enumerate(temperatures):
                released, absorbed, _ = network.advance(i, temp, half_dt)
                temperatures[i] -= absorbed / p.heat_capacity
                gas += released
                heat += absorbed
        elif prefactor:
            for i, temp in enumerate(temperatures):
                converted = (1.0 - reacted[i]) * -math.expm1(-prefactor * math.exp(-activation / temp) * half_dt)
                reacted[i] += converted
                temperatures[i] -= cooling * converted
                total += converted
        if any(not (1 < value < 3000) for value in temperatures):
            raise ValueError("温度超出当前模型的计算范围。可缩小时间步长并检查输入参数。")
        return gas + total * (1 - char), heat + total * PYROLYSIS_HEAT

    def boundary(source, convection, surface):
        h = convection + radiation * (surface + ambient) * (surface * surface + ambient * ambient)
        denominator = 1 + h * resistance
        return source / denominator, h / denominator

    def frame(time, loss_rate):
        composition = ()
        if network is None:
            residual = tuple(100 * (1 - alpha * (1 - char)) for alpha in reacted)
        else:
            residual = tuple(100 * sum(mass) for mass in network.mass)
            composition = (*(100 * sum(column) / count for column in zip(*network.mass)),
                           100 * vented / initial_mass)
        return SlabFrame(time, front - KELVIN, back - KELVIN,
                         (front - KELVIN, *(value - KELVIN for value in temperatures), back - KELVIN) if profiles else (),
                         residual if profiles else (), sum(residual) / count, loss_rate, composition)

    frames = [frame(0, 0)]
    for step in range(1, steps + 1):
        if cancelled():
            raise FitCancelled()
        first_gas, first_heat = react_half()
        source_f, h_f = boundary(applied, p.front_h, front)
        source_b, h_b = boundary(0.0, p.back_h, back)
        factor = dt / capacity

        diagonal = [1 + 2 * share] * count
        diagonal[0] = 1 + share + factor * h_f
        diagonal[-1] = 1 + share + factor * h_b
        rhs = temperatures.copy()
        rhs[0] += factor * (source_f + h_f * ambient)
        rhs[-1] += factor * (source_b + h_b * ambient)
        adjacent = [-share] * (count - 1)
        temperatures = tridiagonal_solve(diagonal, rhs, adjacent, adjacent)

        q_f = source_f + h_f * (ambient - temperatures[0])
        q_b = source_b + h_b * (ambient - temperatures[-1])
        sum_boundary_heat += dt * (q_f + q_b)
        second_gas, second_heat = react_half()
        # Report surface values at the end of the full split step.
        front = temperatures[0] + (source_f + h_f * (ambient - temperatures[0])) * resistance
        back = temperatures[-1] + (source_b + h_b * (ambient - temperatures[-1])) * resistance
        loss = (first_gas + second_gas) * p.density * dx
        vented += loss
        sum_reaction_heat += (first_heat + second_heat) * p.density * dx
        loss_rate = 1000 * loss / dt
        peak_loss = max(peak_loss, loss_rate)
        if network is None:
            remaining = initial_mass - sum(reacted) * (1 - char) * p.density * dx
        else:
            remaining = sum(map(sum, network.mass)) * p.density * dx
        max_mass_error = max(max_mass_error, abs(remaining + vented - initial_mass) / initial_mass)
        sensible = capacity * sum(temp - ambient for temp in temperatures)
        max_energy_error = max(max_energy_error, abs(sensible + sum_reaction_heat - sum_boundary_heat))
        if step % sample_every == 0 or step == steps:
            frames.append(frame(step * dt, loss_rate))

    return SlabResult(depth, tuple(frames), peak_loss, max_mass_error, max_energy_error,
                      kinetics.names if network is not None else ())


def export_slab(path, result):
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        composition_columns = ["raw_solid_pct", "intermediate_pct", "char_pct", "released_volatile_pct"] \
            if result.frames[0].composition else []
        if result.component_names:
            composition_columns = ["component_" + name + "_pct" for name in result.component_names]
            composition_columns.append("released_volatile_pct")
        writer.writerow(["time_s", "front_C", "back_C", "remaining_mass_pct", "mass_loss_g_m2_s"] +
                        [f"T_C_at_z_{depth:.6f}_mm" for depth in result.depth_mm[1:-1]] +
                        [f"solid_mass_pct_at_z_{depth:.6f}_mm" for depth in result.depth_mm[1:-1]] + composition_columns)
        for frame in result.frames:
            values = (frame.time, frame.front, frame.back, frame.remaining_mass, frame.loss_rate,
                      *frame.temperature[1:-1], *frame.residual, *frame.composition)
            writer.writerow([f"{value:.7f}" for value in values])


SLAB_HELP = """一维平板 · 惰性气氛
从受热面 z = 0 沿厚度离散。初始温度与环境温度相同。正面输入恒定的吸收热通量，两面分别与环境对流，并使用同一发射率计算长波辐射换热。背面无外部热通量。其余方向按一维假设处理。

「组分与反应表」可设置各凝聚相组分的初始质量分数和一阶 Arrhenius 反应。From 独立消耗，To 为凝聚相时其产率以所消耗质量的百分比表示，余量归入 gases。各组分共享平板热物性。在整体组分视图中可选组分或累计生成的 gases。

单步模式使用主窗口的热解参数和固定热物性。两步模式在「多步反应与材料」中选择并联或串联，各步独立设置动力学、固体产率和吸热量。中间产物与残炭可配置密度、比热和导热系数，原材料热物性取自平板左栏。

多步计算保持网格厚度固定。有效热容由各相质量与比热决定，导热系数用以密度换算的相体积加权，并估算孔隙气体的热物性。估算固相总体积超过网格容积时，按固相相对体积混合导热系数。挥发质量计入累计失重。本模块的计算气氛为惰性气氛。

温度用隐式有限体积法计算，辐射换热系数在每步更新，反应与传热分步推进。质量损失率是该计算步的平均值，单位按样品表面积归一。剖面横坐标包含两个表面及网格中心，局部残余质量对应网格中心。

测点位于受热面（0 mm）与背面之间，可输入深度或在剖面中点选。测点温度在现有网格值间线性插值，局部剩余质量在网格中心插值，两侧表面采用最近网格的值。可分别导出测点时间曲线和当前时刻的厚度剖面。

算例保存设置、测点、视图，以及与当前参数一致的计算结果。重新打开单步算例时，同步恢复主窗口中供平板使用的三个热解参数。

示例为合成参数。改变网格和步长可比较结果的数值收敛性。实际实验需用实测温度和质量损失验证参数及边界。"""


class ReactionTableEditor:
    """Edit both tables as a draft and apply them together after validation."""
    def __init__(self, slab):
        self.slab, self.tk, self.ttk = slab, slab.tk, slab.ttk
        tk, ttk = self.tk, self.ttk
        self.window = tk.Toplevel(slab.window)
        self.window.title("组分与反应表")
        self.window.geometry("940x530")
        self.window.minsize(725, 380)
        self.window.transient(slab.window)
        body = ttk.Frame(self.window, padding=12)
        body.pack(fill="both", expand=True)
        self.tabs = ttk.Notebook(body)
        self.tabs.pack(fill="both", expand=True)
        self.rows = {"components": [], "reactions": []}
        self.grids = {}
        for key, title, hint, labels in (
            ("components", "组分与初始质量分数",
             "每行一个凝聚相组分。质量分数填 0–1，总和须为 1。只作为产物出现的组分填 0。",
             ("组分名称", "初始质量分数")),
            ("reactions", "From → To 反应",
             "From、To 取组分表名称，纯气体产物写 gases 并将凝聚相产率填 0。其余产率按消耗质量计，余量进入气相。Z 可用科学计数法。反应热正值为吸热。",
             ("From", "To", "Z (1/s)", "E (kJ/mol)", "凝聚相产率 (%)", "反应热 (kJ/kg)")),
        ):
            panel = ttk.Frame(self.tabs, padding=8)
            self.tabs.add(panel, text=title)
            slab.lab.text(panel, hint, bg=BG, fg=MUTED, wraplength=680, justify="left").pack(anchor="w")
            actions = ttk.Frame(panel)
            actions.pack(fill="x", pady=5)
            ttk.Button(actions, text="添加一行", command=lambda k=key: self.add(k)).pack(side="left")
            ttk.Button(actions, text="粘贴多行", command=lambda k=key: self.paste(k)).pack(side="left", padx=6)
            area = ttk.Frame(panel)
            area.pack(fill="both", expand=True)
            canvas = tk.Canvas(area, bg=BG, highlightthickness=0)
            scrollbar = ttk.Scrollbar(area, orient="vertical", command=canvas.yview)
            scrollbar.pack(side="right", fill="y")
            canvas.configure(yscrollcommand=scrollbar.set)
            canvas.pack(fill="both", expand=True)
            grid = ttk.Frame(canvas)
            item = canvas.create_window(0, 0, anchor="nw", window=grid)
            grid.bind("<Configure>", lambda _, c=canvas: c.configure(scrollregion=c.bbox("all")))
            canvas.bind("<Configure>", lambda event, c=canvas, item=item: c.itemconfigure(item, width=event.width))
            for col, label in enumerate(labels):
                grid.columnconfigure(col, weight=1)
                ttk.Label(grid, text=label).grid(row=0, column=col, sticky="w")
            self.grids[key] = grid
        self.status = tk.StringVar(value="填写后点击「应用到平板」。可从电子表格复制与表头同序的数据行。")
        slab.lab.text(body, value=self.status, bg=BG, fg=MUTED, wraplength=680).pack(anchor="w", pady=6)
        actions = ttk.Frame(body)
        actions.pack(fill="x")
        ttk.Button(actions, text="应用到平板", command=self.apply, style="Primary.TButton").pack(side="right")
        ttk.Button(actions, text="取消", command=self.window.destroy).pack(side="right", padx=6)
        model = slab.mechanism
        if isinstance(model, ReactionNetwork):
            for item in model.components:
                self.add("components", (item.name, item.initial))
            for r in model.reactions:
                self.add("reactions", (r.source, r.target, r.prefactor, r.energy, r.yield_pct, r.heat))
        else:
            self.add("components")
            self.add("reactions")

    def add(self, key, values=None):
        grid = self.grids[key]
        row = grid.grid_size()[1]
        if values is None:
            values = ("", "") if key == "components" else ("", "", "", "", "", "")
        widgets = []
        for col, value in enumerate(values):
            field = self.ttk.Entry(grid, width=10)
            field.grid(row=row, column=col, sticky="ew", padx=(0, 4), pady=2)
            field.insert(0, str(value))
            widgets.append(field)
        delete = self.ttk.Button(grid, text="删除", width=5, command=lambda: self.remove(key, widgets))
        delete.grid(row=row, column=len(values), padx=2)
        widgets.append(delete)
        self.rows[key].append(widgets)

    def remove(self, key, widgets):
        self.rows[key].remove(widgets)
        for widget in widgets:
            widget.destroy()

    def paste(self, key):
        try:
            lines = self.window.clipboard_get().splitlines()
        except self.tk.TclError:
            self.status.set("剪贴板中没有文本。")
            return
        rows = [line.split("\t") for line in lines if line.strip()]
        columns = 2 if key == "components" else 6
        if not rows or any(len(row) != columns for row in rows):
            self.status.set(f"复制数据行时请按本表顺序选择 {columns} 列，使用制表符分隔。")
            return
        for widgets in self.rows[key][:]:
            if all(not field.get().strip() for field in widgets[:-1]):
                self.remove(key, widgets)
        for row in rows:
            self.add(key, [value.strip() for value in row])
        self.status.set(f"已添加 {len(rows)} 行。")

    def read_rows(self, key, cls, text_columns):
        records = []
        for index, widgets in enumerate(self.rows[key], 1):
            values = [field.get().strip() for field in widgets[:-1]]
            try:
                numbers = [float(value) for value in values[text_columns:]]
            except ValueError as error:
                name = "组分" if key == "components" else "反应"
                raise ValueError(f"{name}第 {index} 行存在空白或无效数字。") from error
            records.append(cls(*values[:text_columns], *numbers))
        return tuple(records)

    def apply(self):
        try:
            network = ReactionNetwork(self.read_rows("components", Component, 1),
                                      self.read_rows("reactions", NetworkReaction, 2))
            network.validate()
        except ValueError as error:
            self.status.set(str(error))
            return
        self.slab.mechanism = network
        self.slab.refresh_label()
        self.slab.mark_dirty()
        self.window.destroy()


class MechanismEditor:
    LABELS = {"单步（主窗口参数）": None, "两步并联": "parallel", "两步串联": "series", "自定义反应表": "network"}

    def __init__(self, slab):
        self.slab = slab
        tk, ttk, lab = slab.tk, slab.ttk, slab.lab
        self.window = tk.Toplevel(slab.window)
        self.window.title("多步反应与材料属性")
        self.window.transient(slab.window)
        self.window.geometry("760x535")
        self.window.minsize(650, 470)
        body = ttk.Frame(self.window, padding=14)
        body.pack(fill="both", expand=True)
        self.status = tk.StringVar()
        current = slab.mechanism if isinstance(slab.mechanism, SlabMechanism) else SlabMechanism()
        mode = ("network" if isinstance(slab.mechanism, ReactionNetwork)
                else slab.mechanism.mode if slab.mechanism else None)
        self.mode = tk.StringVar(value=next(name for name, value in self.LABELS.items() if value == mode))
        header = ttk.Frame(body)
        header.pack(fill="x")
        lab.text(header, "反应模式", bg=BG, size=12, bold=True).pack(side="left", padx=(0, 9))
        choice = ttk.Combobox(header, textvariable=self.mode, values=tuple(self.LABELS), state="readonly", width=20)
        choice.pack(side="left")
        choice.bind("<<ComboboxSelected>>", self.update)
        self.hint = tk.StringVar()
        lab.text(body, value=self.hint, bg=BG, fg=MUTED, justify="left", wraplength=700).pack(
            anchor="w", pady=(7, 9))

        data = ttk.Frame(body)
        data.pack(fill="x")
        self.groups = {}
        specs = (
            ("first", "反应 1", REACTION_FIELDS),
            ("second", "反应 2", REACTION_FIELDS),
            ("intermediate", "中间产物热物性", PROPERTY_FIELDS),
            ("char", "残炭热物性", PROPERTY_FIELDS),
        )
        for index, (key, label, fields) in enumerate(specs):
            panel = ttk.Labelframe(data, text=label, padding=7)
            panel.grid(row=index // 2, column=index % 2, sticky="nsew", padx=4, pady=4)
            data.columnconfigure(index % 2, weight=1, uniform="mechanism")
            entries = {}
            for row, (name, title, _, _) in enumerate(fields):
                var = tk.StringVar(value=f"{getattr(getattr(current, key), name):g}")
                lab.text(panel, title, bg=BG, size=9).grid(row=row, column=0, sticky="w", padx=(0, 4))
                entry = ttk.Entry(panel, textvariable=var, width=8, justify="right")
                entry.grid(row=row, column=1, sticky="e", pady=1)
                entries[name] = (var, entry)
            self.groups[key] = entries
        lab.text(body, value=self.status, bg=BG, fg=AIR, wraplength=710,
                 justify="left", size=9).pack(anchor="w", pady=(4, 0))
        footer = ttk.Frame(body)
        footer.pack(side="bottom", fill="x")
        ttk.Button(footer, text="应用到平板", command=self.apply, style="Primary.TButton").pack(side="right")
        ttk.Button(footer, text="取消", command=self.window.destroy).pack(side="right", padx=6)
        self.update()

    def update(self, _event=None):
        mode = self.LABELS[self.mode.get()]
        descriptions = {
            None: "单步取自主窗口的热解参数。原材料热物性在平板左栏设置。",
            "network": "点击「应用到平板」进入逐行组分与反应编辑。",
            "parallel": "两条反应竞争消耗原材料，各自产生残炭和挥发物。原材料热物性取自平板左栏。",
            "series": "反应 1 生成中间产物，反应 2 消耗中间产物生成残炭。两步都可释放挥发物。",
        }
        self.hint.set(descriptions[mode])
        self.status.set("")
        for key, entries in self.groups.items():
            enabled = mode in ("parallel", "series") and (key != "intermediate" or mode == "series")
            for _, entry in entries.values():
                entry.configure(state="normal" if enabled else "disabled")

    def apply(self):
        mode = self.LABELS[self.mode.get()]
        if mode == "network":
            self.window.destroy()
            self.slab.open_table()
            return
        mechanism = None
        if mode:
            def params(key, cls, label):
                try:
                    return cls(**{name: float(var.get()) for name, (var, _) in self.groups[key].items()})
                except ValueError as error:
                    raise ValueError(f"{label}请输入有效数值。") from error
            try:
                intermediate = (params("intermediate", SolidProperties, "中间产物")
                    if mode == "series" else SlabMechanism().intermediate)
                mechanism = SlabMechanism(mode,
                    params("first", ReactionStep, "反应 1"), params("second", ReactionStep, "反应 2"),
                    intermediate, params("char", SolidProperties, "残炭"))
                validate_mechanism(mechanism)
            except ValueError as error:
                self.status.set(str(error))
                return
        self.slab.mechanism = mechanism
        self.slab.refresh_label()
        self.slab.mark_dirty()
        self.window.destroy()


@dataclass(frozen=True)
class Observation:
    time: float
    values: tuple[float | None, ...]


@dataclass(frozen=True)
class SlabExperiment:
    name: str
    parameters: SlabParameters
    observations: tuple[Observation, ...]
    validation: bool = False


OBSERVABLES = (
    ("remaining_mass_pct", "剩余质量 (%)", "remaining_mass", 1.0),
    ("mass_loss_g_m2_s", "失重率 (g/m²/s)", "loss_rate", .1),
    ("front_C", "受热面温度 (°C)", "front", 1.0),
    ("back_C", "背面温度 (°C)", "back", 1.0),
)


def read_experiment(path, parameters):
    path = Path(path)
    if path.stat().st_size > 8_000_000:
        raise ValueError("单个实验 CSV 不能超过 8 MB。")
    with path.open(encoding="utf-8-sig", newline="") as file:
        head = file.read(4096)
        file.seek(0)
        try:
            dialect = csv.Sniffer().sniff(head, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(file, dialect=dialect)
        names = {name.strip().lower(): name for name in reader.fieldnames or ()}
        columns = tuple(names.get(key.lower()) for key, *_ in OBSERVABLES)
        if "time_s" not in names or not any(columns):
            raise ValueError("CSV 需有 time_s 和至少一个受支持的观测列。")
        observations = []
        for row in reader:
            if not any(row.values()):
                continue
            try:
                time = float(row[names["time_s"]])
                raw = tuple((row.get(column) or "").strip() if column else "" for column in columns)
                values = tuple(float(value) if value else None for value in raw)
                if (not math.isfinite(time) or time < 0 or
                    any(value is not None and not math.isfinite(value) for value in values) or
                    (observations and time <= observations[-1].time)):
                    raise ValueError()
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path.name} 第 {reader.line_num} 行有无效数值，或时间未严格递增。") from error
            if any(value is not None for value in values):
                observations.append(Observation(time, values))
            if len(observations) > 5000:
                raise ValueError("单个实验至多 5,000 个有效观测时刻。")
    if len(observations) < 2:
        raise ValueError("实验需至少两个有观测值的时刻。")
    parameters = replace(parameters, duration=max(1, observations[-1].time))
    validate_slab(parameters)
    return SlabExperiment(path.name, parameters, tuple(observations))


def inverse_fields(kinetics):
    fields = [
        ("density", "原材料密度 (kg/m³)", 20, 10000),
        ("conductivity", "原材料导热系数 (W/m·K)", .01, 400),
        ("heat_capacity", "原材料比热 (J/kg·K)", 300, 6000),
    ]
    if isinstance(kinetics, ReactionNetwork):
        for i, reaction in enumerate(kinetics.reactions):
            prefix = f"反应 {i + 1} · {reaction.source} → {reaction.target}"
            fields.extend(((f"{i}.energy", prefix + " · E (kJ/mol)", 0, max(500, reaction.energy * 2)),
                           (f"{i}.log_z", prefix + " · log₁₀Z", -300, 300)))
    elif isinstance(kinetics, SlabMechanism):
        for side, name in (("first", "反应 1"), ("second", "反应 2")):
            fields.extend((f"{side}.{key}", f"{name} · {label}", low, high)
                          for key, label, low, high in REACTION_FIELDS)
    else:
        fields.extend((key, title, low, high) for key, title, _, low, high in FIELDS if key in FIT_KEYS)
    return tuple(fields)


def inverse_value(parameters, kinetics, key):
    if "." in key:
        side, field = key.split(".")
        if isinstance(kinetics, ReactionNetwork):
            reaction = kinetics.reactions[int(side)]
            return math.log10(reaction.prefactor) if field == "log_z" else getattr(reaction, field)
        return getattr(getattr(kinetics, side), field)
    if key in FIT_KEYS:
        return kinetics[FIT_KEYS.index(key)]
    return getattr(parameters, key)


def with_inverse_values(parameters, kinetics, keys, values):
    changes = {}
    selected = dict(zip(keys, values, strict=True))
    for key in ("density", "conductivity", "heat_capacity"):
        if key in selected:
            changes[key] = selected[key]
    if isinstance(kinetics, ReactionNetwork):
        rows = list(kinetics.reactions)
        for key, value in selected.items():
            if "." in key:
                index, field = key.split(".")
                i = int(index)
                change = {"prefactor": 10 ** value} if field == "log_z" else {field: value}
                rows[i] = replace(rows[i], **change)
        kinetics = replace(kinetics, reactions=tuple(rows))
    elif isinstance(kinetics, SlabMechanism):
        replacements = {}
        for side in ("first", "second"):
            reaction = {key.split(".")[1]: value for key, value in selected.items() if key.startswith(side + ".")}
            if reaction:
                replacements[side] = replace(getattr(kinetics, side), **reaction)
        kinetics = replace(kinetics, **replacements)
    else:
        kinetics = tuple(selected.get(key, value) for key, value in zip(FIT_KEYS, kinetics))
    return replace(parameters, **changes), kinetics


def aligned_predictions(frames, observations):
    index = 0
    for observed in observations:
        while index + 1 < len(frames) - 1 and frames[index + 1].time < observed.time:
            index += 1
        start, end = frames[index], frames[index + 1]
        x = min(1.0, max(0.0, (observed.time - start.time) / (end.time - start.time)))
        yield tuple((1 - x) * getattr(start, field) + x * getattr(end, field)
                    for _, _, field, _ in OBSERVABLES)


def evaluate_case(case, kinetics, keys, values, cancelled):
    parameters, reactions = with_inverse_values(case.parameters, kinetics, keys, values)
    result = slab_simulate(parameters, reactions, cancelled, profiles=False)
    prediction = tuple(aligned_predictions(result.frames, case.observations))
    errors = [0.0] * len(OBSERVABLES)
    counts = [0] * len(OBSERVABLES)
    for observed, fitted in zip(case.observations, prediction):
        for index, (value, calculated) in enumerate(zip(observed.values, fitted)):
            if value is not None:
                errors[index] += (value - calculated) ** 2
                counts[index] += 1
    return tuple(math.sqrt(v / n) if n else None for v, n in zip(errors, counts)), prediction


def training_scales(training):
    columns = [[] for _ in OBSERVABLES]
    for case in training:
        for row in case.observations:
            for column, value in zip(columns, row.values):
                if value is not None:
                    column.append(value)
    return tuple(max(floor, max(column) - min(column)) if column else None
                 for column, (*_, floor) in zip(columns, OBSERVABLES))


def weighted_score(errors, weights, scales):
    mean_squares = [0.0] * len(OBSERVABLES)
    counts = [0] * len(OBSERVABLES)
    for case in errors:
        for index, value in enumerate(case):
            if value is not None and scales[index] is not None and weights[index] > 0:
                mean_squares[index] += (value / scales[index]) ** 2
                counts[index] += 1
    total = sum(weight for weight, count in zip(weights, counts) if count)
    if not total:
        return None
    return math.sqrt(sum(w * s / n for w, s, n in zip(weights, mean_squares, counts) if n) / total)


@dataclass(frozen=True)
class InverseResult:
    fit: FitResult
    validation_score: float | None
    case_results: tuple


def fit_experiments(cases, kinetics, specs, weights, method, budget, seed, cancelled, report):
    train = tuple(case for case in cases if not case.validation)
    held = tuple(case for case in cases if case.validation)
    if not train or not held:
        raise ValueError("需要至少一个训练工况和一个留出验证工况。")
    if any(a.observations == b.observations for a in train for b in held):
        raise ValueError("训练和验证需使用不同的实验数据。")
    scales = training_scales(train)
    if not any(weight > 0 and scale is not None for weight, scale in zip(weights, scales)):
        raise ValueError("训练工况缺少有正权重的观测。")
    if not specs:
        raise ValueError("请至少选择一个待拟合参数。")
    keys = tuple(key for key, _, _ in specs)

    def objective(values):
        try:
            errors = (evaluate_case(case, kinetics, keys, values, cancelled)[0] for case in train)
            return weighted_score(errors, weights, scales)
        except (ValueError, ArithmeticError):
            return float("inf")

    fit = optimize_parameters(objective, tuple((lo, hi) for _, lo, hi in specs),
                              method, budget, seed, cancelled, report)
    case_results = []
    for case in cases:
        try:
            errors, predictions = evaluate_case(case, kinetics, keys, fit.values, cancelled)
        except (ValueError, ArithmeticError) as error:
            if not case.validation:
                raise
            case_results.append((None, None, str(error)))
        else:
            case_results.append((errors, predictions, ""))
    validation_errors = [errors for case, (errors, _, _) in zip(cases, case_results)
                         if case.validation and errors is not None]
    validation_score = (weighted_score(validation_errors, weights, scales)
                        if len(validation_errors) == len(held) else None)
    return InverseResult(fit, validation_score, tuple(case_results))


def write_experiment_template(path):
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        csv.writer(file).writerow(["time_s", *(key for key, *_ in OBSERVABLES)])


@dataclass(frozen=True)
class ComparisonPoint:
    time: float
    observed: float | None
    predicted: float

    @property
    def residual(self):
        return None if self.observed is None else self.predicted - self.observed


def comparison_points(case, predictions, metric):
    return tuple(ComparisonPoint(row.time, row.values[metric], fitted[metric])
                 for row, fitted in zip(case.observations, predictions, strict=True))


def comparison_statistics(points):
    residuals = [p.residual for p in points if p.observed is not None]
    if not residuals:
        return 0, None, None, None
    n = len(residuals)
    return (n, math.sqrt(math.fsum(r * r for r in residuals) / n),
            math.fsum(abs(r) for r in residuals) / n, math.fsum(residuals) / n)


def export_comparison(path, metric, points):
    key = OBSERVABLES[metric][0]
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["time_s", "observed_" + key, "predicted_" + key, "residual_" + key])
        writer.writerows((p.time, p.observed, p.predicted, p.residual) for p in points)


class ComparisonWindow:
    """A snapshot of a single case and model, aligned to the experimental times."""

    def __init__(self, inverse, case, predictions, method):
        self.lab, self.case, self.predictions = inverse.lab, case, predictions
        tk, ttk = inverse.tk, inverse.ttk
        self.window = tk.Toplevel(inverse.window)
        self.window.title(f"实验对比 · {case.name} · {method}")
        self.window.geometry("860x520")
        self.window.minsize(600, 380)
        panel = ttk.Frame(self.window, padding=12)
        panel.pack(fill="both", expand=True)
        role = "留出验证" if case.validation else "训练工况"
        self.lab.text(panel, f"{case.name} · {role} · {method}", bg=BG, size=13, bold=True,
                      wraplength=660).pack(anchor="w", pady=(0, 8))
        bar = ttk.Frame(panel)
        bar.pack(fill="x")
        self.metrics = [i for i in range(len(OBSERVABLES))
                        if any(obs.values[i] is not None for obs in case.observations)]
        self.choice = ttk.Combobox(bar, values=[OBSERVABLES[i][1] for i in self.metrics],
                                   state="readonly", width=24)
        self.choice.current(0)
        self.choice.pack(side="left")
        self.choice.bind("<<ComboboxSelected>>", self.select)
        self.mode = tk.StringVar(value="overlay")
        for label, mode in (("实测与预测", "overlay"), ("残差", "residual")):
            ttk.Radiobutton(bar, text=label, value=mode, variable=self.mode,
                            command=self.draw).pack(side="left", padx=(6, 0))
        ttk.Button(bar, text="导出当前指标 CSV", command=self.save).pack(side="right")
        self.summary, self.readout = tk.StringVar(), tk.StringVar()
        self.lab.text(panel, value=self.summary, bg=BG, size=10, fg=ACCENT).pack(anchor="w", pady=(8, 4))
        self.canvas = tk.Canvas(panel, bg=PANEL, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.lab.text(panel, value=self.readout, bg=BG, size=9, fg=MUTED).pack(anchor="w", pady=(6, 0))
        self.times = tuple(row.time for row in case.observations)
        self.canvas.bind("<Configure>", self.draw)
        self.canvas.bind("<Motion>", self.inspect)
        self.canvas.bind("<Leave>", self.clear_cursor)
        self.select()

    def select(self, _event=None):
        self.metric = self.metrics[self.choice.current()]
        self.points = comparison_points(self.case, self.predictions, self.metric)
        n, rmse, mae, bias = comparison_statistics(self.points)
        self.summary.set(f"有效观测 {n} 个    RMSE {rmse:.4g}    平均绝对误差 {mae:.4g}    平均偏差 {bias:+.4g}")
        self.draw()

    def clear_cursor(self, _event=None):
        self.canvas.delete("cursor")
        self.readout.set("移动鼠标可读取实验时刻的数值。残差 = 预测 − 实测。")

    def draw(self, _event=None):
        c = self.canvas
        c.delete("all")
        self.clear_cursor()
        left, right, top, bottom = 78, c.winfo_width() - 28, 37, c.winfo_height() - 47
        self.box = None
        if right <= left or bottom <= top:
            return
        self.box = (left, right, top, bottom)
        residual = self.mode.get() == "residual"
        measured = [p for p in self.points if p.observed is not None]
        if residual:
            limit = max(abs(p.residual) for p in measured) or 1.0
            low, high = -1.05 * limit, 1.05 * limit
        else:
            values = [p.predicted for p in self.points] + [p.observed for p in measured]
            low, high = min(values), max(values)
            pad = (high - low) * .06 or max(abs(low) * .01, 1e-3)
            low, high = low - pad, high + pad
        start, end = self.times[0], self.times[-1]

        def px(t):
            return left + (t - start) / (end - start) * (right - left)

        def py(v):
            return bottom - (v - low) / (high - low) * (bottom - top)

        c.create_text(9, 14, text=OBSERVABLES[self.metric][1], fill=INK, anchor="w")
        legend = "点 = 预测 − 实测" if residual else "点 = 实测    线 = 预测"
        c.create_text(right, 14, text=legend, fill=MUTED, anchor="e")
        for tick in range(5):
            f = tick / 4
            x, y = left + f * (right - left), bottom - f * (bottom - top)
            c.create_line(left, y, right, y, fill=LINE)
            c.create_text(left - 6, y, text=f"{low + f * (high - low):.4g}", anchor="e", fill=MUTED)
            c.create_text(x, bottom + 12, text=f"{start + f * (end - start):.4g}", fill=MUTED)
        c.create_text((left + right) / 2, bottom + 32, text="时间 (s)", fill=MUTED)
        if residual:
            c.create_line(left, py(0), right, py(0), fill=FURNACE, dash=(5, 3))
        else:
            coords = [v for p in self.points for v in (px(p.time), py(p.predicted))]
            c.create_line(*coords, fill=ACCENT, width=2)
        for p in measured:
            x, y = px(p.time), py(p.residual if residual else p.observed)
            c.create_oval(x - 2.5, y - 2.5, x + 2.5, y + 2.5, fill=AIR, outline="")

    def inspect(self, event):
        if not self.box:
            return
        left, right, top, bottom = self.box
        if not (left <= event.x <= right and top <= event.y <= bottom):
            self.clear_cursor()
            return
        start, end = self.times[0], self.times[-1]
        t = start + (event.x - left) / (right - left) * (end - start)
        j = min(bisect.bisect_left(self.times, t), len(self.times) - 1)
        if j and abs(self.times[j-1] - t) <= abs(self.times[j] - t):
            j -= 1
        p = self.points[j]
        x = left + (p.time - start) / (end - start) * (right - left)
        self.canvas.delete("cursor")
        self.canvas.create_line(x, top, x, bottom, fill=MUTED, dash=(2, 3), tags="cursor")
        measured = "缺测" if p.observed is None else f"{p.observed:.5g}"
        delta = "—" if p.residual is None else f"{p.residual:+.5g}"
        self.readout.set(f"{p.time:g} s    实测 {measured}    预测 {p.predicted:.5g}    残差 {delta}")

    def save(self):
        path = self.lab.dialog.asksaveasfilename(parent=self.window, defaultextension=".csv",
            initialfile=f"comparison_{OBSERVABLES[self.metric][0]}.csv", filetypes=[("CSV", "*.csv")])
        if path:
            try:
                export_comparison(path, self.metric, self.points)
            except OSError as error:
                self.readout.set(f"导出失败 · {error}")
            else:
                self.readout.set(f"已导出当前指标 · {Path(path).name}")


class InverseWindow:
    def __init__(self, slab, parameters, kinetics):
        self.slab, self.lab = slab, slab.lab
        self.tk, self.ttk = slab.tk, slab.ttk
        tk, ttk = self.tk, self.ttk
        self.parameters, self.kinetics = parameters, kinetics
        self.last_applied = kinetics
        self.cases = []
        self.results = ()
        self.specs = ()
        self.busy = False
        self.queue = queue.Queue()
        self.stop = threading.Event()
        self.window = tk.Toplevel(slab.window)
        self.window.title("多指标反演与独立验证")
        self.window.geometry("1060x720")
        self.window.minsize(810, 620)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        root = ttk.Frame(self.window, padding=12)
        root.pack(fill="both", expand=True)
        self.lab.text(root, "多指标反演与独立验证", bg=BG, size=18, bold=True).pack(anchor="w")
        self.status = tk.StringVar(value="先添加实验工况，并指定用于独立验证的数据。")
        self.lab.text(root, value=self.status, bg=BG, fg=MUTED, size=9,
                      justify="left", wraplength=970).pack(anchor="w", pady=(4, 6))
        self.tabs = ttk.Notebook(root)
        self.tabs.pack(fill="both", expand=True)
        self.case_tab = self.tab("1 · 实验工况")
        self.fit_tab = self.tab("2 · 参数与权重")
        self.result_tab = self.tab("3 · 结果与验证")
        self.draw_cases()
        self.draw_settings()
        self.draw_results()
        self.controls()
        self.timer = self.window.after(100, self.poll)

    def tab(self, name):
        child = self.ttk.Frame(self.tabs, padding=10)
        self.tabs.add(child, text=name)
        return child

    def tree(self, parent, columns, labels, height):
        tree = self.ttk.Treeview(parent, columns=columns, show="headings", height=height, selectmode="browse")
        for key, label in zip(columns, labels):
            tree.heading(key, text=label)
            tree.column(key, width=110, minwidth=56, anchor="center")
        return tree

    def note(self, parent, text):
        self.lab.text(parent, text, bg=BG, fg=MUTED, size=9, justify="left",
                      wraplength=790).pack(anchor="w", pady=(3, 6))

    def draw_cases(self):
        ttk = self.ttk
        bar = ttk.Frame(self.case_tab)
        bar.pack(fill="x")
        self.case_buttons = []
        for label, action in (("导入实验 CSV", self.load), ("加载合成示例", self.demo),
                              ("切换训练 / 验证", self.toggle), ("移除选中工况", self.remove)):
            button = ttk.Button(bar, text=label, command=action)
            button.pack(side="left", padx=(0, 6))
            self.case_buttons.append(button)
        self.note(self.case_tab, "导入前在一维平板左栏填好该实验的厚度、边界、热物性初值和网格。导入会保留当时的工况设置，模拟时长取最后观测时刻（至少 1 s）。")
        self.note(self.case_tab, "表头须有 time_s，观测列任选 remaining_mass_pct、mass_loss_g_m2_s、front_C、back_C；缺测可留空。剩余质量为初始质量百分比，失重率为正向 g/(m²·s)，温度为 °C。")
        self.case_list = self.tree(self.case_tab, ("name", "role", "thickness", "flux", "points"),
                                   ("实验", "用途", "厚度 (mm)", "热通量 (kW/m²)", "观测时刻数"), 12)
        self.case_list.column("name", width=230)
        self.case_list.pack(fill="both", expand=True)
        self.case_list.bind("<<TreeviewSelect>>", lambda _: self.controls())
        actions = ttk.Frame(self.case_tab)
        actions.pack(fill="x", pady=(6, 0))
        self.preview_button = ttk.Button(actions, text="用当前材料对比", command=self.preview_case)
        self.preview_button.pack(side="left", padx=(0, 6))
        self.preview_stop = ttk.Button(actions, text="停止计算", command=self.stop.set)
        self.preview_stop.pack(side="left")
        ttk.Button(actions, text="导出实验 CSV 模板", command=self.save_template).pack(side="right")

    def draw_settings(self):
        tk, ttk = self.tk, self.ttk
        bar = ttk.Frame(self.fit_tab)
        bar.pack(fill="x", pady=(0, 6))
        self.method = tk.StringVar(value="GA 与 SCE-UA")
        self.budget = tk.StringVar(value="200")
        self.seed = tk.StringVar(value="13")
        ttk.Combobox(bar, textvariable=self.method, values=("GA 与 SCE-UA", *METHODS),
                     state="readonly", width=14).pack(side="left", padx=(0, 7))
        for name, value in (("每种算法评价次数", self.budget), ("种子", self.seed)):
            self.lab.text(bar, name, bg=BG, size=9).pack(side="left", padx=(0, 3))
            ttk.Entry(bar, textvariable=value, width=6).pack(side="left", padx=(0, 9))
        self.start_button = ttk.Button(bar, text="开始反演", command=self.start, style="Primary.TButton")
        self.start_button.pack(side="left", padx=(0, 5))
        self.stop_button = ttk.Button(bar, text="停止", command=self.stop.set)
        self.stop_button.pack(side="left")
        weight_panel = ttk.Labelframe(self.fit_tab, text="指标权重", padding=7)
        weight_panel.pack(fill="x", pady=(0, 5))
        self.weights = []
        for idx, (_, label, _, _) in enumerate(OBSERVABLES):
            var = tk.StringVar(value="1")
            self.weights.append(var)
            text = label.split(" (")[0]
            col, row = (idx % 2) * 2, idx // 2
            self.lab.text(weight_panel, text, bg=BG, size=9).grid(row=row, column=col, sticky="w", padx=(0, 5))
            ttk.Entry(weight_panel, textvariable=var, width=7).grid(row=row, column=col + 1, padx=(0, 18), pady=1)
        self.note(self.fit_tab, "各指标按训练集观测范围归一化，工况等权；权重填 0 可停用该指标。验证沿用训练归一化系数，只在反演完成后计算。")
        scroll = tk.Canvas(self.fit_tab, bg=BG, highlightthickness=0)
        track = ttk.Scrollbar(self.fit_tab, orient="vertical", command=scroll.yview)
        track.pack(side="right", fill="y")
        scroll.configure(yscrollcommand=track.set)
        scroll.pack(side="left", fill="both", expand=True)
        bounds = ttk.Frame(scroll)
        scroll.create_window(0, 0, window=bounds, anchor="nw")
        bounds.bind("<Configure>", lambda _: scroll.configure(scrollregion=scroll.bbox("all")))
        for col, title in enumerate(("拟合参数", "下界", "上界")):
            self.lab.text(bounds, title, bg=BG, bold=True, size=9).grid(row=0, column=col, sticky="w", padx=(0, 8))
        self.bound_rows = {}
        for row, (key, label, lo, hi) in enumerate(inverse_fields(self.kinetics), 1):
            current = inverse_value(self.parameters, self.kinetics, key)
            span = abs(current) * .4 if current else (hi - lo) * .1
            if key.endswith(".log_z"):
                span = 1.0  # Search one decade either side of the supplied rate factor.
            low, high = max(lo, current - span), min(hi, current + span)
            enabled = tk.BooleanVar(value=key in ("conductivity", "heat_capacity"))
            bottom, top = tk.StringVar(value=f"{low:.7g}"), tk.StringVar(value=f"{high:.7g}")
            ttk.Checkbutton(bounds, variable=enabled, text=label).grid(row=row, column=0, sticky="w")
            ttk.Entry(bounds, textvariable=bottom, width=10).grid(row=row, column=1, padx=(0, 6), pady=1)
            ttk.Entry(bounds, textvariable=top, width=10).grid(row=row, column=2, pady=1)
            self.bound_rows[key] = (enabled, bottom, top, lo, hi)

    def draw_results(self):
        ttk = self.ttk
        self.note(self.result_tab, "训练分数衡量拟合误差。验证分数使用同一套参数对留出工况单独计算；下表 RMSE 保留各指标原单位。")
        self.methods = self.tree(self.result_tab, ("method", "train", "valid", "eval"),
                                 ("算法", "训练分数", "独立验证分数", "评价次数"), 2)
        self.methods.pack(fill="x", pady=(0, 4))
        self.methods.bind("<<TreeviewSelect>>", self.show_method)
        self.metrics = self.tree(self.result_tab, ("name", "role", "mass", "rate", "front", "back"),
                                 ("工况", "用途", "余量 RMSE (%)", "失重率 RMSE", "正面 RMSE (°C)", "背面 RMSE (°C)"), 7)
        self.metrics.column("name", width=210)
        self.metrics.pack(fill="both", expand=True, pady=(0, 4))
        self.parameters_readout = self.tk.StringVar()
        self.lab.text(self.result_tab, value=self.parameters_readout, bg=BG, justify="left",
                      wraplength=805, size=9).pack(anchor="w", pady=5)
        bar = ttk.Frame(self.result_tab)
        bar.pack(fill="x")
        self.apply_button = ttk.Button(bar, text="回填所选算法的参数", command=self.apply)
        self.apply_button.pack(side="left", padx=(0, 6))
        self.export_button = ttk.Button(bar, text="导出训练和验证对比 CSV", command=self.save)
        self.export_button.pack(side="left")
        self.compare_button = ttk.Button(bar, text="查看工况曲线与残差", command=self.compare_fit)
        self.compare_button.pack(side="left", padx=(6, 0))
        self.metrics.bind("<<TreeviewSelect>>", lambda _: self.controls())
        self.metrics.bind("<Double-1>", self.compare_row)

    def close(self):
        self.stop.set()
        self.window.after_cancel(self.timer)
        self.window.destroy()

    def controls(self):
        for button in self.case_buttons:
            button.configure(state="disabled" if self.busy else "normal")
        self.start_button.configure(state="disabled" if self.busy else "normal")
        self.stop_button.configure(state="normal" if self.busy else "disabled")
        state = "normal" if self.results and not self.busy else "disabled"
        self.preview_button.configure(state="normal" if self.case_list.selection() and not self.busy else "disabled")
        self.preview_stop.configure(state="normal" if self.busy else "disabled")
        selection = self.metrics.selection()
        report = self.selected()
        valid = report and selection and report.case_results[int(selection[0])][1] is not None
        self.compare_button.configure(state="normal" if valid and not self.busy else "disabled")
        self.apply_button.configure(state=state)
        self.export_button.configure(state=state)

    def clear_results(self):
        self.results = ()
        self.methods.delete(*self.methods.get_children())
        self.metrics.delete(*self.metrics.get_children())
        self.parameters_readout.set("")

    def cases_changed(self):
        self.case_list.delete(*self.case_list.get_children())
        for i, case in enumerate(self.cases):
            self.case_list.insert("", "end", iid=str(i), values=(case.name, "验证" if case.validation else "训练",
                f"{case.parameters.thickness:g}", f"{case.parameters.flux:g}", len(case.observations)))
        self.clear_results()
        self.controls()

    def load(self):
        path = self.lab.dialog.askopenfilename(parent=self.window, filetypes=[("实验 CSV", "*.csv"), ("所有文件", "*")])
        if path:
            try:
                case = read_experiment(path, self.slab.parameter_snapshot())
            except (OSError, UnicodeError, ValueError, csv.Error) as error:
                self.status.set(f"导入失败 · {error}")
            else:
                self.cases.append(case)
                self.cases_changed()
                self.case_list.selection_set(str(len(self.cases) - 1))
                self.status.set("已添加训练工况。可选中它切换为验证工况。")

    def save_template(self):
        path = self.lab.dialog.asksaveasfilename(parent=self.window, defaultextension=".csv",
            initialfile="slab_experiment.csv", filetypes=[("CSV", "*.csv")])
        if path:
            try:
                write_experiment_template(path)
            except OSError as error:
                self.status.set(f"模板保存失败 · {error}")
            else:
                self.status.set("模板已导出。填入至少两个递增时刻和对应实测值，缺测留空。")

    def preview_case(self):
        selection = self.case_list.selection()
        if self.busy or not selection:
            return
        case = self.cases[int(selection[0])]
        try:
            p, kinetics = self.slab.parameter_snapshot(), self.slab.current_kinetics()
        except ValueError as error:
            self.status.set(f"平板参数有误 · {error}")
            return
        # Reuse the imported experiment's boundary conditions, time span and mesh.
        material = replace(case.parameters, density=p.density, conductivity=p.conductivity,
                           heat_capacity=p.heat_capacity)
        case = replace(case, parameters=material)
        self.busy = True
        self.stop.clear()
        self.controls()
        self.status.set(f"正在按当前材料参数计算 · {case.name}")

        def worker():
            try:
                _, predictions = evaluate_case(case, kinetics, (), (), self.stop.is_set)
                if self.stop.is_set():
                    raise FitCancelled()
            except FitCancelled:
                self.queue.put(("error", "已停止实验对比。"))
            except (ValueError, ArithmeticError) as error:
                self.queue.put(("error", f"实验对比失败 · {error}"))
            else:
                self.queue.put(("preview", (case, predictions)))
        threading.Thread(target=worker, daemon=True).start()

    def toggle(self):
        if selection := self.case_list.selection():
            index = int(selection[0])
            self.cases[index] = replace(self.cases[index], validation=not self.cases[index].validation)
            self.cases_changed()
            self.case_list.selection_set(str(index))

    def remove(self):
        if selection := self.case_list.selection():
            self.cases.pop(int(selection[0]))
            self.cases_changed()

    def demo(self):
        p = self.parameters
        try:
            samples = []
            for holdout, flux, thickness, duration in ((False, 14, 4, 35), (True, 19, 6, 38)):
                settings = replace(p, thickness=thickness, flux=flux, duration=duration, cells=8, step=.5)
                truth = replace(settings, conductivity=min(400, settings.conductivity * 1.19),
                                heat_capacity=max(300, settings.heat_capacity * .88))
                forward = slab_simulate(truth, self.kinetics, profiles=False)
                observations = tuple(Observation(f.time, tuple(getattr(f, attr) for _, _, attr, _ in OBSERVABLES))
                                     for f in forward.frames[::2])
                samples.append(SlabExperiment("合成留出工况" if holdout else "合成训练工况",
                                               settings, observations, holdout))
        except (ValueError, ArithmeticError) as error:
            self.status.set(f"无法生成合成数据 · {error}")
            return
        # Preserve imported work. Re-create the demo only when no demo is present.
        if any(case.name in ("合成留出工况", "合成训练工况") for case in self.cases):
            self.status.set("列表中已有合成示例。")
            return
        self.cases.extend(samples)
        self.cases_changed()
        self.status.set("已添加两组合成数据。用于体验时，可保留默认的导热系数和比热拟合选项。")
        self.tabs.select(self.fit_tab)

    def start(self):
        if self.busy:
            return
        try:
            specs = []
            for key, (enabled, left, right, minimum, maximum) in self.bound_rows.items():
                if enabled.get():
                    lo, hi = float(left.get()), float(right.get())
                    if not minimum <= lo < hi <= maximum:
                        label = next(label for name, label, *_ in inverse_fields(self.kinetics) if name == key)
                        raise ValueError(f"{label}的上下界应位于 {minimum:g}–{maximum:g}，且下界小于上界。")
                    specs.append((key, lo, hi))
            weights = tuple(float(value.get()) for value in self.weights)
            if any(not math.isfinite(value) or not 0 <= value <= 1000 for value in weights):
                raise ValueError("指标权重应为 0–1000。")
            budget, seed = int(self.budget.get()), int(self.seed.get())
            if not 100 <= budget <= 4000:
                raise ValueError("每种算法须使用 100–4,000 次评价。")
        except ValueError as error:
            self.status.set(f"设置有误 · {error}")
            return
        method = self.method.get()
        methods = METHODS if method == "GA 与 SCE-UA" else (method,)
        cases = tuple(self.cases)
        self.specs = tuple(specs)
        self.clear_results()
        self.busy = True
        self.stop.clear()
        self.controls()
        self.status.set("正在反演，留出数据将在训练完成后单独计算。")

        def worker():
            try:
                results = []
                for method in methods:
                    results.append(fit_experiments(cases, self.kinetics, specs, weights, method, budget, seed,
                        self.stop.is_set, lambda count: self.queue.put(("progress", f"{method} · {count:,}/{budget:,} 次评价"))))
                self.queue.put(("done", tuple(results)))
            except FitCancelled:
                self.queue.put(("error", "已停止反演。"))
            except (ValueError, ArithmeticError) as error:
                self.queue.put(("error", f"反演失败 · {error}"))
        threading.Thread(target=worker, daemon=True).start()

    def poll(self):
        try:
            while True:
                event, data = self.queue.get_nowait()
                if event == "done":
                    self.finish(data)
                elif event == "preview":
                    self.busy = False
                    self.controls()
                    if self.stop.is_set():
                        self.status.set("已停止实验对比。")
                    else:
                        ComparisonWindow(self, *data, "当前材料")
                        self.status.set("对比完成。热物性和反应采用当前平板设置，边界与网格取自所选工况。")
                else:
                    self.status.set(data)
                    if event == "error":
                        self.busy = False
                        self.controls()
        except queue.Empty:
            pass
        self.timer = self.window.after(100, self.poll)

    @staticmethod
    def fmt(value):
        return "—" if value is None else f"{value:.5g}"

    def finish(self, results):
        self.results, self.busy = results, False
        for result in results:
            fit = result.fit
            self.methods.insert("", "end", iid=fit.method,
                values=(fit.method, self.fmt(fit.rmse), self.fmt(result.validation_score), fit.evaluations))
        self.methods.selection_set(min(results, key=lambda item: item.fit.rmse).fit.method)
        self.show_method()
        self.controls()
        self.tabs.select(self.result_tab)

    def selected(self):
        selection = self.methods.selection()
        return next((item for item in self.results if selection and item.fit.method == selection[0]), None)

    def show_method(self, _event=None):
        report = self.selected()
        if report is None:
            return
        selection = self.metrics.selection()
        self.metrics.delete(*self.metrics.get_children())
        failed = []
        for index, (case, (errors, _, failure)) in enumerate(zip(self.cases, report.case_results)):
            role = "验证" if case.validation else "训练"
            if failure:
                failed.append(f"{case.name}：{failure}")
            self.metrics.insert("", "end", iid=str(index), values=(case.name, role,
                *(self.fmt(value) for value in errors)) if errors else (case.name, role, "计算失败", "", "", ""))
        self.metrics.selection_set(selection[0] if selection else "0")
        self.controls()
        descriptions = {key: label for key, label, *_ in inverse_fields(self.kinetics)}
        self.parameters_readout.set("    ".join(f"{descriptions[key]}  {value:.6g}"
                                    for (key, _, _), value in zip(self.specs, report.fit.values)))
        self.status.set("反演与留出验证已完成。" if not failed else "验证未完成 · " + "；".join(failed))

    def compare_row(self, event):
        row = self.metrics.identify_row(event.y)
        if row:
            self.metrics.selection_set(row)
            self.compare_fit()

    def compare_fit(self):
        report, selection = self.selected(), self.metrics.selection()
        if self.busy or not report or not selection:
            return
        index = int(selection[0])
        _, predictions, failure = report.case_results[index]
        if predictions is None:
            self.status.set(f"此工况缺少预测 · {failure}")
            return
        ComparisonWindow(self, self.cases[index], predictions, report.fit.method)

    def apply(self):
        report = self.selected()
        if report is None:
            return
        try:
            if self.slab.current_kinetics() != self.last_applied:
                self.status.set("平板反应机制或动力学已修改。请核对模式后重新打开本窗口。")
                return
            keys = tuple(key for key, _, _ in self.specs)
            settings, reactions = with_inverse_values(self.slab.parameter_snapshot(), self.kinetics, keys, report.fit.values)
        except ValueError as error:
            self.status.set(str(error))
            return
        for key in ("density", "conductivity", "heat_capacity"):
            if key in keys:
                self.slab.inputs[key].set(f"{getattr(settings, key):.12g}")
        if isinstance(reactions, (SlabMechanism, ReactionNetwork)):
            self.slab.mechanism = reactions
        else:
            for key, value in zip(FIT_KEYS, reactions):
                if key in keys:
                    self.lab.inputs[key].set(f"{value:.12g}")
        self.last_applied = self.slab.current_kinetics()
        self.slab.refresh_label()
        self.slab.mark_dirty()
        self.status.set(f"已把 {report.fit.method} 的所选参数回填到平板。")

    def save(self):
        path = self.lab.dialog.asksaveasfilename(parent=self.window, defaultextension=".csv",
            initialfile="inverse_validation.csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            write_inverse_csv(path, self.cases, self.results)
        except OSError as error:
            self.status.set(f"导出失败 · {error}")
        else:
            self.status.set(f"已导出训练和验证对比 · {Path(path).name}")


def write_inverse_csv(path, cases, reports):
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["case", "role", "method", "time_s"] +
                        [prefix + key for prefix in ("observed_", "predicted_") for key, *_ in OBSERVABLES] + ["error"])
        for report in reports:
            for case, (_, predictions, error) in zip(cases, report.case_results):
                name = "'" + case.name if case.name.lstrip().startswith(("=", "-", "+", "@")) else case.name
                columns = (name, "validation" if case.validation else "training", report.fit.method)
                for index, row in enumerate(case.observations):
                    fitted = predictions[index] if predictions is not None else ("",) * len(OBSERVABLES)
                    writer.writerow([*columns, row.time, *row.values, *fitted, error])


@dataclass(frozen=True)
class SectionParameters:
    width: float = 30.0          # mm
    nx: int = 16
    nz: int = 16
    heated: float = 50.0         # centered strip, % of front width
    end_thickness: float = 80.0  # % initial thickness at complete virgin conversion
    porosity_raw: float = 30.0   # connected pore volume, %
    porosity_end: float = 60.0
    diffusivity: float = 2e-6    # unimpeded gas diffusivity, m²/s
    vent: str = "front"


SECTION_FIELDS = (
    ("width", "试样宽度 (mm)", 0.2, 500),
    ("nx", "宽向网格数", 3, 64),
    ("nz", "厚向网格数", 3, 64),
    ("heated", "正面加热宽度 (%)", 0, 100),
    ("end_thickness", "完全转化厚度 (% 初始)", 10, 500),
    ("porosity_raw", "初始连通孔隙率 (%)", 1, 99),
    ("porosity_end", "完全转化孔隙率 (%)", 1, 99),
    ("diffusivity", "自由气体扩散系数 (m²/s)", 0, 0.01),
)
VENTS = {"front": "正面开放", "back": "背面开放", "both": "两面开放", "closed": "两面封闭"}


def validate_section(s):
    for key, label, low, high in SECTION_FIELDS:
        value = getattr(s, key)
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{label}须在 {low:g}–{high:g}。")
        if key in ("nx", "nz") and int(value) != value:
            raise ValueError(f"{label}须为整数。")
    if s.vent not in VENTS:
        raise ValueError("排气边界无效。")


@dataclass(frozen=True)
class SectionFrame:
    time: float
    thickness: float          # mm, current physical depth
    temperature: tuple[float, ...]    # °C, row-major from the heated face
    residual: tuple[float, ...]       # % of original mass in each material cell
    concentration: tuple[float, ...]  # kg volatile / m³ current connected pore volume
    solid: float              # all inventories below as % of sample's initial mass
    pore: float
    front_out: float
    back_out: float


@dataclass(frozen=True)
class SectionResult:
    settings: SectionParameters
    frames: tuple[SectionFrame, ...]
    mass_error: float         # max absolute relative error over all steps


def section_sweep(values, capacity, mobility, lines, dt, spacing, boundary):
    """Implicit finite-volume sweep. Boundary terms are (sink, source) per bulk volume."""
    factor = dt / spacing ** 2
    for line in lines:
        diagonal = [capacity[i] for i in line]
        rhs = [capacity[i] * values[i] for i in line]
        off = []
        for j in range(len(line) - 1):
            a, b = mobility[line[j]], mobility[line[j + 1]]
            g = factor * 2 * a * b / (a + b) if a + b else 0.0
            diagonal[j] += g
            diagonal[j + 1] += g
            off.append(-g)
        if boundary:
            for j in (0, -1):
                sink, source = boundary[line[j]]
                diagonal[j] += dt * sink
                rhs[j] += dt * source
        solution = tridiagonal_solve(diagonal, rhs, off, off)
        for i, value in zip(line, solution):
            values[i] = value


def section_simulate(p, kinetics, s=SectionParameters(), cancelled=lambda: False, report=lambda n: None):
    """Inert x-z material grid with conservative volatile diffusion in moving pore volumes."""
    validate_slab(p)
    validate_section(s)
    nx, nz = int(s.nx), int(s.nz)
    count = nx * nz
    steps = math.ceil(p.duration / p.step)
    dt = p.duration / steps
    multi = isinstance(kinetics, SlabMechanism)
    network = ReactionGrid(kinetics, count) if isinstance(kinetics, ReactionNetwork) else None
    single = kinetics is not None and not multi and network is None
    if multi:
        validate_mechanism(kinetics)
        r1, r2 = kinetics.first, kinetics.second
        parallel = kinetics.mode == "parallel"
        props = (SolidProperties(p.density, p.heat_capacity, p.conductivity),
                 kinetics.intermediate, kinetics.char)
        specific_volume = tuple(p.density / m.density for m in props)
    elif single:
        if len(kinetics) != 3 or any(not FIT_BOUNDS[key][0] <= v <= FIT_BOUNDS[key][1]
                                     for key, v in zip(FIT_KEYS, kinetics)):
            raise ValueError("主窗口热解参数超出输入范围。")
        r1 = ReactionStep(*kinetics, PYROLYSIS_HEAT / 1000)
    if multi or single:
        a1, e1, y1, h1 = 10 ** r1.log_a, 1000 * r1.energy / R, r1.solid_yield / 100, 1000 * r1.heat
    if multi:
        a2, e2, y2, h2 = 10 ** r2.log_a, 1000 * r2.energy / R, r2.solid_yield / 100, 1000 * r2.heat

    rows = tuple(range(j * nx, (j + 1) * nx) for j in range(nz))
    columns = tuple(range(i, count, nx) for i in range(nx))
    dx, dz0 = s.width / (1000 * nx), p.thickness / (1000 * nz)
    exposed_left = nx * (1 - s.heated / 100) / 2
    exposed_right = nx - exposed_left
    flux = [1000 * p.flux * max(0.0, min(i + 1, exposed_right) - max(i, exposed_left))
            for i in range(nx)]
    ambient = p.ambient + KELVIN
    sigma = STEFAN_BOLTZMANN * p.emissivity
    front, back = [ambient] * nx, [ambient] * nx
    temperature = [ambient] * count
    raw, mid, char, gas = [1.0] * count, [0.0] * count, [0.0] * count, [0.0] * count
    front_out = back_out = mass_error = 0.0
    open_front, open_back = s.vent in ("front", "both"), s.vent in ("back", "both")
    every = max(1, math.ceil(steps / 160))

    def stretch():
        return 1 + (s.end_thickness / 100 - 1) * (1 - sum(raw) / count)

    def pores():
        return [(s.porosity_raw * u + s.porosity_end * (1 - u)) / 100 for u in raw]

    def thermal(i, scale):
        # Sensible capacity per initial volume and conductivity of the current cell.
        if not multi:
            return p.density * p.heat_capacity, p.conductivity
        u, v, w = raw[i], mid[i], char[i]
        vol0, vol1, vol2 = u, v * specific_volume[1], w * specific_volume[2]
        solid_volume = vol0 + vol1 + vol2
        void = max(0.0, scale - solid_volume)
        ref_capacity = p.density * (u * p.heat_capacity + v * props[1].heat_capacity + w * props[2].heat_capacity)
        mixed_k = vol0 * p.conductivity + vol1 * props[1].conductivity + vol2 * props[2].conductivity
        return ref_capacity + 1200 * void, (mixed_k + .026 * void) / (solid_volume + void)

    def react_half(scale):
        if network is not None:
            for i, t in enumerate(temperature):
                released, absorbed, consumed = network.advance(i, t, dt / 2)
                gas[i] += released
                temperature[i] -= absorbed / p.heat_capacity
                raw[i] = max(0.0, raw[i] - consumed)
            check_temperature()
            return
        if not (single or multi):
            return
        for i, t in enumerate(temperature):
            k1 = a1 * math.exp(-e1 / t)
            if multi:
                k2 = a2 * math.exp(-e2 / t)
                first, second, next_mid = reaction_amounts(parallel, raw[i], mid[i], k1, k2, dt / 2, y1)
                old_capacity, _ = thermal(i, scale)
                raw[i] = max(0.0, raw[i] - first - (second if parallel else 0.0))
                mid[i] = next_mid
                char[i] += second * y2 + (first * y1 if parallel else 0.0)
                gas[i] += first * (1 - y1) + second * (1 - y2)
                new_capacity, _ = thermal(i, scale)
                temperature[i] -= p.density * (first * h1 + second * h2) / ((old_capacity + new_capacity) / 2)
            else:
                first = raw[i] * -math.expm1(-k1 * dt / 2)
                raw[i] -= first
                char[i] += first * y1
                gas[i] += first * (1 - y1)
                temperature[i] -= first * h1 / p.heat_capacity
        check_temperature()

    def check_temperature():
        if any(not 1 < value < 3000 for value in temperature):
            raise ValueError("温度超出计算范围，请减小步长并检查输入的热通量和反应热。")

    def heat_step(scale):
        dz = dz0 * scale
        capacities, conductivities = zip(*(thermal(i, scale) for i in range(count)))
        capacity = [value / scale for value in capacities]
        section_sweep(temperature, capacity, conductivities, rows, dt, dx, {})
        boundary, faces = {}, []
        for j, column in enumerate(columns):
            for face, i, incident, h in ((front, column[0], flux[j], p.front_h),
                                         (back, column[-1], 0.0, p.back_h)):
                ts = face[j]
                h += sigma * (ts + ambient) * (ts * ts + ambient * ambient)
                resistance = dz / (2 * conductivities[i])
                denom = 1 + h * resistance
                sink, source = h / denom, (incident + h * ambient) / denom
                boundary[i] = (sink / dz, source / dz)
                faces.append((face, j, i, resistance, sink, source))
        section_sweep(temperature, capacity, conductivities, columns, dt, dz, boundary)
        check_temperature()
        for face, j, i, resistance, sink, source in faces:
            face[j] = temperature[i] + resistance * (source - sink * temperature[i])

    def transport(scale):
        if s.diffusivity == 0 or not any(gas):
            return 0.0, 0.0
        dz = dz0 * scale
        phi = pores()
        ref_density = p.density / scale
        concentration = [m * ref_density / f for m, f in zip(gas, phi)]
        # Superficial diffusive flux = -D_free * phi^(3/2) * grad(pore concentration).
        mobility = [s.diffusivity * f ** 1.5 for f in phi]
        section_sweep(concentration, phi, mobility, rows, dt, dx, {})
        boundary = {}
        for col in columns:
            for i, opened in ((col[0], open_front), (col[-1], open_back)):
                boundary[i] = (2 * mobility[i] / dz ** 2 if opened else 0.0, 0.0)
        section_sweep(concentration, phi, mobility, columns, dt, dz, boundary)
        for i, c in enumerate(concentration):
            gas[i] = phi[i] * c / ref_density
        # Integrate face fluxes independently of the solid and pore inventories.
        weight = dt / (count * ref_density)
        return tuple(weight * sum(boundary[col[face]][0] * concentration[col[face]] for col in columns)
                     for face in (0, -1))

    def frame(t, scale):
        if network is None:
            residual = tuple(100 * (u + v + w) for u, v, w in zip(raw, mid, char))
        else:
            residual = tuple(100 * sum(mass) for mass in network.mass)
        c = tuple(g * p.density / (f * scale) for g, f in zip(gas, pores()))
        return SectionFrame(t, p.thickness * scale, tuple(value - KELVIN for value in temperature),
                            residual, c, sum(residual) / count, 100 * sum(gas) / count,
                            100 * front_out, 100 * back_out)

    scale = 1.0
    frames = [frame(0.0, scale)]
    last_percent = -1
    for step in range(1, steps + 1):
        if cancelled():
            raise FitCancelled()
        react_half(scale)
        midpoint_scale = stretch()
        heat_step(midpoint_scale)
        react_half(midpoint_scale)
        scale = stretch()
        out_f, out_b = transport(scale)
        front_out += out_f
        back_out += out_b
        condensed = sum(raw) + sum(mid) + sum(char) if network is None else sum(map(sum, network.mass))
        total = (condensed + sum(gas)) / count + front_out + back_out
        mass_error = max(mass_error, abs(total - 1))
        if step % every == 0 or step == steps:
            frames.append(frame(step * dt, scale))
        percent = 100 * step // steps
        if percent != last_percent:
            report(percent)
            last_percent = percent
    return SectionResult(s, tuple(frames), mass_error)


def export_section(path, result, index=None):
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        if index is None:
            writer.writerow(("time_s", "thickness_mm", "solid_pct_initial", "pore_gas_pct_initial",
                             "escaped_front_pct_initial", "escaped_back_pct_initial"))
            for f in result.frames:
                writer.writerow((f.time, f.thickness, f.solid, f.pore, f.front_out, f.back_out))
            return
        f, s = result.frames[index], result.settings
        nx, nz = int(s.nx), int(s.nz)
        writer.writerow(("time_s", "x_center_mm", "z_center_mm", "thickness_mm", "temperature_C",
                         "local_solid_pct_initial", "gas_kg_per_m3_pore"))
        for i, (t, solid, concentration) in enumerate(zip(f.temperature, f.residual, f.concentration)):
            z, x = divmod(i, nx)
            writer.writerow((f.time, (x + .5) * s.width / nx, (z + .5) * f.thickness / nz,
                             f.thickness, t, solid, concentration))


SLAB_VIEWS = {
    "temperature": "正面与背面温度",
    "mass": "整体剩余质量",
    "loss": "质量损失率",
    "profile": "温度剖面",
    "residual_profile": "局部剩余质量剖面",
    "probe_temperature": "测点温度",
    "probe_residual": "测点局部剩余质量",
    "composition": "整体组分",
}
PROFILES = {"profile", "residual_profile"}
CASE_FORMAT = "pyro-ignition-lab/slab"
CASE_VERSION = 1


def slab_probe(result, depth):
    """Return (temperature, local remaining mass %) for each saved time."""
    grid = result.depth_mm
    if not math.isfinite(depth) or not grid[0] <= depth <= grid[-1]:
        raise ValueError(f"测点深度须在 0–{grid[-1]:g} mm 之间。")
    right = max(1, min(len(grid) - 1, bisect.bisect_left(grid, depth)))
    left = right - 1
    fraction = (depth - grid[left]) / (grid[right] - grid[left])
    # Temperature includes boundary values; local mass is cell-centered.
    last = len(grid) - 3
    before = max(0, min(last, left - 1))
    after = max(0, min(last, right - 1))
    return tuple(((1 - fraction) * f.temperature[left] + fraction * f.temperature[right],
                  (1 - fraction) * f.residual[before] + fraction * f.residual[after])
                 for f in result.frames)


def export_probe(path, result, depth):
    readings = slab_probe(result, depth)
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("time_s", "depth_mm", "temperature_C", "local_remaining_mass_pct"))
        for frame, (temp, mass) in zip(result.frames, readings):
            writer.writerow(f"{v:.8g}" for v in (frame.time, depth, temp, mass))


def export_profile(path, result, index):
    frame = result.frames[index]
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("time_s", "depth_mm", "temperature_C", "local_remaining_mass_pct"))
        mass = (frame.residual[0], *frame.residual, frame.residual[-1])
        for depth, temp, remaining in zip(result.depth_mm, frame.temperature, mass, strict=True):
            writer.writerow(f"{v:.8g}" for v in (frame.time, depth, temp, remaining))


@dataclass(frozen=True)
class SlabCase:
    parameters: SlabParameters
    kinetics: tuple[float, ...] | SlabMechanism | ReactionNetwork
    result: SlabResult | None
    probe_depth: float = 0.0
    view: str = "temperature"
    time_index: int = 0
    section: SectionParameters | None = None


def save_slab_case(path, case):
    target = Path(path)
    temp_path = None
    if isinstance(case.kinetics, ReactionNetwork):
        case.kinetics.validate()
        reactions = {"type": "network", "data": asdict(case.kinetics)}
    elif isinstance(case.kinetics, SlabMechanism):
        reactions = {"type": "multistep", "data": asdict(case.kinetics)}
    else:
        reactions = {"type": "single", "data": case.kinetics}
    payload = {
        "format": CASE_FORMAT, "version": CASE_VERSION, "parameters": asdict(case.parameters),
        "kinetics": reactions,
        "result": asdict(case.result) if case.result else None,
        "view": {"probe_depth_mm": case.probe_depth, "plot": case.view, "time_index": case.time_index},
    }
    if case.section is not None:
        validate_section(case.section)
        payload["section"] = asdict(case.section)
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                         prefix=".pyro-", suffix=".tmp", delete=False) as file:
            temp_path = Path(file.name)
            json.dump(payload, file, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        os.replace(temp_path, target)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def checked_case_number(value):
    try:
        if type(value) in (float, int) and math.isfinite(value):
            return float(value)
    except OverflowError:
        pass
    raise ValueError("算例包含无效数值。")


def checked_case_fields(data, fields):
    fields = tuple(fields)
    if not isinstance(data, dict) or set(data) != set(fields):
        raise ValueError("算例字段缺失或格式不匹配。")
    return {key: checked_case_number(data[key]) for key in fields}


def case_vector(data, size):
    if not isinstance(data, list) or len(data) != size:
        raise ValueError("算例数据长度与网格不一致。")
    return tuple(checked_case_number(value) for value in data)


def decode_case_result(data, parameters, multistep, component_names=()):
    if not isinstance(data, dict):
        raise ValueError("算例结果格式有误。")
    count = int(parameters.cells)
    depths = case_vector(data.get("depth_mm"), count + 2)
    expected = (0.0, *(parameters.thickness * (i + .5) / count for i in range(count)), parameters.thickness)
    if any(not math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-8) for a, b in zip(depths, expected)):
        raise ValueError("算例结果的深度网格与设置不一致。")
    stored_names = data.get("component_names", [])
    if not isinstance(stored_names, list) or tuple(stored_names) != component_names:
        raise ValueError("算例结果与反应表的组分顺序不一致。")
    ncomp = len(component_names) + 1 if component_names else (4 if multistep else 0)
    rows = data.get("frames")
    if not isinstance(rows, list) or not 2 <= len(rows) <= 1602:
        raise ValueError("算例结果的采样数无效。")
    frames = []
    previous = -1.0
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("算例帧格式有误。")
        numbers = {key: checked_case_number(row.get(key)) for key in
                   ("time", "front", "back", "remaining_mass", "loss_rate")}
        if numbers["time"] <= previous or not 0 <= numbers["time"] <= parameters.duration + 1e-7:
            raise ValueError("算例结果的时间序列无效。")
        previous = numbers["time"]
        temp = case_vector(row.get("temperature"), count + 2)
        residual = case_vector(row.get("residual"), count)
        composition = case_vector(row.get("composition"), ncomp)
        if not math.isclose(temp[0], numbers["front"], abs_tol=1e-7) or not math.isclose(temp[-1], numbers["back"], abs_tol=1e-7):
            raise ValueError("算例的表面温度与温度剖面不一致。")
        if (any(not -1e-7 <= value <= 100 + 1e-7 for value in (*residual, *composition))
                or not math.isclose(sum(residual) / count, numbers["remaining_mass"], abs_tol=1e-7)
                or numbers["loss_rate"] < 0):
            raise ValueError("算例的局部剩余质量或质量损失率无效。")
        if composition and (abs(sum(composition) - 100) > 1e-6
                            or abs(sum(composition[:-1]) - numbers["remaining_mass"]) > 1e-6):
            raise ValueError("算例的组分与剩余质量不一致。")
        frames.append(SlabFrame(**numbers, temperature=temp, residual=residual, composition=composition))
    if frames[0].time != 0 or not math.isclose(previous, parameters.duration, abs_tol=1e-7):
        raise ValueError("算例结果的起止时刻与模拟时长不一致。")
    summary = [checked_case_number(data.get(key)) for key in ("peak_loss", "mass_error", "energy_error")]
    if any(value < 0 for value in summary) or summary[0] + 1e-8 < max(frame.loss_rate for frame in frames):
        raise ValueError("算例结果的汇总数据无效。")
    return SlabResult(expected, tuple(frames), *summary, component_names)


def load_slab_case(path):
    path = Path(path)
    if path.stat().st_size > 32 * 1024 ** 2:
        raise ValueError("算例文件超过 32 MB。")
    try:
        with path.open(encoding="utf-8-sig") as file:
            data = json.load(file)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise ValueError("无法读取算例 JSON。") from error
    if (not isinstance(data, dict) or data.get("format") != CASE_FORMAT
            or type(data.get("version")) is not int or data.get("version") != CASE_VERSION):
        raise ValueError("算例格式或版本不受支持。")
    parameters = SlabParameters(**checked_case_fields(data.get("parameters"), (v[0] for v in SLAB_FIELDS)))
    validate_slab(parameters)
    reactions = data.get("kinetics")
    if not isinstance(reactions, dict) or reactions.get("type") not in ("single", "multistep", "network"):
        raise ValueError("算例的反应模型格式有误。")
    multi = reactions["type"] == "multistep"
    source = reactions.get("data")
    component_names = ()
    if reactions["type"] == "network":
        if not isinstance(source, dict):
            raise ValueError("算例缺少组分与反应表。")
        def records(key, cls, strings):
            rows = source.get(key)
            if not isinstance(rows, list):
                raise ValueError("组分或反应表格式错误。")
            items = []
            for row in rows:
                if not isinstance(row, dict) or set(row) != set(cls.__dataclass_fields__):
                    raise ValueError("组分或反应行的字段不完整。")
                if any(not isinstance(row[k], str) for k in strings):
                    raise ValueError("组分名称必须是文本。")
                items.append(cls(**{k: v if k in strings else checked_case_number(v) for k, v in row.items()}))
            return tuple(items)
        kinetics = ReactionNetwork(records("components", Component, ("name",)),
                                   records("reactions", NetworkReaction, ("source", "target")))
        kinetics.validate()
        component_names = kinetics.names
    elif multi:
        if not isinstance(source, dict):
            raise ValueError("算例缺少多步反应参数。")
        reaction_fields = tuple(v[0] for v in REACTION_FIELDS)
        material_fields = tuple(v[0] for v in PROPERTY_FIELDS)
        kinetics = SlabMechanism(source.get("mode"),
            ReactionStep(**checked_case_fields(source.get("first"), reaction_fields)),
            ReactionStep(**checked_case_fields(source.get("second"), reaction_fields)),
            SolidProperties(**checked_case_fields(source.get("intermediate"), material_fields)),
            SolidProperties(**checked_case_fields(source.get("char"), material_fields)))
        validate_mechanism(kinetics)
    else:
        kinetics = case_vector(source, len(FIT_KEYS))
        for key, value in zip(FIT_KEYS, kinetics):
            low, high = FIT_BOUNDS[key]
            if not low <= value <= high:
                raise ValueError("算例的单步热解参数超出范围。")
    view = data.get("view")
    if not isinstance(view, dict) or not isinstance(view.get("plot"), str) or view["plot"] not in SLAB_VIEWS:
        raise ValueError("算例的图表设置无效。")
    depth = checked_case_number(view.get("probe_depth_mm"))
    if not 0 <= depth <= parameters.thickness:
        raise ValueError("算例测点不在平板范围内。")
    result = decode_case_result(data["result"], parameters, multi, component_names) if data.get("result") is not None else None
    index = view.get("time_index")
    if type(index) is not int or not 0 <= index < (len(result.frames) if result else 1):
        raise ValueError("算例的剖面时刻无效。")
    section = None
    if "section" in data:
        source = data["section"]
        if not isinstance(source, dict) or not isinstance(source.get("vent"), str):
            raise ValueError("二维截面设置无效。")
        values = checked_case_fields({key: val for key, val in source.items() if key != "vent"},
                                     (field[0] for field in SECTION_FIELDS))
        section = SectionParameters(**values, vent=source["vent"])
        validate_section(section)
        section = replace(section, nx=int(section.nx), nz=int(section.nz))
    return SlabCase(parameters, kinetics, result, depth, view["plot"], index, section)


SECTION_QUANTITIES = {
    "温度 (°C)": "temperature",
    "局部剩余固体 (% 初始)": "residual",
    "孔内挥发物 (kg/m³ 孔隙)": "concentration",
}

SECTION_HELP = """二维截面在惰性气氛中计算宽向 x 与厚向 z 的导热。受热面为 z = 0。正面中央条带输入平板设置的吸收热通量，两面均与环境对流和辐射换热。两侧绝热且封闭。单元中心温度和固体质量随时间输出。

每次运行读取一维平板中当前的样品、传热、反应和时间设置。宽向与厚向网格数在本窗口单独指定。单步和自定义反应表采用固定的参考热容和导热系数，预设两步机制采用组分热物性。反应产生的单一等效挥发物进入当地连通孔隙，再沿两个方向扩散。有效扩散系数为自由气体扩散系数乘以孔隙率的 1.5 次方。开放面取外部挥发物浓度为零，封闭面取零通量。结果计入孔内库存以及两面的净排出量。

完全转化厚度以初始厚度为 100%。例如 80% 表示缩至 0.8 倍，120% 表示膨胀至 1.2 倍。厚度随原材料平均转化率线性变化，宽度固定，各列同步伸缩。孔隙率则随各单元原材料转化率在两个输入端点间插值。几何伸缩和孔隙率是独立指定的经验关系。模型假设孔隙中的气体以扩散传质，忽略压力驱动流动、气体携热及变形功。

色标在一次计算的所有时刻固定。纵轴按该次计算的最大厚度绘制，横纵轴分别标度。悬停可读取单元中心值。质量汇总以初始试样质量为基准。时间历程 CSV 给出厚度和四项质量，当前截面 CSV 给出网格中心的物理坐标和三个场。

在一维平板窗口保存算例可同时保存本窗口的参数。打开算例后运行二维计算即可重建各时刻的截面。示例参数用于检查模型行为，实际材料需用实验标定孔隙率、扩散系数和厚度演化。"""


class SectionWindow:
    def __init__(self, slab):
        self.slab, self.lab = slab, slab.lab
        tk, ttk = slab.tk, slab.ttk
        self.window = tk.Toplevel(slab.window)
        self.window.title("二维截面 · 热解与多孔传质")
        self.window.geometry("1040x680")
        self.window.minsize(815, 570)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.events = queue.Queue()
        self.stop = threading.Event()
        self.busy = False
        self.result = None
        self.inputs = {}
        self.status = tk.StringVar(value="运行时读取一维平板当前的材料、反应、热边界和时间设置。")
        self.summary = tk.StringVar(value="运行后可查看厚度、存留质量和两面的排出量。")
        self.readout = tk.StringVar(value="受热面 z = 0 mm")
        self.position = tk.DoubleVar(value=0)
        outer = ttk.Frame(self.window, padding=12)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(1, weight=1)
        head = ttk.Frame(outer)
        head.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        self.lab.text(head, "二维截面", bg=BG, bold=True, size=18).pack(side="left")
        ttk.Button(head, text="模型说明", command=self.help).pack(side="right")
        panel = ttk.Frame(outer, style="Panel.TFrame", padding=10)
        panel.grid(row=1, column=0, sticky="ns", padx=(0, 9))
        defaults = slab.section_settings or SectionParameters()
        row = 0
        for key, label, _, _ in SECTION_FIELDS:
            if key in ("width", "end_thickness", "porosity_raw"):
                title = {"width": "试样与加热区", "end_thickness": "厚度演化", "porosity_raw": "孔内传质"}[key]
                self.lab.text(panel, title, bold=True, fg=ACCENT).grid(row=row, columnspan=2, sticky="w", pady=(6, 3))
                row += 1
            self.lab.text(panel, label, size=9).grid(row=row, column=0, sticky="w", padx=(0, 8))
            var = tk.StringVar(value=f"{getattr(defaults, key):g}")
            self.inputs[key] = var
            ttk.Entry(panel, textvariable=var, width=8, justify="right").grid(row=row, column=1, sticky="e", pady=1)
            row += 1
        self.lab.text(panel, "排气边界", size=9).grid(row=row, column=0, sticky="w")
        self.vent = ttk.Combobox(panel, state="readonly", values=tuple(VENTS.values()), width=9)
        self.vent.set(VENTS[defaults.vent])
        self.vent.grid(row=row, column=1, sticky="e")
        self.run_button = ttk.Button(panel, text="运行二维模拟", command=self.run, style="Primary.TButton")
        self.run_button.grid(row=row + 1, columnspan=2, sticky="ew", pady=(14, 4))
        self.cancel_button = ttk.Button(panel, text="停止计算", command=self.stop.set, state="disabled")
        self.cancel_button.grid(row=row + 2, columnspan=2, sticky="ew")

        view = ttk.Frame(outer, style="Panel.TFrame", padding=10)
        view.grid(row=1, column=1, sticky="nsew")
        self.lab.text(view, value=self.summary, fg=ACCENT, bold=True, size=10,
                      wraplength=490, justify="left").pack(anchor="w", pady=(0, 5))
        self.quantity = ttk.Combobox(view, values=tuple(SECTION_QUANTITIES), state="readonly", width=28)
        self.quantity.current(0)
        self.quantity.pack(anchor="w")
        self.quantity.bind("<<ComboboxSelected>>", self.draw)
        self.canvas = tk.Canvas(view, bg=PANEL, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self.draw)
        self.canvas.bind("<Motion>", self.hover)
        self.canvas.bind("<Leave>", lambda _: self.readout.set("移动鼠标可读取单元中心值"))
        self.lab.text(view, value=self.readout, fg=MUTED, size=9).pack(anchor="w")
        self.scale = ttk.Scale(view, variable=self.position, from_=0, to=1, command=self.draw, state="disabled")
        self.scale.pack(fill="x", pady=(2, 6))
        actions = ttk.Frame(view, style="Panel.TFrame")
        actions.pack(fill="x")
        self.exports = []
        for label, spatial in (("导出当前截面 CSV", True), ("导出时间历程 CSV", False)):
            button = ttk.Button(actions, text=label, command=lambda field=spatial: self.export(field), state="disabled")
            button.pack(side="left", padx=(0, 5))
            self.exports.append(button)
        self.lab.text(outer, value=self.status, bg=BG, fg=MUTED, size=9, wraplength=765,
                      justify="left").grid(row=2, column=0, columnspan=2, sticky="w", pady=(7, 0))
        self.timer = self.window.after(100, self.poll)

    def settings(self):
        values = {}
        for key, label, _, _ in SECTION_FIELDS:
            try:
                values[key] = float(self.inputs[key].get())
            except ValueError as error:
                raise ValueError(f"{label}请输入数字。") from error
        vent = next(key for key, label in VENTS.items() if label == self.vent.get())
        settings = SectionParameters(**values, vent=vent)
        validate_section(settings)
        return replace(settings, nx=int(settings.nx), nz=int(settings.nz))

    def close(self):
        try:
            self.slab.section_settings = self.settings()
        except ValueError:
            pass
        self.stop.set()
        self.window.after_cancel(self.timer)
        self.window.destroy()

    def help(self):
        popup = self.slab.tk.Toplevel(self.window)
        popup.title("二维截面 · 模型说明")
        popup.geometry("660x480")
        view = self.slab.tk.Text(popup, wrap="word", padx=14, pady=12)
        view.pack(fill="both", expand=True)
        view.insert("1.0", SECTION_HELP)
        view.configure(state="disabled")

    def run(self):
        if self.busy:
            return
        try:
            p, k, s = self.slab.parameter_snapshot(), self.slab.current_kinetics(), self.settings()
        except ValueError as error:
            self.status.set(f"输入有误 · {error}")
            return
        self.slab.section_settings = s
        self.busy = True
        self.stop.clear()
        self.run_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.status.set("正在计算二维截面…")

        def work():
            try:
                result = section_simulate(p, k, s, self.stop.is_set, lambda n: self.events.put(("progress", n)))
            except FitCancelled:
                self.events.put(("error", "本次计算已停止。"))
            except (ValueError, ArithmeticError) as error:
                self.events.put(("error", f"计算失败 · {error}"))
            else:
                self.events.put(("done", result))
        threading.Thread(target=work, daemon=True).start()

    def poll(self):
        while True:
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "progress":
                self.status.set(f"正在计算二维截面 · {value}%")
                continue
            self.busy = False
            self.run_button.configure(state="normal")
            self.cancel_button.configure(state="disabled")
            if kind == "error":
                self.status.set(value)
                continue
            self.result = value
            frames = value.frames
            self.max_depth = max(f.thickness for f in frames)
            self.limits = {
                "temperature": (min(min(f.temperature) for f in frames), max(max(f.temperature) for f in frames)),
                "concentration": (0.0, max(max(f.concentration) for f in frames)), "residual": (0.0, 100.0),
            }
            self.scale.configure(state="normal", to=len(frames) - 1)
            self.position.set(len(frames) - 1)
            for button in self.exports:
                button.configure(state="normal")
            self.readout.set("移动鼠标可读取单元中心值")
            self.status.set(f"图与导出对应最近完成的计算 · 全程最大相对质量误差 {value.mass_error:.2e}")
            self.draw()
        self.timer = self.window.after(100, self.poll)

    def index(self):
        return max(0, min(len(self.result.frames) - 1, round(self.position.get())))

    @staticmethod
    def color(fraction):
        stops = ((236, 247, 253), (102, 193, 205), (42, 105, 149), (53, 48, 105))
        pos = max(0.0, min(1.0, fraction)) * (len(stops) - 1)
        j = min(len(stops) - 2, int(pos))
        rgb = tuple(round(a + (b - a) * (pos - j)) for a, b in zip(stops[j], stops[j + 1]))
        return "#%02x%02x%02x" % rgb

    def draw(self, *_args):
        c = self.canvas
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if w < 200 or h < 110:
            return
        if self.result is None:
            c.create_text(w / 2, h / 2, text="运行二维模拟后查看截面", fill=MUTED)
            return
        f, s = self.result.frames[self.index()], self.result.settings
        key = SECTION_QUANTITIES[self.quantity.get()]
        lo, hi = self.limits[key]
        nx, nz = int(s.nx), int(s.nz)
        left, right, top = 49, w - 19, 36
        bottom = top + (h - 100) * f.thickness / self.max_depth
        dx, dz = (right - left) / nx, (bottom - top) / nz
        self.plot_box = (left, right, top, bottom, dx, dz)
        for i, val in enumerate(getattr(f, key)):
            z, x = divmod(i, nx)
            fraction = (val - lo) / (hi - lo) if hi > lo else 0.0
            c.create_rectangle(left + dx * x, top + dz * z, left + dx * (x + 1), top + dz * (z + 1),
                               fill=self.color(fraction), outline="")
        c.create_rectangle(left, top, right, bottom, outline=MUTED)
        strip = (right - left) * s.heated / 100
        mid = (left + right) / 2
        if strip > 0:
            c.create_line(mid - strip / 2, top - 2, mid + strip / 2, top - 2, width=3, fill=AIR)
        c.create_text(left, 14, anchor="w", text="受热面 · 红线为加热区", fill=MUTED)
        c.create_text(left - 5, top, anchor="e", text="0", fill=MUTED)
        c.create_text(left - 5, bottom, anchor="e", text=f"{f.thickness:.2g}", fill=MUTED)
        c.create_text(left, bottom + 11, text="0", fill=MUTED)
        c.create_text(right, bottom + 11, text=f"{s.width:g}", fill=MUTED)
        c.create_text(mid, bottom + 13, text="宽 x (mm)    /    深 z (mm)", fill=MUTED)
        for j in range(48):
            c.create_rectangle(left + (right - left) * j / 48, h - 29, left + (right - left) * (j + 1) / 48,
                               h - 21, outline="", fill=self.color(j / 47 if hi > lo else 0))
        c.create_text(left, h - 12, text=f"{lo:.4g}", anchor="w", fill=MUTED)
        c.create_text(right, h - 12, text=f"{hi:.4g}", anchor="e", fill=MUTED)
        self.summary.set(f"{f.time:.2f} s   ·   厚度 {f.thickness:.3f} mm   ·   剩余固体 {f.solid:.2f}%\n"
                         f"孔内 {f.pore:.3f}%   ·   正面排出 {f.front_out:.3f}%   ·   背面排出 {f.back_out:.3f}%")

    def hover(self, event):
        if self.result is None or not hasattr(self, "plot_box"):
            return
        left, right, top, bottom, dx, dz = self.plot_box
        if not (left <= event.x < right and top <= event.y < bottom):
            self.readout.set("移动鼠标可读取单元中心值")
            return
        f, s = self.result.frames[self.index()], self.result.settings
        x, z = int((event.x - left) / dx), int((event.y - top) / dz)
        nx, nz = int(s.nx), int(s.nz)
        i = z * nx + x
        self.readout.set(f"x {(x + .5) * s.width / nx:.2f}, z {(z + .5) * f.thickness / nz:.2f} mm"
                         f"  ·  {f.temperature[i]:.1f} °C  ·  固体 {f.residual[i]:.2f}%  ·  孔内 {f.concentration[i]:.3g} kg/m³")

    def export(self, spatial):
        if self.result is None:
            return
        path = self.lab.dialog.asksaveasfilename(parent=self.window, defaultextension=".csv",
            initialfile="section_field.csv" if spatial else "section_history.csv", filetypes=[("CSV", "*.csv")])
        if path:
            try:
                export_section(path, self.result, self.index() if spatial else None)
            except OSError as error:
                self.status.set(f"导出失败 · {error}")
            else:
                self.status.set(f"已导出 {Path(path).name} · 来自最近完成的二维计算")


class SlabWindow:
    def __init__(self, lab):
        self.lab, self.tk, self.ttk = lab, lab.tk, lab.ttk
        tk, ttk = self.tk, self.ttk
        self.window = tk.Toplevel(lab.root)
        self.window.title("一维平板热解")
        self.window.geometry("1180x810")
        self.window.minsize(860, 690)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.stop = threading.Event()
        self.queue = queue.Queue()
        self.result = None
        self.mechanism = None
        self.busy = False
        self.dirty = True
        self.inputs = {}
        self.mode = tk.StringVar(value="temperature")
        self.status = tk.StringVar(value="示例工况 · 使用主窗口的热解参数")
        self.summary = tk.StringVar(value="运行后查看温度与失重")
        self.frame_label = tk.StringVar(value="剖面时刻")
        self.position = tk.DoubleVar(value=0)
        self.probe_depth = 0.0
        self.probe_input = tk.StringVar(value="0")
        self.section_settings = None
        self.section_window = None
        self.case_name = ""
        self.case_label = tk.StringVar(value="未命名算例")
        outer = ttk.Frame(self.window, padding=12)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(1, weight=1)
        top = ttk.Frame(outer)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        lab.text(top, "一维平板热解", bg=BG, bold=True, size=18).pack(side="left")
        ttk.Button(top, text="边界与模型说明", command=self.help).pack(side="right")
        ttk.Button(top, text="组分与反应表", command=self.open_table).pack(side="right", padx=(0, 7))
        ttk.Button(top, text="多步反应与材料", command=self.edit_mechanism).pack(side="right", padx=(0, 7))
        ttk.Button(top, text="多指标反演", command=self.open_inverse).pack(side="right", padx=(0, 7))
        ttk.Button(top, text="二维截面", command=self.open_section).pack(side="right", padx=(0, 7))

        options = ttk.Frame(outer, style="Panel.TFrame", padding=(10, 9))
        options.grid(row=1, column=0, sticky="nsw", padx=(0, 9))
        defaults = SlabParameters()
        groups = {0: "样品与热物性", 4: "边界条件", 9: "时间与网格"}
        row = 0
        for index, (name, label, unit, _, _) in enumerate(SLAB_FIELDS):
            if index in groups:
                lab.text(options, groups[index], fg=ACCENT, bold=True).grid(
                    row=row, column=0, columnspan=3, sticky="w", pady=(5, 2))
                row += 1
            variable = tk.StringVar(value=f"{getattr(defaults, name):g}")
            self.inputs[name] = variable
            lab.text(options, label, size=9).grid(row=row, column=0, sticky="w", padx=(0, 5))
            ttk.Entry(options, textvariable=variable, width=7, justify="right").grid(row=row, column=1, pady=1)
            lab.text(options, unit, fg=MUTED, size=9).grid(row=row, column=2, sticky="w", padx=(4, 0))
            row += 1
        self.kinetics_label = tk.StringVar()
        lab.text(options, value=self.kinetics_label, fg=MUTED, size=9, justify="left",
                 wraplength=270).grid(row=row, column=0, columnspan=3, sticky="w", pady=(7, 3))
        self.run_button = ttk.Button(options, text="运行一维模拟", command=self.run, style="Primary.TButton")
        self.run_button.grid(row=row + 1, column=0, columnspan=3, sticky="ew", pady=(2, 4))
        self.stop_button = ttk.Button(options, text="停止计算", command=self.stop.set, state="disabled")
        self.stop_button.grid(row=row + 2, column=0, columnspan=3, sticky="ew", pady=(0, 4))
        self.export_button = ttk.Button(options, text="导出曲线与剖面 CSV", command=self.save, state="disabled")
        self.export_button.grid(row=row + 3, column=0, columnspan=3, sticky="ew")
        cases = ttk.Frame(options, style="Panel.TFrame")
        cases.grid(row=row + 4, column=0, columnspan=3, sticky="ew", pady=(8, 2))
        self.open_button = ttk.Button(cases, text="打开算例", command=self.open_case)
        self.open_button.pack(side="left", padx=(0, 5))
        ttk.Button(cases, text="保存算例", command=self.write_case).pack(side="left")
        lab.text(options, value=self.case_label, fg=MUTED, size=9, wraplength=260).grid(
            row=row + 5, column=0, columnspan=3, sticky="w")
        for variable in self.inputs.values():
            variable.trace_add("write", self.mark_dirty)
        self.kinetics_traces = [
            (lab.inputs[key], lab.inputs[key].trace_add("write", self.update_kinetics))
            for key in FIT_KEYS
        ]
        self.refresh_label()

        view = ttk.Frame(outer, style="Panel.TFrame", padding=8)
        view.grid(row=1, column=1, sticky="nsew")
        lab.text(view, value=self.summary, fg=ACCENT, bold=True, wraplength=690,
                 justify="left").pack(anchor="w", pady=(0, 4))
        tabs = ttk.Frame(view, style="Panel.TFrame")
        tabs.pack(fill="x")
        self.plot_choice = ttk.Combobox(tabs, values=tuple(SLAB_VIEWS.values()), state="readonly", width=19)
        self.plot_choice.set(SLAB_VIEWS[self.mode.get()])
        self.plot_choice.pack(side="left", padx=(0, 6))
        self.plot_choice.bind("<<ComboboxSelected>>", self.select_view)
        self.component_choice = ttk.Combobox(tabs, state="readonly", width=16)
        self.component_choice.bind("<<ComboboxSelected>>", lambda _: self.draw())
        lab.text(tabs, "测点 (mm)", size=9).pack(side="left", padx=(0, 3))
        probe = ttk.Entry(tabs, textvariable=self.probe_input, width=7)
        probe.pack(side="left", padx=(0, 3))
        probe.bind("<Return>", self.set_probe)
        ttk.Button(tabs, text="应用", command=self.set_probe).pack(side="left")
        spatial = ttk.Frame(view, style="Panel.TFrame")
        spatial.pack(side="bottom", fill="x", pady=(4, 0))
        self.probe_export = ttk.Button(spatial, text="导出测点 CSV", command=self.save_spatial, state="disabled")
        self.probe_export.pack(side="left", padx=(0, 5))
        self.profile_export = ttk.Button(spatial, text="导出当前剖面 CSV",
            command=lambda: self.save_spatial(profile=True), state="disabled")
        self.profile_export.pack(side="left")
        self.canvas = tk.Canvas(view, bg=PANEL, height=330, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self.draw())
        self.canvas.bind("<Button-1>", self.pick_probe)
        self.scale = ttk.Scale(view, from_=0, to=1, variable=self.position, command=self.slide, state="disabled")
        self.scale.pack(fill="x", padx=10, pady=(1, 2))
        lab.text(view, value=self.frame_label, fg=MUTED, size=9).pack(anchor="w", padx=10)
        lab.text(outer, value=self.status, bg=BG, fg=MUTED, wraplength=1000,
                 justify="left").grid(row=2, column=0, columnspan=2, sticky="ew", pady=(7, 0))
        self.window.after(70, self.poll)

    def open_section(self):
        if self.section_window is not None and self.section_window.window.winfo_exists():
            self.section_window.window.lift()
        else:
            self.section_window = SectionWindow(self)

    def close(self):
        if self.section_window is not None and self.section_window.window.winfo_exists():
            self.section_window.close()
        inverse = getattr(self, "inverse", None)
        if inverse is not None and inverse.window.winfo_exists():
            inverse.close()
        self.stop.set()
        for variable, trace in self.kinetics_traces:
            variable.trace_remove("write", trace)
        self.window.destroy()

    def update_kinetics(self, *_args):
        if self.mechanism is None:
            self.refresh_label()
            self.mark_dirty()

    def open_table(self):
        self.open_editor(ReactionTableEditor)

    def edit_mechanism(self):
        self.open_editor(MechanismEditor)

    def open_editor(self, editor_type):
        editor = getattr(self, "editor", None)
        if editor is not None and editor.window.winfo_exists():
            editor.window.lift()
            return
        self.editor = editor_type(self)
        self.editor.window.grab_set()

    def help(self):
        popup = self.tk.Toplevel(self.window)
        popup.title("一维平板 · 边界与模型")
        popup.geometry("650x460")
        view = self.tk.Text(popup, wrap="word", padx=15, pady=14)
        view.pack(fill="both", expand=True)
        view.insert("1.0", SLAB_HELP)
        view.configure(state="disabled")

    def current_kinetics(self):
        if self.mechanism is not None:
            return self.mechanism
        return tuple(float(self.lab.inputs[key].get()) for key in FIT_KEYS)

    def refresh_label(self):
        if isinstance(self.mechanism, ReactionNetwork):
            self.kinetics_label.set(f"自定义模型 · {len(self.mechanism.components)} 种组分"
                                    f" · {len(self.mechanism.reactions)} 条反应 · 共用平板热物性")
            return
        if self.mechanism is not None:
            mode = "两步并联" if self.mechanism.mode == "parallel" else "两步串联"
            self.kinetics_label.set(f"{mode} · 随组分更新热物性")
            return
        try:
            energy, log_a, char = self.current_kinetics()
        except ValueError:
            self.kinetics_label.set("主窗口热解参数待检查")
        else:
            self.kinetics_label.set(f"主窗口热解参数 · E {energy:g} kJ/mol\nlog₁₀A {log_a:g} · 残炭产率 {char:g}%")

    def mark_dirty(self, *_args):
        self.dirty = True
        self.update_exports(ready=False)
        if not self.busy:
            self.status.set("参数已修改。运行模拟以更新曲线。")

    def parameter_snapshot(self):
        values = {}
        for name, label, _, _, _ in SLAB_FIELDS:
            try:
                values[name] = float(self.inputs[name].get())
            except ValueError as error:
                raise ValueError(f"{label}请输入数字。") from error
        parameters = SlabParameters(**values)
        validate_slab(parameters)
        return parameters

    def open_inverse(self):
        existing = getattr(self, "inverse", None)
        if existing is not None and existing.window.winfo_exists():
            existing.window.lift()
            return
        try:
            parameters, kinetics = self.parameter_snapshot(), self.current_kinetics()
            if isinstance(kinetics, (SlabMechanism, ReactionNetwork)):
                validate_mechanism(kinetics)
            else:
                for key, value in zip(FIT_KEYS, kinetics):
                    low, high = FIT_BOUNDS[key]
                    if not low <= value <= high:
                        raise ValueError("主窗口热解参数超出输入范围。")
        except ValueError as error:
            self.status.set(f"输入有误 · {error}")
            return
        self.inverse = InverseWindow(self, parameters, kinetics)

    def run(self):
        if self.busy:
            return
        try:
            parameters, kinetics = self.parameter_snapshot(), self.current_kinetics()
        except ValueError as error:
            self.status.set(f"输入有误 · {error}")
            return
        self.refresh_label()
        self.stop.clear()
        self.dirty, self.busy = False, True
        self.status.set("正在计算一维温度与质量分布…")
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.open_button.configure(state="disabled")
        self.update_exports(ready=False)

        def work():
            try:
                result = slab_simulate(parameters, kinetics, self.stop.is_set)
            except FitCancelled:
                self.queue.put(("error", "计算已停止。"))
            except (ValueError, ArithmeticError) as error:
                self.queue.put(("error", f"计算失败 · {error}"))
            else:
                self.queue.put(("done", result, (parameters, kinetics)))
        threading.Thread(target=work, daemon=True).start()

    def poll(self):
        try:
            item = self.queue.get_nowait()
        except queue.Empty:
            pass
        else:
            self.busy = False
            self.run_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            self.open_button.configure(state="normal")
            if item[0] == "done":
                _, result, self.solved_settings = item
                self.show_result(result, len(result.frames) - 1)
                mode = "随组分更新热物性" if isinstance(self.solved_settings[1], SlabMechanism) else "固定热物性"
                self.status.set(f"计算完成 · 惰性气氛 · 固定厚度 · {mode}")
                if self.dirty:
                    self.status.set("已显示上次输入的结果，运行模拟以更新修改后的参数。")
            else:
                self.dirty = True
                self.status.set(item[1])
            self.draw()
        self.window.after(70, self.poll)

    def slide(self, _value):
        if self.mode.get() in PROFILES:
            self.draw()

    def save(self):
        if self.current_result() is None:
            return
        path = self.lab.dialog.asksaveasfilename(parent=self.window, defaultextension=".csv",
            initialfile="slab_pyrolysis.csv", filetypes=[("CSV", "*.csv")])
        if path:
            try:
                export_slab(path, self.result)
            except OSError as error:
                self.status.set(f"导出失败 · {error}")
            else:
                self.status.set(f"已导出时间曲线及厚度方向数据 · {Path(path).name}")

    def frame_index(self):
        if self.result is None:
            return 0
        return max(0, min(len(self.result.frames) - 1, round(self.position.get())))

    def select_view(self, _event=None):
        key = next(key for key, label in SLAB_VIEWS.items() if label == self.plot_choice.get())
        self.mode.set(key)
        self.draw()

    def set_probe(self, _event=None):
        try:
            depth = float(self.probe_input.get())
            limit = self.result.depth_mm[-1] if self.result else self.parameter_snapshot().thickness
            if not math.isfinite(depth) or not 0 <= depth <= limit:
                raise ValueError()
        except ValueError:
            self.status.set("请输入受热面到背面之间的深度（mm）。")
            return
        self.probe_depth = depth
        self.probe_input.set(f"{depth:.8g}")
        if self.mode.get() not in PROFILES and self.mode.get() != "probe_residual":
            self.mode.set("probe_temperature")
        self.draw()

    def pick_probe(self, event):
        if self.result is None or self.mode.get() not in PROFILES:
            return
        left, right = 59, self.canvas.winfo_width() - 18
        if left <= event.x <= right:
            depth = (event.x - left) / (right - left) * self.result.depth_mm[-1]
            self.probe_input.set(str(depth))
            self.set_probe()

    def show_result(self, result, index):
        self.result = result
        last = result.frames[-1]
        self.probe_depth = min(self.probe_depth, result.depth_mm[-1])
        self.probe_input.set(f"{self.probe_depth:.8g}")
        self.scale.configure(to=len(result.frames) - 1)
        self.position.set(index)
        self.summary.set(
            f"终点温度  正面 {last.front:.1f} °C / 背面 {last.back:.1f} °C"
            f"    ·    剩余 {last.remaining_mass:.1f}%"
            f"    ·    峰值失重率 {result.peak_loss:.2f} g/(m²·s)")
        self.update_exports()

    def update_exports(self, ready=None):
        if ready is None:
            ready = self.result is not None and not self.dirty and not self.busy
        self.export_button.configure(state="normal" if ready else "disabled")
        self.probe_export.configure(state="normal" if ready else "disabled")
        self.profile_export.configure(state="normal" if ready and self.mode.get() in PROFILES else "disabled")

    def current_result(self):
        if self.result is None or self.dirty or self.busy:
            return None
        try:
            if (self.parameter_snapshot(), self.current_kinetics()) == self.solved_settings:
                return self.result
        except ValueError:
            pass
        self.mark_dirty()
        return None

    def save_spatial(self, profile=False):
        result = self.current_result()
        if result is None:
            return
        name = "slab_profile.csv" if profile else "slab_probe.csv"
        path = self.lab.dialog.asksaveasfilename(parent=self.window, defaultextension=".csv",
            initialfile=name, filetypes=[("CSV", "*.csv")])
        if path:
            try:
                if profile:
                    export_profile(path, result, self.frame_index())
                else:
                    export_probe(path, result, self.probe_depth)
            except OSError as error:
                self.status.set(f"导出失败 · {error}")
            else:
                self.status.set(f"已导出 {'当前剖面' if profile else '测点时间曲线'} · {Path(path).name}")

    def write_case(self):
        try:
            if self.section_window is not None and self.section_window.window.winfo_exists():
                self.section_settings = self.section_window.settings()
            parameters = self.parameter_snapshot()
            kinetics = self.current_kinetics()
            if isinstance(kinetics, (SlabMechanism, ReactionNetwork)):
                validate_mechanism(kinetics)
            elif any(not FIT_BOUNDS[key][0] <= value <= FIT_BOUNDS[key][1]
                     for key, value in zip(FIT_KEYS, kinetics)):
                raise ValueError("主窗口热解参数超出输入范围。")
        except ValueError as error:
            self.status.set(f"无法保存算例 · {error}")
            return
        result = self.current_result()
        case = SlabCase(parameters, kinetics, result, min(self.probe_depth, parameters.thickness),
                        self.mode.get(), self.frame_index() if result else 0, self.section_settings)
        path = self.lab.dialog.asksaveasfilename(parent=self.window, defaultextension=".json",
            initialfile=self.case_name or "slab_case.json", filetypes=[("平板算例", "*.json")])
        if path:
            try:
                save_slab_case(path, case)
            except (OSError, ValueError) as error:
                self.status.set(f"保存算例失败 · {error}")
            else:
                self.case_name = Path(path).name
                self.case_label.set(self.case_name)
                content = "设置与结果" if result else "设置"
                self.status.set(f"已保存算例 {self.case_name} · {content}")

    def open_case(self):
        if self.busy:
            return
        path = self.lab.dialog.askopenfilename(parent=self.window,
            filetypes=[("平板算例", "*.json"), ("所有文件", "*")])
        if not path:
            return
        try:
            case = load_slab_case(path)
        except (OSError, ValueError) as error:
            self.status.set(f"打开算例失败 · {error}")
            return
        if self.section_window is not None and self.section_window.window.winfo_exists():
            self.section_window.close()
        self.section_settings = case.section
        for key, variable in self.inputs.items():
            variable.set(str(getattr(case.parameters, key)))
        if isinstance(case.kinetics, (SlabMechanism, ReactionNetwork)):
            self.mechanism = case.kinetics
        else:
            self.mechanism = None
            for key, value in zip(FIT_KEYS, case.kinetics):
                self.lab.inputs[key].set(str(value))
        self.refresh_label()
        self.probe_depth = case.probe_depth
        self.probe_input.set(f"{case.probe_depth:.8g}")
        self.mode.set(case.view)
        self.case_name = Path(path).name
        self.case_label.set(self.case_name)
        self.dirty = case.result is None
        if case.result is not None:
            self.solved_settings = (case.parameters, case.kinetics)
            self.show_result(case.result, case.time_index)
        else:
            self.result = None
            self.position.set(0)
            self.summary.set("运行后查看温度与失重")
            self.update_exports()
        self.status.set(f"已打开算例 {self.case_name} · " +
                        ("设置与结果已恢复" if case.result else "设置已恢复，可运行模拟"))
        self.draw()


    def draw(self):
        canvas = self.canvas
        canvas.delete("all")
        w, h = canvas.winfo_width(), canvas.winfo_height()
        if w < 260 or h < 125:
            return
        mode = self.mode.get()
        self.plot_choice.set(SLAB_VIEWS[mode])
        if self.result and self.result.component_names and mode == "composition":
            names = (*self.result.component_names, "gases")
            self.component_choice.configure(values=names)
            if self.component_choice.get() not in names:
                self.component_choice.current(0)
            if not self.component_choice.winfo_manager():
                self.component_choice.pack(side="left", after=self.plot_choice, padx=(0, 6))
        else:
            self.component_choice.pack_forget()
        self.update_exports()
        if self.result is None:
            self.scale.configure(state="disabled")
            self.frame_label.set("受热面为 0 mm · 运行后可在剖面图上点选测点")
            canvas.create_text(w / 2, h / 2, text="设置参数后点击「运行一维模拟」", fill=MUTED)
            return
        frames = self.result.frames
        profile = mode in PROFILES
        self.scale.configure(state="normal" if profile else "disabled")
        if profile:
            selected = frames[self.frame_index()]
            self.frame_label.set(f"剖面 {selected.time:.2f} s · 测点 {self.probe_depth:g} mm · 点击图中位置可移动测点")
            xs = self.result.depth_mm
            if mode == "profile":
                yname, traces = "温度 (°C)", [("温度", AIR, selected.temperature)]
            else:
                mass = (selected.residual[0], *selected.residual, selected.residual[-1])
                yname, traces = "局部剩余质量 (%)", [("剩余质量", NITROGEN, mass)]
            xname = "样品深度 (mm)"
        else:
            self.frame_label.set(f"测点 {self.probe_depth:g} mm · 受热面为 0 mm")
            xs = tuple(frame.time for frame in frames)
            xname = "时间 (s)"
            if mode in ("probe_temperature", "probe_residual"):
                readings = slab_probe(self.result, self.probe_depth)
                temp = mode == "probe_temperature"
                yname = "温度 (°C)" if temp else "局部剩余质量 (%)"
                traces = [("测点", AIR if temp else NITROGEN, tuple(row[0 if temp else 1] for row in readings))]
            elif mode == "mass":
                yname = "剩余质量 (%)"
                traces = [("剩余质量", NITROGEN, tuple(f.remaining_mass for f in frames))]
            elif mode == "composition":
                if not frames[0].composition:
                    canvas.create_text(w / 2, h / 2, text="使用两步反应或自定义反应表运行后显示组分", fill=MUTED)
                    return
                yname = "组分占初始质量 (%)"
                labels = (("原材料", FURNACE), ("中间产物", "#794e99"),
                          ("残炭", AIR), ("挥发物", NITROGEN))
                if self.result.component_names:
                    i = self.component_choice.current()
                    traces = [(self.component_choice.get(), NITROGEN, tuple(f.composition[i] for f in frames))]
                else:
                    traces = [(name, color, tuple(f.composition[i] for f in frames))
                              for i, (name, color) in enumerate(labels)]
            elif mode == "loss":
                yname = "质量损失率 (g/(m²·s))"
                traces = [("失重率", AIR, tuple(f.loss_rate for f in frames))]
            else:
                yname = "温度 (°C)"
                traces = [("受热面", AIR, tuple(f.front for f in frames)),
                          ("背面", NITROGEN, tuple(f.back for f in frames))]
        left, right, top, bottom = 59, w - 18, 35, h - 39
        ymin = min(0, *(min(trace) for _, _, trace in traces))
        ymax = max(1, *(max(trace) for _, _, trace in traces))
        if mode in ("temperature", "profile", "probe_temperature"):
            ymin = min(trace_value for _, _, trace in traces for trace_value in trace)
        span = max(1, ymax - ymin)
        xmax = xs[-1]
        xx = lambda x: left + x / xmax * (right - left)
        yy = lambda y: bottom - (y - ymin) / span * (bottom - top)
        canvas.create_text(7, 12, text=yname, anchor="w", fill=INK)
        for i in range(6):
            x = xmax * i / 5
            y = ymin + span * i / 5
            canvas.create_line(xx(x), top, xx(x), bottom, fill=LINE)
            canvas.create_line(left, yy(y), right, yy(y), fill=LINE)
            canvas.create_text(xx(x), bottom + 11, text=f"{x:.4g}", fill=MUTED)
            canvas.create_text(left - 6, yy(y), text=f"{y:.4g}", fill=MUTED, anchor="e")
        canvas.create_text((left + right) / 2, h - 10, text=xname, fill=MUTED)
        if profile:
            x = xx(self.probe_depth)
            canvas.create_line(x, top, x, bottom, fill=ACCENT, dash=(3, 4))
        for i, (name, color, trace) in enumerate(traces):
            coords = [coordinate for x, y in zip(xs, trace) for coordinate in (xx(x), yy(y))]
            canvas.create_line(*coords, fill=color, width=2)
            x = right - 80 * (len(traces) - i)
            canvas.create_line(x, 13, x + 16, 13, fill=color, width=2)
            canvas.create_text(x + 20, 13, text=name, fill=MUTED, anchor="w")


class LabWindow:
    def __init__(self, root, tk, ttk, dialog, messages):
        self.root, self.tk, self.ttk = root, tk, ttk
        self.dialog, self.messages = dialog, messages
        self.inputs = {}
        self.result = None
        self.mode = tk.StringVar(value="temperature")
        self.status = tk.StringVar(value="正在计算示例…")
        self.hint = tk.StringVar(value="移动鼠标查看指定时刻的数值")
        self.indicators = {}
        self.graph = None
        self.cursor_time = None
        root.title("热解与着火模拟")
        root.geometry("1180x795")
        root.minsize(850, 635)
        root.configure(bg=BG)
        root.option_add("*Font", ("Microsoft YaHei UI", 10) if sys.platform == "win32" else ("TkDefaultFont", 10))
        self.style()
        self.layout()
        for variable in self.inputs.values():
            variable.trace_add("write", self.mark_dirty)
        self.root.after(20, self.run)

    def style(self):
        style = self.ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", background=BG, foreground=INK)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TButton", padding=(10, 7))
        style.configure("Primary.TButton", background=ACCENT, foreground="#ffffff")
        style.map("Primary.TButton", background=[("active", "#184887"), ("disabled", "#7295c3")])
        style.configure("TEntry", fieldbackground="#f7f9fc", padding=5)
        style.configure("Panel.TRadiobutton", background=PANEL, padding=(5, 4))

    def text(self, parent, content=None, value=None, bg=PANEL, fg=INK, size=10, bold=False, **options):
        font = ("Microsoft YaHei UI" if sys.platform == "win32" else "TkDefaultFont", size, "bold" if bold else "normal")
        return self.tk.Label(parent, text=content, textvariable=value, font=font, bg=bg, fg=fg, anchor="w", **options)

    def layout(self):
        tk, ttk = self.tk, self.ttk
        banner = tk.Frame(self.root, bg=HEADER)
        banner.pack(fill="x")
        title_block = tk.Frame(banner, bg=HEADER)
        title_block.pack(side="left", padx=22, pady=13)
        self.text(title_block, "热解与着火模拟", bg=HEADER, fg="#fff", size=23, bold=True).pack(anchor="w")
        self.text(title_block, "空气 / 氮气 · 同一炉温程序", bg=HEADER, fg="#c3d7ef", size=10).pack(anchor="w")
        ttk.Button(banner, text="模型与判据", command=self.show_model).pack(side="right", padx=(6, 20), pady=14)
        ttk.Button(banner, text="TG 参数拟合", command=self.show_calibration).pack(side="right", padx=(6, 0), pady=14)
        ttk.Button(banner, text="一维平板热解", command=self.show_slab).pack(side="right", pady=14)

        body = ttk.Frame(self.root, padding=(16, 13, 16, 6))
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        controls = ttk.Frame(body, style="Panel.TFrame", padding=(13, 11))
        controls.grid(row=0, column=0, sticky="nsw", padx=(0, 10))
        self.text(controls, "模拟参数", size=14, bold=True).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))

        defaults = Parameters()
        group_names = {0: "炉温程序", 3: "热解动力学", 6: "模型着火判据"}
        row = 1
        for index, (key, label, unit, _, _) in enumerate(FIELDS):
            if index in group_names:
                self.text(controls, group_names[index], fg=ACCENT, bold=True).grid(
                    row=row, column=0, columnspan=3, sticky="ew", pady=(7, 3))
                row += 1
            var = tk.StringVar(value=f"{getattr(defaults, key):g}")
            self.inputs[key] = var
            self.text(controls, label).grid(row=row, column=0, sticky="w", padx=(0, 5), pady=2)
            entry = ttk.Entry(controls, textvariable=var, width=7, justify="right")
            entry.grid(row=row, column=1, sticky="e", pady=2)
            entry.bind("<Return>", lambda _event: self.run())
            if unit:
                self.text(controls, unit, fg=MUTED, size=9).grid(row=row, column=2, sticky="w", padx=(5, 0))
            row += 1
        self.text(controls, "阈值同时满足并保持指定时长", fg=MUTED, size=9).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(8, 3))
        self.text(controls, "kg₀ 为初始样品质量", fg=MUTED, size=9).grid(row=row + 1, column=0, columnspan=3, sticky="w")
        actions = ttk.Frame(controls, style="Panel.TFrame")
        actions.grid(row=row + 2, column=0, columnspan=3, sticky="ew", pady=(13, 6))
        ttk.Button(actions, text="运行模拟", style="Primary.TButton", command=self.run).pack(side="left", padx=(0, 7))
        ttk.Button(actions, text="恢复示例", command=self.reset).pack(side="left")
        self.export_button = ttk.Button(controls, text="导出曲线 CSV", command=self.save, state="disabled")
        self.export_button.grid(row=row + 3, column=0, columnspan=3, sticky="ew", pady=(2, 0))

        work = ttk.Frame(body)
        work.grid(row=0, column=1, sticky="nsew")
        work.rowconfigure(1, weight=1)
        work.columnconfigure(0, weight=1)
        metrics = ttk.Frame(work)
        metrics.grid(row=0, column=0, sticky="ew", pady=(0, 9))
        for col, (key, title, color) in enumerate((
            ("ignition", "空气 · 模型着火", AIR),
            ("loss", "空气 / 氮气 · 最终残余", INK),
            ("heat", "空气 · 峰值放热率", ACCENT),
        )):
            metrics.columnconfigure(col, weight=1, uniform="metric")
            card = ttk.Frame(metrics, padding=(10, 7), style="Panel.TFrame")
            card.grid(row=0, column=col, sticky="nsew", padx=(0, 7) if col < 2 else (0, 0))
            self.text(card, title, fg=MUTED, size=9).pack(anchor="w")
            value = self.indicators[key] = tk.StringVar(value="—")
            self.text(card, value=value, fg=color, bold=True, size=15).pack(anchor="w", pady=(1, 0))

        plots = ttk.Frame(work, padding=(9, 7), style="Panel.TFrame")
        plots.grid(row=1, column=0, sticky="nsew")
        toolbar = ttk.Frame(plots, style="Panel.TFrame")
        toolbar.pack(fill="x")
        for mode, label in (("temperature", "温度"), ("mass", "残余质量"), ("heat", "放热率")):
            ttk.Radiobutton(toolbar, text=label, variable=self.mode, value=mode,
                            command=self.draw, style="Panel.TRadiobutton").pack(side="left", padx=(0, 6))
        self.canvas = tk.Canvas(plots, bg=PANEL, height=360, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self.draw())
        self.canvas.bind("<Motion>", self.inspect)
        self.canvas.bind("<Leave>", self.leave)
        self.text(plots, value=self.hint, fg=MUTED, size=9,
                  justify="left", wraplength=620).pack(fill="x", padx=6, pady=(0, 2))
        self.text(self.root, value=self.status, bg=BG, fg=MUTED, size=9,
                  justify="left", wraplength=1040).pack(anchor="w", padx=18, pady=(0, 7))

    def show_slab(self):
        existing = getattr(self, "slab", None)
        if existing is not None and existing.window.winfo_exists():
            existing.refresh_label()
            existing.window.lift()
            return
        self.slab = SlabWindow(self)

    def show_calibration(self):
        existing = getattr(self, "calibration", None)
        if existing is not None and existing.window.winfo_exists():
            existing.window.lift()
            return
        self.calibration = CalibrationWindow(self)

    def show_model(self):
        popup = self.tk.Toplevel(self.root)
        popup.title("模型与判据")
        popup.geometry("690x590")
        popup.transient(self.root)
        body = self.ttk.Frame(popup, padding=12)
        body.pack(fill="both", expand=True)
        viewport = self.tk.Text(body, wrap="word", padx=10, pady=8, relief="flat", foreground=INK)
        scrollbar = self.ttk.Scrollbar(body, orient="vertical", command=viewport.yview)
        viewport.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        viewport.pack(side="left", fill="both", expand=True)
        viewport.insert("1.0", MODEL_TEXT)
        viewport.configure(state="disabled")

    def mark_dirty(self, *_args):
        self.export_button.configure(state="disabled")
        self.status.set("参数已修改。点击「运行模拟」更新曲线。")

    def reset(self):
        defaults = Parameters()
        for key, var in self.inputs.items():
            var.set(f"{getattr(defaults, key):g}")
        self.run()

    def run(self):
        values = {}
        for key, label, _, _, _ in FIELDS:
            try:
                values[key] = float(self.inputs[key].get().strip())
            except ValueError:
                self.messages.showerror("输入有误", f"{label}请输入数字。", parent=self.root)
                return
        try:
            params = Parameters(**values)
            air, nitrogen = simulate(params, 1), simulate(params, 0)
        except ValueError as error:
            self.messages.showerror("无法计算", str(error), parent=self.root)
            return
        self.result = (air, nitrogen)
        self.times = [point.time for point in air.points]
        self.cursor_time = None
        self.hint.set("移动鼠标查看指定时刻的数值")
        self.indicators["ignition"].set(f"{air.ignition:.1f} s" if air.ignition is not None else "未达判据")
        self.indicators["loss"].set(f"{air.points[-1].mass:.1f}% / {nitrogen.points[-1].mass:.1f}%")
        self.indicators["heat"].set(f"{air.peak_release:.1f} kW/kg₀")
        onset_air = "未达到" if air.onset is None else f"{air.onset:.1f} s"
        onset_n2 = "未达到" if nitrogen.onset is None else f"{nitrogen.onset:.1f} s"
        self.status.set(f"热解转化率达 5% · 空气 {onset_air} · 氮气 {onset_n2}    |    "
                        "0D 模型 · 内置参数为合成示例")
        self.export_button.configure(state="normal")
        self.draw()

    def save(self):
        if self.result is None:
            return
        path = self.dialog.asksaveasfilename(parent=self.root, defaultextension=".csv",
            initialfile="pyro_ignition_curves.csv", filetypes=[("CSV", "*.csv")])
        if path:
            try:
                export_csv(path, *self.result)
            except OSError as error:
                self.messages.showerror("导出失败", str(error), parent=self.root)
            else:
                self.messages.showinfo("导出完成", f"已保存 {Path(path).name}", parent=self.root)

    def series(self):
        air, n2 = self.result
        mode = self.mode.get()
        if mode == "mass":
            return "残余固体 (%)", [(air, "mass", AIR, "空气", ()), (n2, "mass", NITROGEN, "氮气", (5, 4))]
        if mode == "heat":
            return "放热率 (kW/kg₀)", [
                (air, "gas_release", AIR, "空气", ()), (n2, "gas_release", NITROGEN, "氮气", (5, 4))]
        return "温度 (°C)", [
            (air, "furnace", FURNACE, "炉温", (3, 4)), (air, "solid", AIR, "空气·固相", (5, 4)),
            (air, "gas", AIR, "空气·气相", ()), (n2, "gas", NITROGEN, "氮气·气相", ()),
        ]

    def draw(self):
        canvas = self.canvas
        canvas.delete("all")
        self.graph = None
        if self.result is None or canvas.winfo_width() < 300 or canvas.winfo_height() < 175:
            return
        w, h = canvas.winfo_width(), canvas.winfo_height()
        x0, x1, y0, y1 = 61, w - 20, 52, h - 38
        title, traces = self.series()
        duration = self.times[-1]
        values = [getattr(point, key) for result, key, _, _, _ in traces for point in result.points]
        ceiling = 100.0 if self.mode.get() == "mass" else max(1.0, max(values) * 1.08)
        if self.mode.get() == "heat":
            ceiling = max(ceiling, self.result[0].peak_release * 1.02)
        scale = 10 ** math.floor(math.log10(ceiling / 5))
        tick = next(scale * n for n in (1, 2, 2.5, 5, 10) if scale * n >= ceiling / 5)
        top = tick * math.ceil(ceiling / tick)
        position_x = lambda time: x0 + time / duration * (x1 - x0)
        position_y = lambda value: y1 - value / top * (y1 - y0)
        self.graph = x0, x1, y0, y1, duration

        canvas.create_text(10, 12, text=title, anchor="w", fill=INK)
        legend_x = x0
        for _, _, color, name, dash in traces:
            canvas.create_line(legend_x, 31, legend_x + 17, 31, fill=color, width=2, dash=dash)
            canvas.create_text(legend_x + 22, 31, text=name, anchor="w", fill=MUTED)
            legend_x += 94 if self.mode.get() == "temperature" else 62

        for index in range(round(top / tick) + 1):
            value = index * tick
            y = position_y(value)
            canvas.create_line(x0, y, x1, y, fill=LINE)
            canvas.create_text(x0 - 7, y, text=f"{value:g}", fill=MUTED, anchor="e")
        for index in range(6):
            time = duration * index / 5
            x = position_x(time)
            canvas.create_line(x, y0, x, y1, fill=LINE)
            canvas.create_text(x, y1 + 12, text=f"{time:g}", fill=MUTED)
        canvas.create_text((x0 + x1) / 2, h - 8, text="时间 (s)", fill=MUTED)
        for result, field, color, _, dash in traces:
            coords = [v for point in result.points for v in (position_x(point.time), position_y(getattr(point, field)))]
            canvas.create_line(*coords, fill=color, width=2, dash=dash)

        ignition = self.result[0].ignition
        if ignition is not None:
            x = position_x(ignition)
            canvas.create_line(x, y0, x, y1, fill=AIR, dash=(4, 4))
            canvas.create_text(x - 4 if x > x1 - 105 else x + 4, y0 + 10,
                               text=f"着火判据 {ignition:.1f} s", fill=AIR,
                               anchor="e" if x > x1 - 105 else "w")
        if self.cursor_time is not None:
            x = position_x(self.cursor_time)
            canvas.create_line(x, y0, x, y1, fill=MUTED, dash=(2, 3), tags="crosshair")

    def inspect(self, event):
        if self.graph is None:
            return
        x0, x1, y0, y1, duration = self.graph
        if not (x0 <= event.x <= x1 and y0 <= event.y <= y1):
            self.leave(None)
            return
        time = (event.x - x0) / (x1 - x0) * duration
        index = min(len(self.times) - 1, bisect.bisect_left(self.times, time))
        if index and time - self.times[index - 1] < self.times[index] - time:
            index -= 1
        self.cursor_time = self.times[index]
        a, n = (result.points[index] for result in self.result)
        mode = self.mode.get()
        if mode == "mass":
            contents = f"残余固体 · 空气 {a.mass:.2f}% · 氮气 {n.mass:.2f}%"
        elif mode == "heat":
            contents = f"气相放热 · 空气 {a.gas_release:.2f} · 氮气 {n.gas_release:.2f} kW/kg₀"
        else:
            contents = (f"炉温 {a.furnace:.1f} °C · 空气固相 {a.solid:.1f} °C · "
                        f"空气气相 {a.gas:.1f} °C · 氮气气相 {n.gas:.1f} °C")
        self.hint.set(f"{a.time:.1f} s  |  " + contents)
        self.canvas.delete("crosshair")
        x = x0 + a.time / duration * (x1 - x0)
        self.canvas.create_line(x, y0, x, y1, fill=MUTED, dash=(2, 3), tags="crosshair")

    def leave(self, _event):
        self.cursor_time = None
        self.canvas.delete("crosshair")
        self.hint.set("移动鼠标查看指定时刻的数值")


def main():
    parser = argparse.ArgumentParser(description="热解与着火模拟，默认启动桌面界面。")
    parser.add_argument("--csv", type=Path, help="在命令行运行默认示例并导出 CSV")
    args = parser.parse_args()
    if args.csv:
        params = Parameters()
        air, nitrogen = simulate(params, 1), simulate(params, 0)
        export_csv(args.csv, air, nitrogen)
        print(f"CSV: {args.csv} | Air model ignition: {air.ignition} s")
        return

    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError as error:
        parser.exit(1, f"无法加载 Tkinter，请使用带 Tcl/Tk 的 Python。{error}\n")
    try:
        root = tk.Tk()
    except tk.TclError as error:
        parser.exit(1, f"无法打开桌面图形界面。可使用 --csv 导出示例。{error}\n")
    LabWindow(root, tk, ttk, filedialog, messagebox)
    root.mainloop()


if __name__ == "__main__":
    main()

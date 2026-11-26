import csv
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pyro_ignition_lab as m


class ReactionTableTests(unittest.TestCase):
    def setUp(self):
        self.net = m.ReactionNetwork(
            (m.Component("A", .6), m.Component("B", .4), m.Component("char", 0)),
            (m.NetworkReaction("A", "gases", .5, 0, 0, 0),
             m.NetworkReaction("B", "char", .25, 0, 30, 0)))
        self.p = m.SlabParameters(flux=0, cells=6, step=.1, duration=2)
        self.section = m.SectionParameters(nx=5, nz=6, end_thickness=100)

    def test_independent_stocks_and_mass_balance(self):
        r = m.slab_simulate(self.p, self.net)
        f = r.frames[-1]
        a, b = .6 * math.exp(-1), .4 * math.exp(-.5)
        char = .3 * (.4 - b)
        expected = (a, b, char, 1 - a - b - char)
        self.assertEqual(r.component_names, self.net.names)
        for observed, value in zip(f.composition, expected, strict=True):
            self.assertAlmostEqual(observed, 100 * value, places=10)
        self.assertLess(r.mass_error, 1e-12)
        for vent in ("closed", "both"):
            result = m.section_simulate(self.p, self.net, replace(self.section, vent=vent))
            last = result.frames[-1]
            self.assertAlmostEqual(last.solid, f.remaining_mass)
            self.assertAlmostEqual(last.pore + last.front_out + last.back_out, f.composition[-1])
            self.assertEqual(last.front_out > 0, vent == "both")
            self.assertLess(result.mass_error, 1e-12)

    def test_competing_channels_and_intermediate(self):
        species = (m.Component("A", 1), m.Component("B", 0), m.Component("char", 0))
        rows = (m.NetworkReaction("A", "B", 2, 0, 100, 0), m.NetworkReaction("A", "gases", 1, 0, 0, 0),
                m.NetworkReaction("B", "char", 1, 0, 100, 0))
        net = m.ReactionNetwork(species, rows)
        grid = m.ReactionGrid(net, 1)
        gas, _, _ = grid.advance(0, 300, .3)
        self.assertAlmostEqual(grid.mass[0][1], 2 * gas)
        self.assertAlmostEqual(grid.mass[0][0], math.exp(-.9))
        def run(model, step):
            return m.slab_simulate(replace(self.p, step=step), model).frames[-1].composition
        slow = run(net, .2)
        fast = run(net, .05)
        exact_b = 100 * (math.exp(-2) - math.exp(-6))
        self.assertGreater(fast[2], 0)
        self.assertLess(abs(fast[1] - exact_b), abs(slow[1] - exact_b))
        for x, y in zip(fast, run(replace(net, reactions=rows[::-1]), .05), strict=True):
            self.assertAlmostEqual(x, y, places=10)

    def test_energy_and_single_step_equivalence(self):
        net = m.ReactionNetwork((m.Component("A", 1), m.Component("char", 0)),
                                (m.NetworkReaction("A", "char", 1e6, 55, 25, 260),))
        p = replace(self.p, flux=16)
        r, old = m.slab_simulate(p, net), m.slab_simulate(p, (55, 6, 25))
        self.assertAlmostEqual(r.frames[-1].remaining_mass, old.frames[-1].remaining_mass)
        self.assertAlmostEqual(r.frames[-1].front, old.frames[-1].front)
        self.assertLess(r.energy_error, 1e-5)
        flat = m.section_simulate(p, net, replace(self.section, heated=100))
        for z, center in enumerate(r.frames[-1].temperature[1:-1]):
            self.assertAlmostEqual(flat.frames[-1].temperature[5*z], center, places=4)

    def test_inputs_and_inverse(self):
        for broken in (replace(self.net, components=(m.Component("A", .9),)),
                       replace(self.net, components=(m.Component("A", .5), m.Component("A", .5))),
                       replace(self.net, reactions=(replace(self.net.reactions[0], source="missing"),)),
                       replace(self.net, reactions=(replace(self.net.reactions[0], yield_pct=1),)),
                       replace(self.net, reactions=(replace(self.net.reactions[0], energy=float("nan")),))):
            with self.assertRaises(ValueError):
                broken.validate()
        p, net = m.with_inverse_values(self.p, self.net, ("1.log_z", "0.energy"), (4.5, 95))
        self.assertAlmostEqual(m.inverse_value(p, net, "1.log_z"), 4.5)
        self.assertEqual(net.reactions[0].energy, 95)
        self.assertEqual(net.components, self.net.components)
        self.assertEqual(net.reactions[1].target, "char")

    def test_roundtrip_export_and_legacy_models(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            for model in (self.net, m.SlabMechanism(), (127, 8, 22)):
                r = m.slab_simulate(self.p, model)
                case = m.SlabCase(self.p, model, r)
                m.save_slab_case(path, case)
                if model != self.net:
                    data = json.loads(path.read_text())
                    del data["result"]["component_names"]
                    path.write_text(json.dumps(data))
                self.assertEqual(m.load_slab_case(path), case)
            csvpath = path.with_suffix(".csv")
            m.export_slab(csvpath, m.slab_simulate(self.p, self.net))
            with csvpath.open(encoding="utf-8-sig", newline="") as fin:
                rows = list(csv.DictReader(fin))
            self.assertEqual(float(rows[0]["component_A_pct"]), 60)
            self.assertGreater(float(rows[-1]["component_char_pct"]), 0)
            self.assertGreater(float(rows[-1]["released_volatile_pct"]), 0)

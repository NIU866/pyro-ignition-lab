import csv
import json
import math
import tempfile
import unittest
from dataclasses import astuple, replace
from pathlib import Path

import pyro_ignition_lab as m


class GasSpeciesTests(unittest.TestCase):
    def setUp(self):
        self.p = m.SlabParameters(cells=8, duration=2, step=.1, flux=0)
        self.components = (m.Component("A", .6), m.Component("B", .4), m.Component("residue", 0))
        self.rows = (m.NetworkReaction("A", "gases", .5, 0, 0, 0, gas_yields=(.8, .2)),
                     m.NetworkReaction("B", "residue", .25, 0, 30, 0, gas_yields=(.2, .8)))
        self.gases = (m.GasSpecies("甲", 2e-6), m.GasSpecies("乙", 0))
        self.net = m.ReactionNetwork(self.components, self.rows, gas_species=self.gases)
        self.section = m.SectionParameters(nx=4, nz=8, heated=100, end_thickness=100, porosity_end=30)

    def assert_budget(self, frames):
        initial = frames[0].gas_balance
        for f in frames:
            for b, start in zip(f.gas_balance, initial, strict=True):
                self.assertAlmostEqual(start.pore + b.produced - b.consumed, b.pore + b.front + b.back, places=9)
                self.assertGreaterEqual(min(b.pore, b.produced, b.consumed), 0)

    def test_analytic_production_and_immobile_species(self):
        result = m.slab_simulate(self.p, self.net)
        for f in result.frames:
            a = 60 * -math.expm1(-.5 * f.time)
            b = 40 * .7 * -math.expm1(-.25 * f.time)
            first, second = f.gas_balance
            self.assertAlmostEqual(first.produced, .8 * a + .2 * b)
            self.assertAlmostEqual(second.produced, .2 * a + .8 * b)
            self.assertAlmostEqual(second.pore, second.produced)
            self.assertEqual((first.consumed, first.back, second.front, second.back), (0, 0, 0, 0))
            self.assertAlmostEqual(f.composition[-1], first.produced + second.produced)
        self.assertGreater(first.front, 0)
        self.assert_budget(result.frames)
        self.assertLess(result.mass_error, 1e-12)

    def test_two_absorbing_faces_against_diffusion_series(self):
        p = replace(self.p, cells=80, thickness=5, duration=4, step=.05)
        gases = (m.GasSpecies("X", 1e-6, 1), m.GasSpecies("Y", 4e-6, 2))
        row = m.NetworkReaction("feed", "residue", .1, 0, 100, 0, gas_yields=(0, 0))
        model = m.ReactionNetwork((m.Component("feed", 1), m.Component("residue", 0)), (row,),
                                  gas_species=gases, slab_vent="both", slab_porosity=36)
        result = m.slab_simulate(p, model)
        coarse = m.slab_simulate(replace(p, step=.1), model)
        for j, gas in enumerate(gases):
            balance = result.frames[-1].gas_balance[j]
            start = gas.initial_concentration * 36 / p.density
            # Exact inventory for a uniform slab with zero concentration at both faces.
            tstar = gas.diffusivity * math.sqrt(.36) * p.duration / (p.thickness / 1000) ** 2
            exact = sum(8 * math.exp(-(n * math.pi) ** 2 * tstar) / (n * math.pi) ** 2 for n in range(1, 80, 2))
            error = abs(balance.pore / start - exact)
            self.assertLess(error, .002)
            self.assertLess(error, abs(coarse.frames[-1].gas_balance[j].pore / start - exact))
            self.assertAlmostEqual(balance.front, balance.back)
            self.assertAlmostEqual(balance.pore + 2 * balance.front, start)
        self.assert_budget(result.frames)

    def test_competing_consumers_share_available_inventory_and_heat(self):
        rows = (m.NetworkReaction("A", "gases", 10, 0, 0, 4, gas_yields=(-.5, 1.5)),
                m.NetworkReaction("B", "gases", 10, 0, 0, 8, gas_yields=(-.5, 1.5)))
        gases = (m.GasSpecies("donor", 0, 2), m.GasSpecies("product", 0))
        model = replace(self.net, reactions=rows, gas_species=gases, slab_vent="closed", slab_porosity=50)
        for reactions in (rows, rows[::-1]):
            grid = m.ReactionGrid(replace(model, reactions=reactions), 1, density=20)
            released, heat, original = grid.advance(0, 300, 1)
            self.assertAlmostEqual(released, .1)
            self.assertAlmostEqual(original, released)
            self.assertAlmostEqual(heat, .06 * 4000 + .04 * 8000)
            self.assertAlmostEqual(grid.mass[0][0], .54)
            self.assertAlmostEqual(grid.mass[0][1], .36)
            donor, product = grid.gases.balances()
            self.assertAlmostEqual(donor.consumed, 5)
            self.assertAlmostEqual(donor.pore, 0)
            self.assertAlmostEqual(product.produced, 15)
            self.assertAlmostEqual(grid.advance(0, 300, 1)[0], 0)

    def test_new_gas_is_available_on_next_reaction_step(self):
        rows = (replace(self.rows[0], gas_yields=(1, 0)), replace(self.rows[1], gas_yields=(-.5, 1.5)))
        grid = m.ReactionGrid(replace(self.net, reactions=rows), 1)
        grid.advance(0, 300, 1)
        self.assertAlmostEqual(grid.mass[0][1], .4)
        self.assertEqual(grid.gases.balances()[1].produced, 0)
        grid.advance(0, 300, 1)
        self.assertLess(grid.mass[0][1], .4)
        self.assertGreater(grid.gases.balances()[1].produced, 0)

    def test_boundary_supply_and_closed_faces_with_deformation(self):
        gases = (m.GasSpecies("donor", None, 0, 2), m.GasSpecies("product", None))
        row = replace(self.rows[0], gas_yields=(-.5, 1.5))
        model = replace(self.net, gas_species=gases, reactions=(row,))
        for vent in ("closed", "front", "back", "both"):
            with self.subTest(vent=vent):
                settings = replace(self.section, vent=vent, end_thickness=80, porosity_end=50)
                results = (m.slab_simulate(self.p, replace(model, slab_vent=vent)),
                           m.section_simulate(self.p, model, settings))
                for result in results:
                    self.assert_budget(result.frames)
                    self.assertLess(result.mass_error, 1e-12)
                    donor, product = result.frames[-1].gas_balance
                    self.assertEqual(donor.front < 0, vent in ("front", "both"))
                    self.assertEqual(donor.back < 0, vent in ("back", "both"))
                    self.assertEqual(product.produced > 0, vent != "closed")
                    self.assertAlmostEqual(donor.consumed * 3, product.produced)
                f = results[1].frames[-1]
                self.assertEqual(f.thickness < self.p.thickness, vent != "closed")
                for i, total in enumerate(f.concentration):
                    self.assertAlmostEqual(total, sum(c[i] for c in f.gas_concentrations))
                    self.assertGreaterEqual(min(c[i] for c in f.gas_concentrations), 0)

    def test_uniform_2d_matches_1d_and_legacy_effective_gas(self):
        materials = tuple(m.SpeciesThermal(c.name, 650, 1500, .2, .7) for c in self.components)
        p = replace(self.p, flux=8)
        gases = (m.GasSpecies("X", None), m.GasSpecies("Y", None))
        legacy = replace(self.net, gas_species=(), reactions=tuple(replace(r, gas_yields=()) for r in self.rows))
        for props in ((), materials):
            model = replace(self.net, materials=props, gas_species=gases)
            slab = m.slab_simulate(p, model)
            section = m.section_simulate(p, model, self.section)
            effective = m.section_simulate(p, replace(legacy, materials=props), self.section)
            for a, b, old in zip(slab.frames, section.frames, effective.frames, strict=True):
                self.assertAlmostEqual(a.remaining_mass, b.solid)
                for one, two in zip(a.gas_balance, b.gas_balance, strict=True):
                    for x, y in zip(astuple(one), astuple(two), strict=True):
                        self.assertAlmostEqual(x, y, places=9)
                self.assertAlmostEqual(b.pore, old.pore, places=9)
                for x, y in zip(b.concentration, old.concentration, strict=True):
                    self.assertAlmostEqual(x, y, places=8)

    def test_roundtrip_validation_and_csv(self):
        gases = (m.GasSpecies("甲", None, .7, .2), self.gases[1])
        model = replace(self.net, gas_species=gases, slab_vent="both")
        result = m.slab_simulate(self.p, model)
        section = m.section_simulate(self.p, model, self.section)
        case = m.SlabCase(self.p, model, result, view="gas_balance", section=self.section, selected_gas="乙")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "case.json"
            m.save_slab_case(path, case)
            self.assertEqual(m.load_slab_case(path), case)
            data = path.read_text(encoding="utf-8")
            self.assertEqual(json.loads(data)["version"], 4)
            for key, val in (("gas_yields", "1"), ("gas_yields", [.2, .2]), ("gas_yields", [1])):
                broken = json.loads(data)
                broken["kinetics"]["data"]["reactions"][0][key] = val
                path.write_text(json.dumps(broken))
                with self.assertRaises(ValueError):
                    m.load_slab_case(path)
            for field in ("pore", "produced", "consumed", "front"):
                broken = json.loads(data)
                broken["result"]["frames"][0]["gas_balance"][0][field] += 1
                path.write_text(json.dumps(broken))
                with self.assertRaises(ValueError):
                    m.load_slab_case(path)
            path = path.with_suffix(".csv")
            for write, args in ((m.export_slab, (result,)), (m.export_section, (section,)), (m.export_section, (section, -1))):
                write(path, *args)
                with path.open(encoding="utf-8-sig", newline="") as f:
                    rows = list(csv.DictReader(f))
                self.assertNotIn(None, rows[-1])
                self.assertNotIn(None, rows[-1].values())
                if len(args) == 2:
                    for row in rows:
                        self.assertAlmostEqual(float(row["gas_kg_per_m3_pore"]),
                            float(row["gas_甲_kg_per_m3_pore"]) + float(row["gas_乙_kg_per_m3_pore"]))
                else:
                    for j, name in enumerate(model.gas_names):
                        self.assertAlmostEqual(float(rows[-1][f"gas_{name}_front_net_out_pct_initial"]),
                                               args[0].frames[-1].gas_balance[j].front, places=6)

    def test_invalid_table_inputs(self):
        for bad in (replace(self.net, gas_species=(self.gases[0], self.gases[0])),
                    replace(self.net, gas_species=(m.GasSpecies("A"), self.gases[1])),
                    replace(self.net, gas_species=(m.GasSpecies("gases"), self.gases[1])),
                    replace(self.net, gas_species=(m.GasSpecies("X", -1), self.gases[1])),
                    replace(self.net, gas_species=(m.GasSpecies("X", None, -1), self.gases[1])),
                    replace(self.net, reactions=(replace(self.rows[0], gas_yields=(float("nan"), 1)),)),
                    replace(self.net, reactions=(replace(self.rows[1], yield_pct=100),)),
                    replace(self.net, gas_species=()), replace(self.net, slab_vent="invalid"),
                    replace(self.net, slab_porosity=0)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                bad.validate()

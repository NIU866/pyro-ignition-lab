import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pyro_ignition_lab as lab


def integral(model, n, a):
    """Independent integrated rate laws evaluated away from endpoint cancellation."""
    u = 1 - a
    if model in (0, 2, 9):
        return -math.log(u) if n == 1 else (1 - u**(1-n)) / (1-n)
    if model == 1:
        return (-math.log(u))**n
    if model == 3:
        return a*a
    if model == 4:
        return a + u*math.log(u)
    if model == 5:
        return 1 - 2*a/3 - u**(2/3)
    if model == 6:
        return (1-u**(4/3))/2 - (2/7)*(1-u**(7/3))
    if model == 7:
        return a**n
    return 1-u**n


class KineticTests(unittest.TestCase):
    def setUp(self):
        self.components = (lab.Component("A", 1), lab.Component("B", 0), lab.Component("C", 0))
        self.row = lab.NetworkReaction("A", "B", 1, 0, 100, 0)

    def network(self, *rows):
        return lab.ReactionNetwork(self.components, rows)

    def test_all_ten_models_follow_integrated_isothermal_law(self):
        for model in range(10):
            orders = (.6, 1, 1.8) if model in lab.ORDER_MODELS else (1,)
            for n in orders:
                with self.subTest(model=model, order=n):
                    row = replace(self.row, kinetic_model=model, order=n, prefactor=2.7)
                    grid = lab.ReactionGrid(self.network(row), 1)
                    last = 0.0
                    for a in (.071, .22, .64, .92):
                        dt = (integral(model, n, a) - integral(model, n, last)) / 2.7
                        grid.advance(0, 410, dt)
                        self.assertAlmostEqual(grid.mass[0][0], 1-a, places=10)
                        self.assertAlmostEqual(grid.mass[0][1], a, places=10)
                        last = a

    def test_singular_start_zero_order_and_finite_exhaustion(self):
        for model, factor in ((3, 1), (4, math.sqrt(2)), (5, 3), (6, math.sqrt(3))):
            with self.subTest(model=model):
                self.assertAlmostEqual(lab.kinetic_fraction(model, 1, 0, 1, 1e-28) / 1e-14, factor, places=9)
                grid = lab.ReactionGrid(self.network(replace(self.row, kinetic_model=model)), 1)
                grid.advance(0, 300, 2)
                self.assertEqual(grid.mass[0], [0, 1, 0])
        for n in (0, .3):
            grid = lab.ReactionGrid(self.network(replace(self.row, order=n)), 1)
            grid.advance(0, 300, 2 / (1-n))
            self.assertEqual(grid.mass[0], [0, 1, 0])
        for n in (1-1e-10, 1+1e-10):
            f = lab.kinetic_fraction(0, n, .7, .3, .07)
            self.assertAlmostEqual(f, 1 - math.exp(-.07), places=10)

    def test_mixed_competing_channels_match_independent_solution(self):
        zero = replace(self.row, order=0)
        first = replace(self.row, target="C")
        for t in (.4, 2):
            first_grid = lab.ReactionGrid(self.network(zero, first), 1)
            reordered = lab.ReactionGrid(self.network(first, zero), 1)
            first_grid.advance(0, 300, t)
            reordered.advance(0, 300, t)
            a, b, c = first_grid.mass[0]
            self.assertEqual(first_grid.mass[0], reordered.mass[0])
            self.assertAlmostEqual(a+b+c, 1, places=13)
            # dA/dt = -1-A until exhausted; B forms at constant unit rate.
            self.assertAlmostEqual(a, max(0, 2*math.exp(-t)-1), delta=5e-6)
            self.assertAlmostEqual(b, min(t, math.log(2)), delta=5e-6)
        plane = replace(self.row, kinetic_model=3)
        other = replace(self.row, target="C", order=0)
        grid = lab.ReactionGrid(self.network(plane, other), 1)
        t = .13
        grid.advance(0, 300, t)
        a = 1 - grid.mass[0][0]
        # Integral of dα/dt = 1 + 1/(2α), starting at α=0.
        self.assertAlmostEqual(a - .5*math.log1p(2*a), t, delta=5e-6)
        self.assertAlmostEqual(grid.mass[0][2], t, delta=5e-7)

    def test_growing_intermediate_gets_its_own_conversion_history(self):
        feeds = replace(self.row, prefactor=.25, order=0)
        consumes = lab.NetworkReaction("B", "gases", 1, 0, 0, 0, order=2)
        grid = lab.ReactionGrid(self.network(feeds, consumes), 1)
        dt = .4
        grid.advance(0, 300, dt)
        self.assertAlmostEqual(grid.mass[0][1], .1)
        # At step 2 B has received .1, then another .1 arrives at the end.
        grid.advance(0, 300, dt)
        current_b = .1 / (1 + dt) + .1
        self.assertAlmostEqual(grid.mass[0][1], current_b)
        grid.advance(0, 300, dt)
        expected = current_b / (1 + dt * current_b / .2) + .1
        self.assertAlmostEqual(grid.mass[0][1], expected)

    def test_catalyst_local_history_and_timestep_convergence(self):
        parts = (lab.Component("A", .5), lab.Component("B", .5), lab.Component("C", 0))
        rows = (lab.NetworkReaction("A", "gases", .5, 0, 0, 0),
                lab.NetworkReaction("B", "C", .7, 0, 100, 0, kinetic_model=9, catalyst_strength=2, catalyst="A"))
        net = lab.ReactionNetwork(parts, rows)
        exact = .5 * math.exp(-.7*(3 + 2*(3 - (1-math.exp(-1.5))/.5)))
        errors = []
        for n in (30, 120):
            grid = lab.ReactionGrid(net, 2)
            reversed_grid = lab.ReactionGrid(replace(net, reactions=rows[::-1]), 2)
            for _ in range(n):
                grid.advance(0, 300, 3/n)
                reversed_grid.advance(0, 300, 3/n)
            errors.append(abs(grid.mass[0][1] - exact))
            self.assertEqual(grid.mass, reversed_grid.mass)
            self.assertEqual(grid.mass[1], [.5, .5, 0])
        self.assertLess(errors[1], errors[0]/3.8)

    def test_1d_2d_and_case_versions(self):
        p = lab.SlabParameters(duration=2, cells=6, step=.1, flux=0)
        materials = tuple(lab.SpeciesThermal(c.name, 600, 1400, .2, .8) for c in self.components)
        row = replace(self.row, kinetic_model=1, order=.6, prefactor=.25, yield_pct=35,
                      heat=None, solid_heat=-3000, gas_heat=5000)
        net = replace(self.network(row), materials=materials)
        one = lab.slab_simulate(p, net)
        two = lab.section_simulate(p, net, lab.SectionParameters(nx=4, nz=6, heated=100, end_thickness=100))
        self.assertLess(one.mass_error, 1e-12)
        self.assertLess(two.mass_error, 1e-12)
        self.assertAlmostEqual(one.frames[-1].remaining_mass, two.frames[-1].solid, places=10)
        for x, y in zip(one.frames[-1].temperature[1:-1], two.frames[-1].temperature[::4]):
            self.assertAlmostEqual(x, y, places=7)
        with tempfile.TemporaryDirectory() as folder:
            f = Path(folder) / "case.json"
            case = lab.SlabCase(p, net, one)
            lab.save_slab_case(f, case)
            self.assertEqual(json.loads(f.read_text())["version"], 3)
            self.assertEqual(lab.load_slab_case(f), case)
            for file, version in (("component_network.json", 1), ("thermal_materials.json", 2)):
                old = lab.load_slab_case(Path(__file__).resolve().parents[1] / "examples" / file)
                self.assertTrue(all(r.kinetic_model == 0 and r.order == 1 for r in old.kinetics.reactions))
                lab.save_slab_case(f, old)
                data = json.loads(f.read_text())
                self.assertEqual(data["version"], version)
                self.assertNotIn("order", data["kinetics"]["data"]["reactions"][0])

    def test_inputs_and_inverse_fields(self):
        wrong = (dict(kinetic_model=10), dict(order=-1), dict(order=float("nan")),
                 dict(kinetic_model=1, order=0), dict(kinetic_model=3, order=2),
                 dict(kinetic_model=9, catalyst="unknown", catalyst_strength=1),
                 dict(kinetic_model=9, catalyst="A", catalyst_strength=-1), dict(catalyst_strength=1))
        for kwargs in wrong:
            with self.assertRaises(ValueError):
                self.network(replace(self.row, **kwargs)).validate()
        cat = replace(self.row, kinetic_model=9, catalyst_strength=2, catalyst="A")
        for model, field, visible in ((cat, "0.catalyst_strength", True), (cat, "0.order", True),
                                      (replace(self.row, kinetic_model=5), "0.order", False)):
            self.assertEqual(field in {row[0] for row in lab.inverse_fields(self.network(model))}, visible)
        p = lab.SlabParameters()
        p, net = lab.with_inverse_values(p, self.network(cat), ("0.order", "0.catalyst_strength"), (.8, 3))
        self.assertEqual(lab.inverse_value(p, net, "0.order"), .8)
        self.assertEqual(net.reactions[0].catalyst_strength, 3)

    def test_ga_and_sce_recover_reaction_order(self):
        p = lab.SlabParameters(cells=4, duration=3, step=.5, flux=0)
        row = replace(self.row, prefactor=.5, target="gases", yield_pct=0)
        truth = self.network(replace(row, order=1.6))
        cases = []
        for holdout, duration in ((False, 3), (True, 4)):
            setup = replace(p, duration=duration)
            forward = lab.slab_simulate(setup, truth)
            obs = tuple(lab.Observation(f.time, (f.remaining_mass, None, None, None)) for f in forward.frames)
            cases.append(lab.SlabExperiment(str(duration), setup, obs, holdout))
        for method in lab.METHODS:
            fit = lab.fit_experiments(cases, self.network(row), (("0.order", .7, 2.3),),
                                      (1, 0, 0, 0), method, 100, 19, lambda: False, lambda n: None)
            self.assertAlmostEqual(fit.fit.values[0], 1.6, delta=.008)
            self.assertLess(fit.validation_score, .003)


if __name__ == "__main__":
    unittest.main()

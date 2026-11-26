import json
import math
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pyro_ignition_lab as m


class MaterialTests(unittest.TestCase):
    def setUp(self):
        self.p = m.SlabParameters(cells=6, step=.2, duration=4)
        self.a = m.SpeciesThermal("A", 800, 1800, .20, .8)
        self.b = m.SpeciesThermal("B", 400, 1000, .10, .2)
        self.row = m.NetworkReaction("A", "B", .15, 0, 30, solid_heat=-20000, gas_heat=100000)
        self.net = m.ReactionNetwork((m.Component("A", .6), m.Component("B", .4)),
                                     (self.row,), (self.b, self.a), 300)

    def test_mass_volume_weighting_and_reference(self):
        material = replace(self.a, density_exponent=-1, capacity_exponent=1, conductivity_exponent=2,
                           pore_length=.002)
        net = replace(self.net, materials=(self.b, material))
        mixture = m.SolidMixture(net, 300)
        rho = 1 / (.6 / 800 + .4 / 400)
        self.assertAlmostEqual(mixture.initial_density, rho)
        # At 600 K both densities are 400 kg/m³, hence volume fractions equal mass fractions.
        cell = mixture.cell((.6, .4), 600, 1)
        self.assertAlmostEqual(cell.capacity(600), rho * (.6 * 3600 + .4 * 1000))
        self.assertAlmostEqual(cell.conductivity, .6 * (.8 + .002 * 5.670374419e-8 * 600**3) + .4 * .1)
        self.assertAlmostEqual(cell.emissivity, .6 * .8 + .4 * .2)
        empty = mixture.cell((0, 0), 500, 1.25)
        self.assertEqual(empty.capacity(500), 1500)
        self.assertAlmostEqual(empty.conductivity, .026)
        self.assertEqual(empty.emissivity, 0)

    def test_gaussian_enthalpy_and_power_limit(self):
        phase = replace(self.a, capacity_exponent=-1, melt_heat=85000, melt_temperature=400, melt_variance=9)
        expected = 1800 * 300 * math.log(2) + 85000
        self.assertAlmostEqual(phase.sensible_change(300, 600, 300), expected, places=7)
        self.assertAlmostEqual(phase.sensible_change(600, 300, 300), -expected, places=7)
        for t in (310, 399, 400, 405):
            slope = phase.sensible_change(t - 1e-4, t + 1e-4, 300) / 2e-4
            self.assertAlmostEqual(slope / phase.cp(t, 300), 1.0, places=7)

    def test_split_heats_units_sign_and_endpoints(self):
        for yield_pct, expected in ((0, 100000), (30, 64000), (100, -20000)):
            row = replace(self.row, yield_pct=yield_pct)
            grid = m.ReactionGrid(replace(self.net, reactions=(row,)), 1)
            gas, heat, original = grid.advance(0, 300, .4)
            self.assertAlmostEqual(heat / original, expected)
            self.assertAlmostEqual(gas / original, 1 - yield_pct / 100)
            self.assertAlmostEqual(row.absorbed_per_kg(), expected)
        old = replace(self.net, reactions=(replace(self.row, heat=64, solid_heat=None, gas_heat=None),))
        self.assertEqual(m.slab_simulate(self.p, old), m.slab_simulate(self.p, self.net))

    def test_net_heat_input_matches_whole_sample_enthalpy(self):
        # No A is present. This is an independent check against the known cumulative boundary energy.
        b = replace(self.b, melt_temperature=304, melt_variance=.6, melt_heat=18000, emissivity=0)
        net = replace(self.net, components=(m.Component("A", 0), m.Component("B", 1)), materials=(self.a, b))
        p = replace(self.p, thickness=2, flux=14, front_h=0, back_h=0, duration=3, step=.5)
        r = m.slab_simulate(p, net)
        initial = p.ambient + 273.15
        areal_mass = 400 * .002
        for frame in r.frames:
            specific = []
            for temp in frame.temperature[1:-1]:
                t = temp + 273.15
                gaussian = math.erf((t - 304) / math.sqrt(1.2)) - math.erf((initial - 304) / math.sqrt(1.2))
                specific.append(1000 * (t - initial) + 9000 * gaussian)
            self.assertAlmostEqual(areal_mass * sum(specific) / p.cells, 14000 * frame.time, places=4)
        self.assertGreater(max(r.frames[-1].temperature[1:-1]) + 273.15, 307)
        self.assertLess(r.energy_error, 1e-5)

    def test_1d_2d_mass_heat_and_transport(self):
        a = replace(self.a, melt_temperature=305, melt_variance=4, melt_heat=40000, conductivity_exponent=.4)
        net = replace(self.net, materials=(self.b, a))
        slab = m.slab_simulate(self.p, net)
        section = m.SectionParameters(nx=5, nz=6, end_thickness=100, heated=100, vent="both")
        plane = m.section_simulate(self.p, net, section)
        for one, two in zip(slab.frames, plane.frames, strict=True):
            for i, t in enumerate(two.temperature):
                self.assertAlmostEqual(t, one.temperature[1 + i // 5], places=7)
            self.assertAlmostEqual(two.pore + two.back_out + two.front_out, one.composition[-1], places=7)
        self.assertLess(slab.mass_error, 1e-12)
        self.assertLess(plane.mass_error, 1e-12)
        self.assertLess(slab.energy_error, 1e-5)
        moved = m.section_simulate(self.p, net, replace(section, heated=45, end_thickness=115))
        f = moved.frames[-1]
        self.assertGreater(f.thickness, self.p.thickness)
        self.assertGreater(f.temperature[2], f.temperature[0])
        self.assertGreater(f.front_out, 0)
        self.assertLess(moved.mass_error, 1e-12)

    def test_constant_limit_and_shared_field_independence(self):
        net = replace(self.net, reactions=(replace(self.row, prefactor=1e-150),),
                      components=(m.Component("A", 1), m.Component("B", 0)))
        p = replace(self.p, density=800, heat_capacity=1800, conductivity=.2, emissivity=.8)
        baseline = m.slab_simulate(p, None)
        matched = m.slab_simulate(p, net)
        for a, b in zip(baseline.frames, matched.frames, strict=True):
            self.assertLess(max(abs(x-y) for x, y in zip(a.temperature, b.temperature)), 1e-7)
        other = replace(p, density=2000, heat_capacity=3000, conductivity=1.4, emissivity=.1)
        self.assertEqual(matched, m.slab_simulate(other, net))
        altered = replace(net, materials=(self.b, replace(self.a, heat_capacity=3000)))
        self.assertLess(m.slab_simulate(p, altered).frames[-1].front, matched.frames[-1].front)

    def test_constant_flux_half_space_solution(self):
        # The rear face is more than 7 diffusion lengths away, so its influence
        # is negligible here. Compare all cell centres and both reported faces.
        phase = replace(self.a, heat_capacity=1600, emissivity=0)
        net = replace(self.net, components=(m.Component("A", 1), m.Component("B", 0)),
                      materials=(phase, self.b), reactions=(m.NetworkReaction("B", "gases", 1, 0, 0, 0),))
        p = replace(self.p, thickness=16, duration=8, flux=2.5, front_h=0, back_h=0)
        root_at = math.sqrt(.2 / (800 * 1600) * 8)
        def exact(x):
            eta = x / (2 * root_at)
            return p.ambient + 2500 / .2 * (2 * root_at / math.sqrt(math.pi) * math.exp(-eta**2)
                                             - x * math.erfc(eta))
        errors = []
        for count, step in ((20, .2), (80, .05)):
            r = m.slab_simulate(replace(p, cells=count, step=step), net)
            errors.append(max(abs(t - exact(z / 1000)) for z, t in zip(r.depth_mm, r.frames[-1].temperature)))
        self.assertLess(errors[1], .03)
        self.assertLess(errors[1], errors[0] / 10)

    def test_ga_and_sce_on_per_species_conductivity(self):
        truth = replace(self.net, materials=(self.b, replace(self.a, conductivity=.32)))
        cases = []
        for flux, holdout in ((8, False), (13, True)):
            p = replace(self.p, flux=flux, cells=4, duration=2, step=.5)
            frames = m.slab_simulate(p, truth).frames
            obs = tuple(m.Observation(f.time, (None, None, f.front, f.back)) for f in frames)
            cases.append(m.SlabExperiment(str(flux), p, obs, holdout))
        for method in m.METHODS:
            report = m.fit_experiments(cases, self.net, (("m1.conductivity", .2, .45),),
                                       (0, 0, 1, 1), method, 100, 13, lambda: False, lambda n: None)
            self.assertAlmostEqual(report.fit.values[0], .32, delta=.004)
            self.assertLess(report.validation_score, .01)

    def test_validation_versions_and_fit_parameters(self):
        for invalid in (replace(self.net, reference_temperature=0), replace(self.net, materials=(self.a,)),
                        replace(self.net, materials=(self.a, self.a)), replace(self.net, materials=(self.a, replace(self.b, melt_variance=0))),
                        replace(self.net, reactions=(replace(self.row, heat=10),)),
                        replace(self.net, reactions=(replace(self.row, solid_heat=None),)),
                        replace(self.net, reactions=(replace(self.row, gas_heat=float("inf")),))):
            with self.assertRaises(ValueError):
                invalid.validate()
        with TemporaryDirectory() as folder:
            path = Path(folder) / "case.json"
            case = m.SlabCase(self.p, self.net, m.slab_simulate(self.p, self.net))
            m.save_slab_case(path, case)
            self.assertEqual(json.loads(path.read_text())["version"], 2)
            self.assertEqual(m.load_slab_case(path), case)
            v1 = m.load_slab_case(Path(__file__).resolve().parents[1] / "examples" / "component_network.json")
            m.save_slab_case(path, v1)
            data = json.loads(path.read_text())
            self.assertEqual(data["version"], 1)
            self.assertNotIn("materials", data["kinetics"]["data"])
            self.assertEqual(m.load_slab_case(path), v1)
        keys = {f[0] for f in m.inverse_fields(self.net)}
        self.assertIn("m1.conductivity", keys)
        self.assertNotIn("conductivity", keys)
        self.assertEqual(m.default_thermal_keys(self.net), ("m1.conductivity", "m1.heat_capacity"))
        p, model = m.with_inverse_values(self.p, self.net, ("m1.conductivity", "0.solid_heat"), (.32, -5500))
        self.assertEqual(m.inverse_value(p, model, "m1.conductivity"), .32)
        self.assertEqual(model.reactions[0].solid_heat, -5500)
        self.assertNotEqual(m.slab_simulate(p, model).frames[-1].front, m.slab_simulate(p, self.net).frames[-1].front)


if __name__ == "__main__":
    unittest.main()

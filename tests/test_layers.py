import csv
import json
import math
import tempfile
import unittest
from dataclasses import astuple, replace
from pathlib import Path

import pyro_ignition_lab as m


class LayerTests(unittest.TestCase):
    def setUp(self):
        self.case = m.load_slab_case(Path(__file__).resolve().parents[1] / "examples" / "layer_contact.json")
        self.net = self.case.kinetics
        self.p = self.case.parameters

    def test_interface_fitted_mesh_and_steady_resistance_solution(self):
        r = m.slab_simulate(self.p, self.net)
        f = r.frames[-1]
        face, = r.interfaces
        a, b, q = f.interface_values[0]
        # 2000 W/m² enters at z=0. The film on the back and all three solid/contact
        # resistances are in series. Temperatures follow from q times each resistance.
        self.assertAlmostEqual(face.depth, .7)
        self.assertLess(r.depth_mm[face.cell+1], .7)
        self.assertGreater(r.depth_mm[face.cell+2], .7)
        for observed, exact in ((f.front, 75.5), (a, 65.5), (b, 61.5), (f.back, 50), (q, 2000)):
            self.assertAlmostEqual(observed, exact, delta=.001)
        for frame in r.frames:
            left, right, flux = frame.interface_values[0]
            self.assertAlmostEqual(left - right, face.resistance * flux, places=10)
        self.assertEqual(m.slab_probe(r, face.depth)[-1][0], a)
        self.assertAlmostEqual(m.slab_probe(r, face.depth + 1e-10)[-1][0], b, places=6)
        self.assertLess(r.energy_error, 1e-6)
        self.assertLess(r.mass_error, 1e-12)

    def test_layer_initial_temperature_and_insulated_energy(self):
        layers = (replace(self.net.layers[0], initial_temperature=10),
                  replace(self.net.layers[1], initial_temperature=80))
        model = replace(self.net, layers=layers)
        p = replace(self.p, flux=0, back_h=0, duration=20, cells=7)
        r = m.slab_simulate(p, model)
        mesh = m.LayerMesh(model, p, p.cells)
        materials = {layer.name: mat for layer, mat in zip(layers, model.materials)}
        self.assertEqual((r.frames[0].front, r.frames[0].back), (10, 80))
        self.assertLess(r.frames[0].interface_values[0][2], 0)
        for f in r.frames:
            energy = sum(w * materials[name].density * materials[name].heat_capacity * (t - initial)
                         for w, name, t, initial in zip(mesh.widths, mesh.names, f.temperature[1:-1], r.frames[0].temperature[1:-1]))
            self.assertAlmostEqual(energy, 0, places=6)
        self.assertGreater(r.frames[-1].front, 10)
        self.assertLess(r.frames[-1].back, 80)

    def reacting(self):
        a, b = self.net.names
        return replace(self.net, reactions=(m.NetworkReaction(a, "gases", .4, 0, 0, 3, gas_yields=(.4, .6)),),
                       gas_species=(m.GasSpecies("甲", 2e-6, .4), m.GasSpecies("乙", 0)),
                       layers=(replace(self.net.layers[0], fractions=(.75,.25)), self.net.layers[1]))

    def test_nonuniform_grid_mass_gases_and_2d_equivalence(self):
        model = self.reacting()
        p = replace(self.p, duration=3, step=.1, cells=9)
        setup = replace(self.case.section, nz=p.cells, nx=3, porosity_end=30)
        slab = m.slab_simulate(p, model)
        section = m.section_simulate(p, model, setup)
        density = 1 / (.75/300 + .25/850)
        first_dry = .7 * density
        dry = first_dry + 2.3 * 850
        normalized = first_dry / dry
        self.assertAlmostEqual(slab.frames[0].composition[0], 100 * normalized * .75)
        self.assertAlmostEqual(slab.frames[0].gas_balance[0].pore, .4 * .3 * 3 / dry * 100)
        for a, b in zip(slab.frames, section.frames, strict=True):
            generated = 100 * normalized * .75 * -math.expm1(-.4*a.time)
            self.assertAlmostEqual(a.composition[-1], generated)
            self.assertAlmostEqual(a.remaining_mass, 100 - generated)
            self.assertAlmostEqual(a.remaining_mass, b.solid)
            self.assertAlmostEqual(a.gas_balance[0].produced, generated*.4)
            for z, (t, solid) in enumerate(zip(a.temperature[1:-1], a.residual)):
                for x in range(setup.nx):
                    self.assertAlmostEqual(t, b.temperature[z*setup.nx+x], places=7)
                    self.assertAlmostEqual(solid, b.residual[z*setup.nx+x])
            for one, two in zip(a.gas_balance, b.gas_balance):
                for x,y in zip(astuple(one),astuple(two)):
                    self.assertAlmostEqual(x,y,places=8)
        self.assertLess(slab.mass_error, 1e-12)
        self.assertLess(section.mass_error, 1e-12)
        shrunk = m.section_simulate(p, model, replace(setup, end_thickness=75, porosity_end=60, heated=50))
        self.assertLess(shrunk.frames[-1].thickness, p.thickness)
        self.assertLess(shrunk.mass_error, 1e-12)

    def test_perfect_contact_and_common_property_limit(self):
        p = replace(self.p, duration=3, step=.1, cells=6)
        a, b = self.net.names
        row = m.NetworkReaction(a, b, .2, 0, 60, 4)
        homogeneous = replace(self.net, components=(m.Component(a, .6), m.Component(b, .4)),
                              reactions=(row,), layers=(), materials=())
        layers = (m.MaterialLayer("top", 1, (.6,.4)), m.MaterialLayer("bottom", 2, (.6,.4)))
        for props in ((), self.net.materials):
            old = replace(homogeneous, materials=props)
            new = replace(old, layers=layers)
            uniform, layered = m.slab_simulate(p, old), m.slab_simulate(p, new)
            for x, y in zip(uniform.frames, layered.frames):
                self.assertAlmostEqual(x.front, y.front, places=7)
                self.assertAlmostEqual(x.back, y.back, places=7)
                self.assertAlmostEqual(x.remaining_mass, y.remaining_mass)
                self.assertAlmostEqual(y.interface_values[0][0], y.interface_values[0][1], places=10)

    def test_insulating_contact_does_not_block_gas(self):
        p = replace(self.p, duration=2, step=.2, cells=8, flux=0)
        model = self.reacting()
        models = [replace(model, layers=(replace(model.layers[0], contact=r), model.layers[1])) for r in (0, 100)]
        results = [m.slab_simulate(p, mdl) for mdl in models]
        for a,b in zip(results[0].frames, results[1].frames):
            self.assertAlmostEqual(a.gas_balance[0].front, b.gas_balance[0].front, places=10)
        self.assertNotAlmostEqual(results[0].frames[-1].front, results[1].frames[-1].front, places=4)

    def test_passive_diffusion_on_unequal_cells(self):
        p = replace(self.p, thickness=5, cells=49, step=.05, duration=4, flux=0)
        model = replace(self.net, gas_species=(m.GasSpecies("示踪", 2e-6, 1),), slab_porosity=36, slab_vent="both")
        r = m.slab_simulate(p, model)
        tstar = 2e-6 * math.sqrt(.36) * p.duration / .005**2
        exact = sum(8 * math.exp(-(n*math.pi)**2 * tstar) / (n*math.pi)**2 for n in range(1,64,2))
        start, end = r.frames[0].gas_balance[0], r.frames[-1].gas_balance[0]
        self.assertAlmostEqual(end.pore / start.pore, exact, delta=.003)
        self.assertAlmostEqual((end.pore + end.front + end.back) / start.pore, 1, places=10)

    def test_case_roundtrip_profiles_and_field_coordinates(self):
        p = replace(self.p, duration=2, step=.25, cells=7)
        model = self.reacting()
        result = m.slab_simulate(p, model)
        settings = replace(self.case.section, nz=7)
        case = m.SlabCase(p, model, result, .7, "interfaces", 3, settings, selected_interface=0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"case.json"
            m.save_slab_case(path, case)
            self.assertEqual(m.load_slab_case(path), case)
            content = path.read_text(encoding="utf-8")
            self.assertEqual(json.loads(content)["version"], 5)
            for field in ("depth_mm", "interface_values", "remaining_mass"):
                broken = json.loads(content)
                if field == "depth_mm":
                    broken["result"][field][1] += .04
                elif field == "interface_values":
                    broken["result"]["frames"][-1][field][0][0] += 1
                else:
                    broken["result"]["frames"][-1][field] += 1
                path.write_text(json.dumps(broken))
                with self.assertRaises(ValueError):
                    m.load_slab_case(path)
            def rows():
                with path.open(encoding="utf-8-sig", newline="") as f:
                    return list(csv.DictReader(f))
            m.export_slab(path, result)
            self.assertAlmostEqual(float(rows()[-1]["interface_1_at_0.7_mm_heat_flux_W_m2"]),
                                   result.frames[-1].interface_values[0][2], places=6)
            m.export_profile(path, result, -1)
            points = [r for r in rows() if r["location"].startswith("interface")]
            self.assertEqual([r["location"] for r in points], ["interface_1_front","interface_1_back"])
            self.assertEqual([float(r["depth_mm"]) for r in points], [.7,.7])
            section = m.section_simulate(p, model, settings)
            m.export_section(path, section, -1)
            field = rows()
            for i in range(settings.nz):
                r = field[settings.nx*i]
                self.assertEqual(r["layer"], section.layer_names[i])
                self.assertAlmostEqual(float(r["z_center_mm"]), result.depth_mm[i+1])

    def test_invalid_layers_and_scaled_thickness(self):
        top, base = self.net.layers
        for layers in ((replace(top, thickness=0),base), (replace(top, fractions=(.2,.3)),base),
                       (replace(top, fractions=(1,)),base), (replace(top, initial_temperature=-280),base),
                       (top,replace(base, name=top.name)), (top,replace(base, contact=.01)),
                       (replace(top, contact=-.1),base)):
            with self.subTest(layers=layers), self.assertRaises(ValueError):
                replace(self.net,layers=layers).validate()
        with self.assertRaisesRegex(ValueError, "网格"):
            m.LayerMesh(self.net, self.p, 1)
        for n in (4, 6, 15):
            mesh = m.LayerMesh(self.net, replace(self.p, thickness=6), n)
            self.assertAlmostEqual(mesh.interfaces[0].depth, 1.4)
            self.assertAlmostEqual(sum(mesh.weights), n)

    def test_contact_resistance_fit_with_independent_condition(self):
        p = replace(self.p,cells=4,duration=4,step=.5)
        truth = self.net
        guess = replace(truth, layers=(replace(truth.layers[0],contact=.0005), truth.layers[1]))
        key = "layer0.contact"
        self.assertIn(key, {k for k,*_ in m.inverse_fields(guess)})
        cases = []
        for flux, thickness, holdout in ((3,3,False),(5,4,True)):
            params = replace(p,flux=flux,thickness=thickness)
            frames = m.slab_simulate(params,truth).frames
            obs = tuple(m.Observation(f.time,(None,None,f.front,f.back)) for f in frames)
            cases.append(m.SlabExperiment(str(flux),params,obs,holdout))
        for method in m.METHODS:
            report = m.fit_experiments(cases,guess,((key,.0001,.0045),),(0,0,1,1),method,100,19,lambda:False,lambda _:None)
            self.assertAlmostEqual(report.fit.values[0], truth.layers[0].contact, delta=2e-5)
            self.assertLess(report.validation_score, .002)
            _, fitted = m.with_inverse_values(p,guess,(key,),report.fit.values)
            self.assertAlmostEqual(m.inverse_value(p,fitted,key),report.fit.values[0])

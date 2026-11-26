import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from hw_gate import GATE_SCHEMA, HOST_SCHEMA, LOCKED_RUNTIME, check_gate, check_host, host_receipt, read_fabric
from make_design import ROOT

UUIDS = [f"GPU-{i}" for i in range(8)]


def matrix(link):
    return "\n".join(f"GPU{i}  "+" ".join("X" if i==j else link for j in range(8))
                     for i in range(8))


def healthy_gate():
    driver = "580.178.04"
    host = dict(schema=HOST_SCHEMA,verdict="PASS",driver=driver,gpu_uuids=UUIDS,
                fabric_manager=dict(source="local_service",gpu_uuids=UUIDS,host_driver=driver,
                                    fm_version=driver,active="active",substate="running"))
    return dict(schema=GATE_SCHEMA,verdict="PASS",driver=driver,gpu_uuids=UUIDS,
                host=host,runtime=LOCKED_RUNTIME,fabric=[("N/A","N/A")]*8,
                a100_checks=dict(verdict="PASS",topology=matrix("NV12"),p2p_read=matrix("OK"),
                                 p2p_write=matrix("OK"),load_test=dict(exit_codes=[0]*8,samples=[[["0","1410"]]])))


class HardwareGateTests(unittest.TestCase):
    def test_r580_threshold_requires_full_release(self):
        host = healthy_gate()["host"]
        for v in ("575.99.09","580.65.05","580"):
            with self.subTest(driver=v), self.assertRaisesRegex(ValueError,"below 580.65.06"):
                check_host(dict(host,driver=v))
        host["driver"] = "580.65.06"
        host["fabric_manager"].update(host_driver=host["driver"],fm_version=host["driver"])
        check_host(host)

    def test_fabric_version_and_running_are_independent_requirements(self):
        for patches,error in (({"fm_version":"580.105.08"},"versions differ"),
                              ({"active":"inactive"},"active/running"),
                              ({"substate":"exited"},"active/running"),
                              ({"gpu_uuids":["GPU-other"]*8},"different GPU"),
                              ({"fm_version":None},"full host driver")):
            with self.subTest(patches=patches), self.assertRaisesRegex(ValueError,error):
                host = healthy_gate()["host"]
                host["fabric_manager"].update(patches)
                check_host(host)

    def test_provider_host_evidence_is_scoped(self):
        host = healthy_gate()["host"]
        host["fabric_manager"].update(source="provider_host",host_driver="580.159.03",
            fm_version="580.159.03",reference="provider-request-id",gpu_uuids=list(reversed(UUIDS)))
        check_host(host)
        host["fabric_manager"].pop("reference")
        with self.assertRaisesRegex(ValueError,"support reference"):
            check_host(host)

    def test_standalone_fm_requires_library_pairing_evidence(self):
        host = healthy_gate()["host"]
        host["fabric_manager"].update(source="provider_host",host_driver="620.70.06",
            fm_version="2.0.2",reference="provider-request-id")
        with self.assertRaisesRegex(ValueError,"fabric driver library"):
            check_host(host)
        host["fabric_manager"]["fm_library_version"] = "2.0.2"
        check_host(host)

    def test_r580_host_probe_parses_versions(self):
        results = [
            dict(returncode=0,stdout="\n".join(f"{u}, 580.178.04" for u in UUIDS),stderr=""),
            dict(returncode=0,stdout="ActiveState=active\nSubState=running",stderr=""),
            dict(returncode=0,stdout="Fabric Manager version: 580.178.04",stderr=""),
        ]
        with patch("hw_gate.run",side_effect=results):
            result = host_receipt(None)
        self.assertEqual(result["verdict"],"PASS")
        self.assertEqual(result["fabric_manager"]["fm_version"],"580.178.04")

    def test_previous_a100_raw_fabric_format(self):
        prior = ROOT.parent/"2026-11-17-real-moe-gpu/recovered/preflight"
        self.assertEqual(read_fabric((prior/"nvidia-details.txt").read_text()),[("N/A","N/A")]*8)
        gate = healthy_gate()
        gate["a100_checks"]["topology"] = (prior/"topology.txt").read_text()
        check_gate(gate)
        gate["fabric"][-1] = ("In Progress","N/A")
        with self.assertRaisesRegex(ValueError,"unexpected fabric state"):
            check_gate(gate)

    def test_legacy_pass_wrong_image_and_off_diagonal_x_fail(self):
        with self.assertRaisesRegex(ValueError,"passing hardware gate"):
            check_gate(dict(verdict="PASS",load_test=True))
        gate = healthy_gate()
        wrong = copy.deepcopy(gate)
        wrong["runtime"]["cuda"] = "12.8"
        with self.assertRaisesRegex(ValueError,"runtime"):
            check_gate(wrong)
        gate["a100_checks"]["p2p_write"] = gate["a100_checks"]["p2p_write"].replace("X OK","X X",1)
        with self.assertRaisesRegex(ValueError,"peer matrix"):
            check_gate(gate)

    def test_replay_detects_changed_live_driver(self):
        import json
        from tempfile import TemporaryDirectory
        from replay import hardware

        with TemporaryDirectory() as tmp:
            path = Path(tmp)/"gate.json"
            path.write_text(json.dumps(healthy_gate()))
            with patch("replay.gpu_inventory",return_value=(UUIDS,"580.178.04")):
                hardware(path)
            with patch("replay.gpu_inventory",return_value=(UUIDS,"580.182.01")):
                with self.assertRaisesRegex(ValueError,"changed after hardware qualification"):
                    hardware(path)


if __name__=="__main__":
    unittest.main()


class TestLoadClockRule(unittest.TestCase):
    """Legacy A100 load check: a low clock is accepted only when the full-limit power cap explains it."""

    def setUp(self):
        import importlib.util
        from hw_gate import LEGACY
        spec = importlib.util.spec_from_file_location("legacy_hw_gate",LEGACY)
        self.legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.legacy)

    def test_power_cap_at_the_full_limit_explains_a_low_clock(self):
        capped = self.legacy.power_capped
        ramp = ["0","1155","70.10","0x0000000000000000"]
        load = ["0","1230","401.50","0x0000000000000004"]
        tail = ["0","1395","103.46","0x0000000000000004"]
        rows = [ramp]+[load]*10+[tail]   # the pattern recorded on the 2026-11-26 node
        self.assertTrue(capped(rows,400.,400.,400.))
        # A lowered limit, a limit below the device maximum, or any slowdown reason remains a failure.
        self.assertFalse(capped(rows,300.,400.,400.))
        self.assertFalse(capped(rows,350.,350.,400.))
        for mask in ("0x0000000000000024","0x0000000000000044","0x000000000000000c","0x0000000000000084"):
            self.assertFalse(capped([ramp]+[load]*9+[["0","1230","401.5",mask]],400.,400.,400.))
        # A low clock without full power draw is not explained by the cap.
        self.assertFalse(capped([["0","1230","250.0","0x0000000000000004"]]*10,400.,400.,400.))
        self.assertFalse(capped([["0","1230","401.5","0x0000000000000000"]]*10,400.,400.,400.))
        self.assertFalse(capped([["0","1230","401.5"]]*10,400.,400.,400.))
        self.assertFalse(capped(rows,None,400.,400.))


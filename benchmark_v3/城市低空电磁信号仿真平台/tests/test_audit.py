import json
import numpy as np
import pytest
from cityrf.config import Scenario
from cityrf.runner import run_scenario
from cityrf.audit import audit_run


def make_run(tmp_path):
    s=Scenario(emitters=[dict(id="tx",position_m=[0,0,10])],
               receivers=[dict(id="rx",position_m=[100,0,10])])
    run_scenario(s,tmp_path/"run")
    return tmp_path/"run"


def test_audit_accepts_physical_capture(tmp_path):
    report=audit_run(make_run(tmp_path))
    assert report["status"]=="verified"
    assert report["stations"][0]["samples"]==50000


def test_audit_rejects_changed_capture(tmp_path):
    root=make_run(tmp_path)
    path=root/"public"/"iq_rx.npy"
    x=np.load(path)
    x[0,0]+=1
    np.save(path,x)
    with pytest.raises(ValueError,match="hash"):
        audit_run(root)

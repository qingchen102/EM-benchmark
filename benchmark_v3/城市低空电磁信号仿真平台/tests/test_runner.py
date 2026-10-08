import json
import numpy as np
import pytest
from cityrf.config import Scenario
from cityrf.runner import run_scenario
from cityrf.analysis import analyze_observations


def make():
    return Scenario(
        emitters=[dict(id="uav",position_m=[0,0,30],power_dbm=10),
                  dict(id="jam",role="interference",position_m=[20,0,2],power_dbm=0,
                       components=[dict(kind="tone",offset_hz=200000)])],
        receivers=[dict(id="r1",position_m=[100,0,5]),dict(id="r2",position_m=[0,200,5])],
        tasks=[dict(id="protect",receiver_ids=["r1","r2"],desired_emitter_ids=["uav"])])


def test_multistation_end_to_end(tmp_path):
    root=tmp_path/"run"
    summary=run_scenario(make(),root)
    assert summary["num_samples"] == 50000
    obs=json.loads((root/"public"/"observations.json").read_text())
    assert len(obs["receivers"]) == 2
    x=np.load(root/"public"/obs["receivers"][0]["iq_file"])
    y=np.load(root/"public"/obs["receivers"][1]["iq_file"])
    assert x.shape == (16,50000) and x.dtype == np.complex64
    assert np.isfinite(x).all() and not np.array_equal(x,y)
    text=json.dumps(obs)
    assert "jam" not in text and "tx_power" not in text and "desired_emitter_ids" not in text
    assert (root/"private"/"truth.json").exists()
    report=analyze_observations(root/"public")
    assert len(report["stations"]) == 2


def test_reproducibility(tmp_path):
    run_scenario(make(),tmp_path/"a")
    run_scenario(make(),tmp_path/"b")
    np.testing.assert_array_equal(np.load(tmp_path/"a/public/iq_r1.npy"),
                                  np.load(tmp_path/"b/public/iq_r1.npy"))


def test_nonempty_output_refused(tmp_path):
    (tmp_path/"keep.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        run_scenario(make(),tmp_path)
    assert (tmp_path/"keep.txt").read_text() == "keep"


def test_resource_budget_before_output(tmp_path):
    config=make().model_copy(update={"sample_rate_hz":40e6,"max_working_mb":64})
    with pytest.raises(ValueError,match="memory"):
        run_scenario(config,tmp_path/"large")
    assert not (tmp_path/"large").exists()


def test_no_wrapped_echo_at_capture_start(tmp_path):
    config=make()
    config.propagation.backend="tapped"
    config.propagation.tap_delays_s=[.0005]
    config.propagation.tap_gains_real=[1]
    config.emitters=config.emitters[:1]
    config.emitters[0].components[0].kind="tone"
    config.receivers[0].frontend.noise_enabled=False
    run_scenario(config,tmp_path/"delayed")
    x=np.load(tmp_path/"delayed/public/iq_r1.npy")
    # A continuous source existed before t=0; delay must not manufacture a startup silence.
    np.testing.assert_allclose(abs(x[0,:100]),abs(x[0,1000]),rtol=1e-5)


def test_public_metadata_rejects_path_id(tmp_path):
    from cityrf.analysis import load_public
    run_scenario(make(),tmp_path/"run")
    path=tmp_path/"run/public/observations.json"
    data=json.loads(path.read_text())
    data["receivers"][0]["receiver_id"]="../escape"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="ID"):
        load_public(path.parent)

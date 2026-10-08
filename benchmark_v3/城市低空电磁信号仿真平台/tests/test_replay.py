import numpy as np
from cityrf.config import Scenario
from cityrf.runner import run_scenario
from cityrf.replay import replay_scenario


def test_frozen_channel_replay_is_bit_exact(tmp_path):
    s=Scenario(emitters=[dict(id="tx",position_m=[0,0,10])],
               receivers=[dict(id="rx",position_m=[100,0,10])])
    run_scenario(s,tmp_path/"original")
    summary=replay_scenario(tmp_path/"original",tmp_path/"replay")
    assert summary["execution_mode"]=="frozen_channel_replay"
    np.testing.assert_array_equal(np.load(tmp_path/"original/public/iq_rx.npy"),
                                  np.load(tmp_path/"replay/public/iq_rx.npy"))

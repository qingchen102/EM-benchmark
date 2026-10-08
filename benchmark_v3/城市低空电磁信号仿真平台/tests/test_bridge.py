import json
import sys
from pathlib import Path
import pytest
from cityrf.config import Scenario
from cityrf.runner import run_scenario
from cityrf.export import legacy_manifest


def test_legacy_repository_can_load_generated_capture(tmp_path):
    old=Path(__file__).resolve().parents[2]/"基本干扰分析功能Agent"/"src"
    if not old.exists():
        pytest.skip("legacy project not available")
    sys.path.insert(0,str(old))
    try:
        from interference_agent.schemas import IQSampleManifest
        from interference_agent.runtime import IQRepository, RunArtifactStore
        from interference_agent.tools.inspection import inspect_iq, InspectIQArgs
        from interference_agent.tools.representations import compute_spectrum, ComputeSpectrumArgs
    except ImportError:
        pytest.skip("legacy Agent dependencies not installed")
    s=Scenario(emitters=[dict(id="tx",position_m=[0,0,10])],
               receivers=[dict(id="rx",position_m=[100,0,10])])
    run_scenario(s,tmp_path/"run")
    public=tmp_path/"run"/"public"
    obs=json.loads((public/"observations.json").read_text())["receivers"][0]
    manifest=IQSampleManifest.model_validate(legacy_manifest(public,obs))
    repo=IQRepository(allowed_roots=[public])
    ref=repo.register(manifest)
    inspected=inspect_iq(InspectIQArgs(source_ref=ref),repository=repo)
    assert inspected.status.value == "success"
    # The default is an 8+8 cross. Old scalar-position MUSIC must not be advertised.
    assert manifest.array_geometry.geometry_type == "custom"
    assert inspected.data["capabilities"]["doa"] is False
    assert inspected.data["capabilities"]["spatial_evidence"] is False
    result=compute_spectrum(ComputeSpectrumArgs(source_ref=ref,method="welch_psd",
                            nfft=2048,segment_length=2048,scaling="power_density"),
                            repository=repo,artifact_store=RunArtifactStore("bridge"))
    assert result.status.value == "success"

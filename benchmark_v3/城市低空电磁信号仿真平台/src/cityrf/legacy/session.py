"""Small Python adapter over the original 13 numerical executors."""
import json
from pathlib import Path
import numpy as np
from .runtime import IQRepository,RunArtifactStore
from .schemas import IQSampleManifest,ProtectedChannel
from .tools import inspection,representations,detection,estimation,spatial


ENTRIES={
    'inspect_iq':(inspection.InspectIQArgs,inspection.inspect_iq,True,False),
    'compute_spectrum':(representations.ComputeSpectrumArgs,representations.compute_spectrum,True,True),
    'compute_time_frequency':(representations.ComputeTimeFrequencyArgs,representations.compute_time_frequency,True,True),
    'detect_activity_intervals':(detection.DetectActivityIntervalsArgs,detection.detect_activity_intervals,True,False),
    'detect_emissions':(detection.DetectEmissionsArgs,detection.detect_emissions,False,True),
    'estimate_source_count':(detection.EstimateSourceCountArgs,detection.estimate_source_count,False,True),
    'estimate_frequency':(estimation.EstimateFrequencyArgs,estimation.estimate_frequency,True,True),
    'estimate_bandwidth':(estimation.EstimateBandwidthArgs,estimation.estimate_bandwidth,True,True),
    'estimate_pulse_parameters':(estimation.EstimatePulseParametersArgs,estimation.estimate_pulse_parameters,True,False),
    'extract_modulation_features':(estimation.ExtractModulationFeaturesArgs,estimation.extract_modulation_features,True,True),
    'measure_power_relations':(estimation.MeasurePowerRelationsArgs,estimation.measure_power_relations,True,True),
    'compute_spatial_evidence':(spatial.ComputeSpatialEvidenceArgs,spatial.compute_spatial_evidence,True,True),
    'estimate_doa':(spatial.EstimateDoaArgs,spatial.estimate_doa,False,True),
}


class LegacySession:
    """One capture and in-memory artifacts; keep dependent calls in one script."""

    def __init__(self,iq_path,*,sample_rate_hz,center_frequency_hz=None,
                 protected_channel=None,expected_desired_source_count=None,
                 array_geometry=None,synchronized=None,calibration=None):
        path=Path(iq_path).resolve()
        array=np.load(path,mmap_mode='r',allow_pickle=False)
        if array.ndim not in (1,2):
            raise ValueError('IQ must be samples or [channels,samples]')
        manifest=IQSampleManifest(sample_id='capture',iq_path=str(path),dtype=str(array.dtype),
            layout='samples' if array.ndim==1 else 'channel_first',sample_rate_hz=sample_rate_hz,
            center_frequency_hz=center_frequency_hz,protected_channel=protected_channel,
            expected_desired_source_count=expected_desired_source_count,array_geometry=array_geometry,
            reference_frequency_hz=center_frequency_hz,synchronized=synchronized,calibration=calibration or {})
        self.repository=IQRepository([path.parent])
        self.source_ref=self.repository.register(manifest)
        self.artifact_store=RunArtifactStore('legacy-analysis')

    @classmethod
    def from_observations(cls,path,receiver_id,*,task_id=None):
        path=Path(path).resolve()
        metadata=json.loads(path.read_text(encoding='utf-8'))
        receiver=next((r for r in metadata['receivers'] if r['receiver_id']==receiver_id),None)
        if receiver is None:
            raise ValueError('Unknown receiver_id')
        if receiver.get('layout')!='channels_samples':
            raise ValueError('Expected cityrf channels_samples layout')
        iq_path=(path.parent/receiver['iq_file']).resolve()
        if not iq_path.is_relative_to(path.parent):
            raise ValueError('IQ path escapes observation directory')
        protected=None
        expected=None
        if task_id is not None:
            task=next((t for t in metadata.get('tasks',[]) if t['id']==task_id),None)
            if task is None or receiver_id not in task['receiver_ids']:
                raise ValueError('Task unavailable for receiver')
            protected=ProtectedChannel(center_offset_hz=task['protected_center_frequency_hz']-receiver['center_frequency_hz'],
                                       nominal_bandwidth_hz=task['protected_bandwidth_hz'])
            expected=task.get('expected_desired_source_count')
        # The old schema cannot express a 3-D cross array; do not pretend it is a ULA.
        return cls(iq_path,sample_rate_hz=receiver['sample_rate_hz'],
                   center_frequency_hz=receiver['center_frequency_hz'],protected_channel=protected,
                   expected_desired_source_count=expected,synchronized=receiver.get('synchronized'))

    @staticmethod
    def names():
        return list(ENTRIES)

    @staticmethod
    def schema(name):
        return ENTRIES[name][0].model_json_schema()

    def call(self,name,**arguments):
        model,function,needs_repository,needs_store=ENTRIES[name]
        if 'source_ref' in arguments and arguments['source_ref']!=self.source_ref:
            raise ValueError('source_ref belongs to another session')
        payload=arguments if name=='estimate_doa' else {'source_ref':self.source_ref,**arguments}
        args=model.model_validate(payload)
        dependencies={}
        if needs_repository:
            dependencies['repository']=self.repository
        if needs_store:
            dependencies['artifact_store']=self.artifact_store
        return function(args,**dependencies).model_dump(mode='json')

"""Preferred checked API for Wiki methods. Low-level modules remain available."""
from .checked_signal import (SignalBlock,signal_block,load_record,select_band,crop,
                             assert_comparable,save_result,brief,save_signal,load_signal)
from .checked_features import (quality,spectrum,time_frequency,ridge_fit,waveform_features,
                               activity,activity_contrast,periodic_decomposition,cp_structure,
                               spatial_evidence,joint_features,structure_table,beamform,template_residual)
from .checked_spatial import (single_path_fit,path_hypotheses,mirror_direction,
                              path_observations,locate_paths)
from .reflection import Geometry as _LegacyGeometry


class Geometry:
    """Building queries for checked path localization; use locate_paths for inference.

    The historical one-direction-per-station locator remains in reflection.py for
    explicit legacy experiments, but is not part of this checked API.
    """
    def __init__(self, triangles):
        self._geometry = _LegacyGeometry(triangles)

    @classmethod
    def load(cls, path='input/geometry/mesh.npz'):
        return cls(_LegacyGeometry.load(path).triangles)

    @property
    def triangles(self):
        return self._geometry.triangles.copy()

    def first_hit(self, origin, direction, max_distance=5000.):
        return self._geometry.first_hit(origin, direction, max_distance)

    def visible(self, a, b):
        return self._geometry.visible(a, b)

    def backtrace(self, origin, direction, max_distance=2000.):
        return self._geometry.backtrace(origin, direction, max_distance)


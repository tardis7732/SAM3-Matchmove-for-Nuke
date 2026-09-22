"""Regression checks for selected motion components and reversible crop geometry."""
from pathlib import Path
import math
import sys
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'daemon'))
import memory_tracking as tracking
# memory_tracking may add an installed helper repo: import this checkout's export code explicitly.
import importlib.util
spec = importlib.util.spec_from_file_location('crop_export_under_test', ROOT / 'nuke/sam3_matchmove_export.py')
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


class MotionCropTests(unittest.TestCase):
    def job(self, mode):
        frames = []
        for i, side in enumerate((10, 20)):
            mask = np.zeros((64, 64), bool)
            mask[10:10+side, 12:12+side] = True
            frames.append(dict(frame=i+1, mask=mask, gray=np.zeros((64, 64), np.uint8)))
        return dict(width=64, height=64, frames=frames, reference_frame=1,
                    tracking_mode=mode, smoothing_window=1, crop_margin=1.2, fixed_crop=True)

    def test_translation_excludes_size_and_rotation(self):
        result = tracking.track_memory(self.job('translation'))
        row = result['frames'][1]
        self.assertEqual(row['translate'], [5., -5.])
        self.assertEqual((row['scale'], row['rotate']), (1., 0.))

    def test_scale_changes_crop_even_with_fixed_crop_box(self):
        result = tracking.track_memory(self.job('translation_scale'))
        reference, row = result['frames']
        self.assertAlmostEqual(row['scale'], 2.)
        self.assertEqual(row['rotate'], 0.)
        self.assertEqual(row['crop_box'][2]-row['crop_box'][0], reference['crop_box'][2]-reference['crop_box'][0])
        first = export.crop_motion_mapping(reference, reference, 720, 720)
        second = export.crop_motion_mapping(row, reference, 720, 720)
        self.assertAlmostEqual(second[1][0] / first[1][0], 2.)

    def test_rotation_includes_estimated_scale(self):
        theta = math.radians(30)
        pair = np.eye(3)
        pair[:2, :2] = 1.5 * np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        with patch.object(tracking, '_estimate_pair', return_value=(pair, 1., None)):
            result = tracking.track_memory(self.job('translation_scale_rotation'))
        row = result['frames'][1]
        self.assertAlmostEqual(row['scale'], 1.5)
        self.assertAlmostEqual(row['rotate'], -30.)
        self.assertEqual(row['status'], 'features')
        # Derived corners must retain the solved scale as well as rotation.
        corners = np.array(row['cornerpin'])
        self.assertAlmostEqual(np.linalg.norm(corners[1]-corners[0]), 15.)

    def test_crop_center_scale_and_inverse_for_all_aspects(self):
        reference = dict(crop_box=[300., 200., 504., 404.], center=[402., 302.], scale=1., rotate=0.)
        row = dict(crop_box=[300., 200., 504., 404.], center=[620., 430.], scale=.418, rotate=37.)
        for width, height in ((720, 720), (1280, 720), (720, 1280)):
            with self.subTest(size=(width, height)):
                offset, scale, angle = export.crop_motion_mapping(row, reference, width, height)
                matrix = tracking.similarity_matrix(offset, scale[0], math.radians(angle), [0., 0.])
                np.testing.assert_allclose(matrix @ [width/2, height/2, 1], [620., 430., 1.], atol=1e-9)
                a = matrix @ [100., 50., 1.]
                b = matrix @ [101., 50., 1.]
                self.assertAlmostEqual(np.linalg.norm(b-a), 204/min(width, height)*.418)
                np.testing.assert_allclose(np.linalg.inv(matrix) @ a, [100., 50., 1.], atol=1e-9)


if __name__ == '__main__':
    unittest.main()

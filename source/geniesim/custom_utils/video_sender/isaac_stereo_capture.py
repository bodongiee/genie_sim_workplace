# =============================================================================
# Isaac Sim stereo camera capture for VR side-by-side (SBS) streaming.
#
#Creates left/right eye cameras offset by the stereo baseline
#
#SBS frame layout:
#    [  Left Eye (W x H)  |  Right Eye (W x H)  ]    Total output: (2*W) x H
#
#Usage:
#    capture = IsaacStereoCapture(
#    stage=stage,
#        parent_camera_path="/robot/head_link2/zed/Head_Camera",
#        eye_resolution=(1280, 720),
#        ipd=0.063,  # ZED Mini baseline
#    )
#    # After world.step(render=True):
#    sbs_frame = capture.get_sbs_frame()  # numpy (H, 2*W, 4) RGBA
# =============================================================================

import numpy as np
from typing import Optional, Tuple


# ZED Mini defaults
ZED_MINI_BASELINE = 0.063
ZED_MINI_FOCAL_LENGTH = 8.484 
ZED_MINI_H_APERTURE = 20.955   
ZED_MINI_V_APERTURE = 9.214


class IsaacStereoCapture:

    def __init__(
        self,
        stage,
        parent_camera_path: str,
        eye_resolution: Tuple[int, int] = (1280, 720),
        ipd: float = ZED_MINI_BASELINE,
        focal_length: float = ZED_MINI_FOCAL_LENGTH,
        h_aperture: float = ZED_MINI_H_APERTURE,
        v_aperture: float = ZED_MINI_V_APERTURE,
        clipping_range: Tuple[float, float] = (0.1, 9.0),
    ):

        from pxr import UsdGeom, Sdf, Gf

        self._eye_w, self._eye_h = eye_resolution
        self._ipd = ipd

        parent_prim = stage.GetPrimAtPath(parent_camera_path)
        if not parent_prim.IsValid():
            raise ValueError(f"Camera prim not found: {parent_camera_path}")
        parent_xform_path = str(parent_prim.GetParent().GetPath())

        # Copy the orientation from the original camera
        orig_cam = UsdGeom.Xformable(parent_prim)
        orig_ops = orig_cam.GetOrderedXformOps()
        orig_orient = None
        for op in orig_ops:
            if "orient" in op.GetOpName():
                orig_orient = op.Get()
                break

        # Create left and right eye cameras
        half_ipd = ipd / 2.0

        left_cam_path = f"{parent_xform_path}/StereoLeft"
        right_cam_path = f"{parent_xform_path}/StereoRight"

        for cam_path, x_offset in [(left_cam_path, -half_ipd),
                                    (right_cam_path, half_ipd)]:
            cam = UsdGeom.Camera.Define(stage, Sdf.Path(cam_path))
            cam.CreateFocalLengthAttr(focal_length)
            cam.CreateHorizontalApertureAttr(h_aperture)
            cam.CreateVerticalApertureAttr(v_aperture)
            cam.CreateClippingRangeAttr(Gf.Vec2f(*clipping_range))
            cam.CreateProjectionAttr("perspective")

            xf = UsdGeom.Xformable(cam.GetPrim())
    
            if orig_orient is not None:
                q = Gf.Quatf(float(orig_orient.GetReal()),
                             Gf.Vec3f(*[float(x) for x in orig_orient.GetImaginary()]))
                xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(q)
            xf.AddTranslateOp().Set(Gf.Vec3d(x_offset, 0, 0))

        print('\033[42m\033[37m' + f"[StereoCapture] Created stereo cameras:" + '\033[0m')
        print(f"  Left:  {left_cam_path}")
        print(f"  Right: {right_cam_path}")
        print(f"  IPD: {ipd * 1000:.2f}mm  Per-eye: {self._eye_w}x{self._eye_h}")

        # Set up render products and annotators
        import omni.replicator.core as rep

        self._left_rp = rep.create.render_product(left_cam_path, resolution=eye_resolution)
        self._right_rp = rep.create.render_product(right_cam_path, resolution=eye_resolution)


        self._left_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self._right_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")

        # Pre-allocate SBS output buffer
        self._sbs_buffer = np.zeros((self._eye_h, self._eye_w * 2, 4), dtype=np.uint8)

        print('\033[42m\033[37m' + f"[StereoCapture] Ready. SBS output: {self._eye_w * 2}x{self._eye_h}" + '\033[0m')

    @property
    def sbs_width(self) -> int:
        return self._eye_w * 2

    @property
    def sbs_height(self) -> int:
        return self._eye_h

    @property
    def eye_resolution(self) -> Tuple[int, int]:
        return (self._eye_w, self._eye_h)

    def _read_annotator(self, annotator) -> Optional[np.ndarray]:
        try:
            data = annotator.get_data()
            if data is None:
                return None
            if isinstance(data, dict):
                data = data.get("data", data)

            arr = np.asarray(data)
            if arr.size == 0:
                return None

            if arr.ndim == 1:
                expected = self._eye_w * self._eye_h * 4
                if arr.size == expected:
                    arr = arr.reshape((self._eye_h, self._eye_w, 4))
                else:
                    return None

            return arr[:, :, :4].astype(np.uint8)
        except Exception:
            return None

    def get_sbs_frame(self) -> Optional[np.ndarray]:
        left = self._read_annotator(self._left_annotator)
        right = self._read_annotator(self._right_annotator)

        if left is None or right is None:
            return None

        self._sbs_buffer[:, :self._eye_w, :] = left
        self._sbs_buffer[:, self._eye_w:, :] = right
        return self._sbs_buffer

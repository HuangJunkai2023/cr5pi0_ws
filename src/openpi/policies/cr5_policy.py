"""CR5 robot policy transforms."""

import dataclasses
import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 (H,W,C) format."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class CR5Inputs(transforms.DataTransformFn):
    """CR5 input transform.
    
    CR5 is a 6-DOF arm with gripper (7 action dims total).
    Following the UR5 pattern: states will be automatically padded to 32 dims by PadStatesAndActions.
    """
    
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        top_image = _parse_image(data["image"])
        
        inputs = {
            "state": data["state"],  # Will be padded to 32 dims automatically
            "image": {
                "base_0_rgb": top_image,
                "left_wrist_0_rgb": np.zeros_like(top_image),
                "right_wrist_0_rgb": np.zeros_like(top_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }
        
        if "actions" in data:
            inputs["actions"] = data["actions"]  # Will be padded to 32 dims automatically
        
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        
        return inputs


@dataclasses.dataclass(frozen=True)
class CR5Outputs(transforms.DataTransformFn):
    """Transform CR5 outputs back to robot-specific format.
    
    Output format:
    - actions: [7] (6 joint velocities + 1 gripper)
    
    Note: Pi0 Base model outputs 32-dim actions, but CR5 only uses the first 7 dims.
    This is the same approach as UR5 (see examples/ur5).
    """
    
    def __call__(self, data: dict) -> dict:
        # Pi0 model outputs 32-dim actions, but CR5 robot only has 7 DoF (6 joints + gripper)
        # Follow the same pattern as UR5: only return the first 7 dimensions
        return {"actions": np.asarray(data["actions"][:, :7])}
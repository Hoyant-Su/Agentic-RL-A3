from io import BytesIO
from typing import Optional, Union
import torch
from PIL import Image
from qwen_vl_utils import fetch_image, fetch_video

def process_image(image: Union[dict, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if "bytes" in image:
        assert "image" not in image, "Cannot have both `bytes` and `image`"
        image["image"] = BytesIO(image["bytes"])
    return fetch_image(image)

VIDEO_FORMAT_HELP = """Currently, we only support the video formats introduced in qwen2-vl.
eg.
{
    "type": "video",
    "video": [
        "file:///path/to/frame1.jpg",
        "file:///path/to/frame2.jpg"
    ]
}
{
    "type": "video",
    "video": "file:///path/to/video.mp4"
}
# Defaults to fps=2, min_frames=4, max_frames=768
{
    "type": "video",
    "video": "file:///path/to/video.mp4",
    "fps": 2,
    "min_frames": 1,
    "max_frames": 32
}
"""

def process_video(
    video: dict,
    nframes: Optional[int] = None,
    fps: Optional[float] = None,
    fps_min_frames: Optional[int] = None,
    fps_max_frames: Optional[int] = None,
) -> torch.Tensor:
    if not isinstance(video, dict) or "video" not in video:
        raise NotImplementedError(VIDEO_FORMAT_HELP)
    assert nframes is None or fps is None, "Can't use both `nframes` or `fps`"
    video = dict(video)
    contains_sampling_rules = "nframes" in video or "fps" in video
    if not contains_sampling_rules:
        if nframes is not None:
            video["nframes"] = nframes
        elif fps is not None:
            video["fps"] = fps
            if fps_min_frames is not None:
                video["min_frames"] = fps_min_frames
            if fps_max_frames is not None:
                video["max_frames"] = fps_max_frames
    return fetch_video(video)

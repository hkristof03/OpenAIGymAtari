from typing import Union

import numpy as np
import cv2


def preprocess_frame(
    screen: np.ndarray,
    exclude: Union[tuple, None] = None,
    output: Union[int, None] = None
) -> np.ndarray:
    """Preprocess Image.

    Params
    ======
        screen (array): RGB Image
        exclude (tuple): Section to be cropped (UP, RIGHT, DOWN, LEFT)
        output (int): Size of output image
    """
    # Convert image to gray scale
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)

    if exclude:
        # Crop screen[Up: Down, Left: right]
        screen = screen[exclude[0]:exclude[1], exclude[2]:exclude[3]]

    # Convert to float, and normalized
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255

    if output:
        # Resize image to 84 * 84
        screen = cv2.resize(screen, (output, output),
                            interpolation=cv2.INTER_AREA)

    return screen


def stack_frame(stacked_frames, frame, is_new):
    """Stacking Frames.

        Params
        ======
            stacked_frames (array): Four Channel Stacked Frame
            frame: Preprocessed Frame to be added
            is_new: Is the state First
        """
    if is_new:
        stacked_frames = np.stack(arrays=[frame, frame, frame, frame])
        stacked_frames = stacked_frames
    else:
        stacked_frames[0] = stacked_frames[1]
        stacked_frames[1] = stacked_frames[2]
        stacked_frames[2] = stacked_frames[3]
        stacked_frames[3] = frame

    return stacked_frames


def stack_frames(
    frames: Union[np.ndarray, None],
    state: np.ndarray,
    exclude: tuple,
    is_new: bool = False
) -> np.ndarray:

    frame = preprocess_frame(state, exclude)
    frames = stack_frame(frames, frame, is_new)

    return frames
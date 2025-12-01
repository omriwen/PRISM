# prism.visualization.animation

Training animation generator for PRISM experiments.

This module provides functionality to generate MP4 or GIF animations showing
training progression over time, including side-by-side comparisons and metric overlays.

Examples
--------
>>> # Generate animation from checkpoint directory
>>> animator = TrainingAnimator("runs/experiment")
>>> animator.generate_video("training.mp4", fps=10)

>>> # Create side-by-side comparison
>>> animator = TrainingAnimator.from_multiple(["runs/exp1", "runs/exp2"])
>>> animator.generate_video("comparison.mp4", layout="side_by_side")

## Classes

### MultiExperimentAnimator

```python
MultiExperimentAnimator(animators: List[prism.visualization.animation.TrainingAnimator])
```

Generate side-by-side comparison animations for multiple experiments.

Parameters
----------
animators : List[TrainingAnimator]
    List of individual experiment animators

#### Methods

##### `__init__`

Initialize multi-experiment animator.

Parameters
----------
animators : List[TrainingAnimator]
    List of training animators to compare

##### `create_frame`

Create comparison frame.

Parameters
----------
frame_idx : int
    Frame number
reconstructions : List[np.ndarray]
    Reconstructions for each experiment
metrics_list : List[Dict[str, float]]
    Metrics for each experiment
layout : str, optional
    Layout style: "grid" or "horizontal" (default: "grid")

Returns
-------
np.ndarray
    Frame as RGB array

##### `generate_gif`

Generate comparison GIF.

Parameters
----------
output_path : Union[str, Path]
    Output GIF file path
duration : int, optional
    Duration per frame in milliseconds (default: 100)
n_frames : int, optional
    Number of frames to generate. If None, auto-determined.
layout : str, optional
    Layout style: "grid" or "horizontal" (default: "grid")
loop : int, optional
    Number of loops (0 = infinite, default: 0)

##### `generate_video`

Generate comparison video.

Parameters
----------
output_path : Union[str, Path]
    Output video file path
fps : int, optional
    Frames per second (default: 10)
n_frames : int, optional
    Number of frames to generate. If None, auto-determined.
layout : str, optional
    Layout style: "grid" or "horizontal" (default: "grid")

Raises
------
ImportError
    If opencv-python is not installed

### TrainingAnimator

```python
TrainingAnimator(experiment_path: Union[str, pathlib.Path], checkpoint_file: str = 'checkpoint.pt')
```

Generate training progression animations.

This class loads checkpoint data and generates MP4 or GIF animations
showing how reconstructions improve over training iterations.

Parameters
----------
experiment_path : Path
    Path to experiment directory containing checkpoint
checkpoint_file : str, optional
    Name of checkpoint file (default: "checkpoint.pt")

Attributes
----------
exp_path : Path
    Experiment directory path
checkpoint : dict
    Loaded checkpoint data
ground_truth : np.ndarray or None
    Ground truth image if available

#### Methods

##### `__init__`

Initialize training animator.

Parameters
----------
experiment_path : Union[str, Path]
    Path to experiment directory
checkpoint_file : str, optional
    Name of checkpoint file (default: "checkpoint.pt")

Raises
------
FileNotFoundError
    If experiment path or checkpoint file doesn't exist
ValueError
    If checkpoint doesn't contain required data

##### `create_frame`

Create a single animation frame.

Parameters
----------
frame_idx : int
    Frame number
reconstruction : np.ndarray
    Reconstruction at this frame
metrics : Dict[str, float]
    Metrics to display
show_metrics : bool, optional
    Whether to show metric overlays (default: True)
show_difference : bool, optional
    Whether to show difference map (default: True)

Returns
-------
np.ndarray
    Frame as RGB array

##### `generate_gif`

Generate animated GIF.

Parameters
----------
output_path : Union[str, Path]
    Output GIF file path
duration : int, optional
    Duration per frame in milliseconds (default: 100)
n_frames : int, optional
    Number of frames to generate. If None, auto-determined.
show_metrics : bool, optional
    Whether to show metric overlays (default: True)
show_difference : bool, optional
    Whether to show difference map (default: True)
loop : int, optional
    Number of loops (0 = infinite, default: 0)

##### `generate_video`

Generate MP4 video animation.

Parameters
----------
output_path : Union[str, Path]
    Output video file path
fps : int, optional
    Frames per second (default: 10)
n_frames : int, optional
    Number of frames to generate. If None, auto-determined.
show_metrics : bool, optional
    Whether to show metric overlays (default: True)
show_difference : bool, optional
    Whether to show difference map (default: True)

Raises
------
ImportError
    If opencv-python is not installed

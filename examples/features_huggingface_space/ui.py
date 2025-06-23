import json
import copy
import time
import os
import tempfile
import subprocess
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import gradio as gr
from PIL import Image
import numpy as np

from lerobot.common.datasets.factory import make_dataset_without_config

# Import your LeRobot classes (adjust import path as needed)
# from lerobot.common.datasets.lerobot_dataset import MultiLeRobotDataset

class FeatureVisualizationUI:
    def __init__(self, results_dir: str, datasets: Any):
        """
        Initialize the UI with the results directory and MultiLeRobotDataset
        
        Args:
            results_dir: Path to feature_analysis_results directory
            datasets: MultiLeRobotDataset instance
        """
        self.results_dir = Path(results_dir)
        self.descriptions_file = self.results_dir / "feature_descriptions.json"
        self.datasets = datasets
        self.descriptions = self._load_descriptions()
        self.available_features = self._scan_available_features()
        
        # Create temp directory for video clips
        self.temp_dir = Path(tempfile.mkdtemp(prefix="feature_viz_videos_"))
        print(f"Video clips will be stored in: {self.temp_dir}")
        
    def _scan_available_features(self) -> List[Tuple[str, str]]:
        """Scan the results directory for available features with data and group them by named/unnamed"""
        features = []
        named_features = []
        unnamed_features = []
        
        # Look for batch directories
        for batch_dir in self.results_dir.glob("batch_*"):
            examples_dir = batch_dir / "examples"
            if examples_dir.exists():
                # Look for feature directories
                for feature_dir in examples_dir.glob("feature_*"):
                    json_file = feature_dir / "episode_top_examples.json"
                    if json_file.exists():
                        # Extract feature number from directory name
                        feature_num = int(feature_dir.name.split("_")[-1])
                        feature_id = f"{batch_dir.name}/feature_{feature_num}"
                        
                        # Check if feature has a name in descriptions
                        feature_info = self.descriptions.get(feature_id, {})
                        feature_name = feature_info.get("name", "").strip()
                        
                        if feature_name:
                            named_features.append((feature_id, feature_name, feature_num))
                        else:
                            unnamed_features.append((feature_id, feature_id, feature_num))
        
        # Sort both lists by feature number
        named_features.sort(key=lambda x: x[2])
        unnamed_features.sort(key=lambda x: x[2])
        
        # Combine with separator, removing the feature_num from the tuples
        if named_features and unnamed_features:
            features = [(f[0], f[1]) for f in named_features] + [("---", "---")] + [(f[0], f[1]) for f in unnamed_features]
        else:
            features = [(f[0], f[1]) for f in named_features] + [(f[0], f[1]) for f in unnamed_features]
            
        return features

    def _format_feature_choices(self, features: List[Tuple[str, str]]) -> List[str]:
        """Format features for dropdown display"""
        choices = []
        for feature_id, display_name in features:
            if feature_id == "---":
                choices.append("--- Unnamed Features ---")
            else:
                choices.append(f"{display_name} ({feature_id})")
        return choices

    def _load_descriptions(self) -> Dict[str, Dict[str, str]]:
        """Load existing feature descriptions from JSON file"""
        if self.descriptions_file.exists():
            try:
                with open(self.descriptions_file, 'r') as f:
                    data = json.load(f)
                    # Handle backward compatibility - convert old format to new format
                    if data and isinstance(list(data.values())[0], str):
                        # Old format: feature_id -> description_string
                        # Convert to new format: feature_id -> {"name": "", "description": description_string}
                        converted_data = {}
                        for feature_id, description in data.items():
                            converted_data[feature_id] = {
                                "name": "",
                                "description": description
                            }
                        return converted_data
                    return data
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load descriptions file: {e}")
                return {}
        return {}
    
    def _save_descriptions(self):
        """Save feature descriptions to JSON file"""
        try:
            # Ensure the directory exists
            self.descriptions_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.descriptions_file, 'w') as f:
                json.dump(self.descriptions, f, indent=2)
        except IOError as e:
            print(f"Error saving descriptions: {e}")
            raise
    
    def _load_feature_data(self, feature_id: str) -> Dict:
        """Load the episode top examples data for a given feature"""
        batch_name, feature_name = feature_id.split("/")
        if int(feature_name.split("_")[-1]) < 10:
            feature_name = f"feature_00{feature_name.split('_')[-1]}"
        elif int(feature_name.split("_")[-1]) < 100:
            feature_name = f"feature_0{feature_name.split('_')[-1]}"
        json_path = self.results_dir / batch_name / "examples" / feature_name / "episode_top_examples.json"
        
        if json_path.exists():
            with open(json_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _get_top_activations(self, feature_data: Dict, top_k: int = 20) -> List[Dict]:
        """Extract top K activations across all datasets and episodes"""
        all_activations = []
        
        for dataset_id, episodes in feature_data.items():
            for episode_id, frames in episodes.items():
                activation_info = {
                    'dataset_id': int(dataset_id),
                    'episode_id': int(episode_id),
                    'frame_idx': frames[0]['frame_idx'],
                    'activation_value': frames[0]['activation_value']
                }
                all_activations.append(activation_info)
        
        # Sort by activation value (descending) and take top K
        all_activations.sort(key=lambda x: x['activation_value'], reverse=True)
        return all_activations[:top_k]
    
    def _get_episode_info(self, dataset_id: int, episode_id: int) -> Dict:
        """Get episode information including total frames and FPS"""
        try:
            dataset = self.datasets._datasets[dataset_id]
            
            # Get FPS from dataset metadata (default to 30 if not available)
            fps = getattr(dataset.meta, 'fps', 20)
            
            return {
                "total_frames": dataset.meta.episodes[episode_id]['length'],
                "fps": fps
            }
            
        except Exception as e:
            print(f"Error getting episode info: {e}")
            return {"total_frames": 100, "fps": 30}  # Fallback values
    
    def _generate_video_clip(self, dataset_id: int, episode_id: int, start_frame: int, 
                           camera_key: str, clip_frames: int = 100) -> Optional[str]:
        """
        Generate a video clip around the center frame using ffmpeg timestamp clipping
        
        Args:
            dataset_id: Dataset index
            episode_id: Episode index
            start_frame: Frame to center the clip around
            camera_key: Camera key to extract video for
            clip_frames: Number of frames to include in clip
            
        Returns:
            Path to generated video file or None if failed
        """
        try:
            # Get episode info
            episode_info = self._get_episode_info(dataset_id, episode_id)
            total_frames = episode_info["total_frames"]
            fps = episode_info["fps"]
            
            # Calculate actual clip length (limit to available frames)
            actual_clip_frames = min(clip_frames, total_frames)
            
            # Calculate end frame
            end_frame = min(total_frames - 1, start_frame + actual_clip_frames - 1)
                        
            # Convert frames to timestamps
            start_time = start_frame / fps
            duration = (end_frame - start_frame + 1) / fps
            
            # Get source video path
            dataset = self.datasets._datasets[dataset_id]
            video_path = dataset.meta.root / dataset.meta.get_video_file_path(episode_id, camera_key)
            
            if not os.path.exists(video_path):
                print(f"Source video not found: {video_path}")
                return None
            
            # Generate consistent filename for caching
            clip_id = f"d{dataset_id}_e{episode_id}_f{start_frame}_{camera_key}_c{actual_clip_frames}"
            clip_hash = hashlib.md5(clip_id.encode()).hexdigest()[:12]
            output_filename = f"clip_{clip_hash}.mp4"
            output_path = self.temp_dir / output_filename
            
            # Check if clip already exists
            if output_path.exists() and output_path.stat().st_size > 0:
                print(f"Using cached video clip: {output_path}")
                return str(output_path)
            
            print(f"Generating video clip: {clip_id}")
            print(f"  Source: {video_path}")
            print(f"  Start time: {start_time:.3f}s, Duration: {duration:.3f}s")
            print(f"  Frames: {start_frame}-{end_frame} ({actual_clip_frames} frames)")
            
            # Check if libx264 encoder is available (more compatible than libsvtav1)
            encoder_check = subprocess.run(
                ['ffmpeg', '-encoders'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if encoder_check.returncode != 0:
                print("Error: ffmpeg not available")
                return None
            
            video_codec = 'libx264' if 'libx264' in encoder_check.stdout else 'copy'
            
            # Build ffmpeg command
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-ss', str(start_time),
                '-t', str(duration),
                '-c:v', video_codec,
                '-c:a', 'copy' if video_codec != 'copy' else 'copy',
                '-avoid_negative_ts', 'make_zero',
                '-y',  # Overwrite output file
                str(output_path)
            ]
            
            # Run ffmpeg command
            result = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True
            )
            
            if result.returncode != 0:
                print(f"FFmpeg error: {result.stderr}")
                return None
            
            # Verify output file exists and has content
            if output_path.exists() and output_path.stat().st_size > 0:
                print(f"Successfully created video clip: {output_path}")
                return str(output_path)
            else:
                print("Error: Output video file was not created or is empty")
                return None
                
        except Exception as e:
            print(f"Error generating video clip: {e}")
            return None
    
    def _create_error_image(self, message: str) -> Image.Image:
        """Create a simple error message image"""
        img = Image.new('RGB', (400, 200), color='lightgray')
        return img
    
    def update_activations(self, feature_choice: str):
        """Update the activations list when a feature is selected"""
        if not feature_choice or feature_choice == "--- Unnamed Features ---":
            return gr.update(choices=[], value=None)
        
        # Extract feature_id from the choice string (format: "name (feature_id)")
        feature_id = feature_choice.split("(")[-1].rstrip(")")
        
        feature_data = self._load_feature_data(feature_id)
        top_activations = self._get_top_activations(feature_data)
        
        # Format choices for the dropdown
        choices = []
        for i, act in enumerate(top_activations):
            label = (f"#{i+1}: Dataset {act['dataset_id']}, Episode {act['episode_id']}, "
                    f"Frame {act['frame_idx']} (activation: {act['activation_value']:.4f})")
            choices.append(label)
        
        return gr.update(choices=choices, value=choices[0] if choices else None)
    
    def update_videos(self, feature_choice: str, activation_choice: str):
        """Update videos when an activation is selected"""
        if not feature_choice or not activation_choice or feature_choice == "--- Unnamed Features ---":
            return []
        
        try:
            # Extract feature_id from the choice string
            feature_id = feature_choice.split("(")[-1].rstrip(")")
            
            # Parse the activation choice to extract info
            parts = activation_choice.split(", ")
            dataset_id = int(parts[0].split()[-1])
            episode_id = int(parts[1].split()[-1])
            frame_idx = int(parts[2].split()[1])
            
            print(f"Generating videos for Dataset {dataset_id}, Episode {episode_id}, Frame {frame_idx}")
            
            # Get available camera keys
            dataset = self.datasets._datasets[dataset_id]
            camera_keys = dataset.meta.video_keys if hasattr(dataset.meta, 'video_keys') else []
            
            if not camera_keys:
                print("No video keys found in dataset")
                return []
            
            # Generate video clips for each camera
            video_files = []
            for camera_key in camera_keys:
                print(f"Processing camera: {camera_key}")
                video_path = self._generate_video_clip(
                    dataset_id, episode_id, frame_idx, camera_key
                )
                
                if video_path:
                    video_files.append(video_path)
                else:
                    print(f"Failed to generate video for camera {camera_key}")
            
            print(f"Generated {len(video_files)} video clips")
            return video_files
            
        except Exception as e:
            print(f"Error updating videos: {e}")
            return []
    
    def load_description(self, feature_choice: str):
        """Load the name and description for the selected feature"""
        if not feature_choice or feature_choice == "--- Unnamed Features ---":
            return "", ""
        
        # Extract feature_id from the choice string
        feature_id = feature_choice.split("(")[-1].rstrip(")")
        
        feature_data = self.descriptions.get(feature_id, {"name": "", "description": ""})
        return feature_data.get("name", ""), feature_data.get("description", "")
    
    def save_description(self, feature_choice: str, feature_name: str, description: str):
        """Save the name and description for the selected feature"""
        if not feature_choice or feature_choice == "--- Unnamed Features ---":
            return "Error: No feature selected"
        
        try:
            # Extract feature_id from the choice string
            feature_id = feature_choice.split("(")[-1].rstrip(")")
            
            # Update the descriptions dictionary
            if feature_name.strip() or description.strip():
                self.descriptions[feature_id] = {
                    "name": feature_name.strip(),
                    "description": description.strip()
                }
            else:
                # Remove empty entries
                self.descriptions.pop(feature_id, None)
            
            # Save to file
            self._save_descriptions()
            
            # Update the feature dropdown to reflect the new name
            features = self._scan_available_features()
            feature_choices = self._format_feature_choices(features)
            
            return f"Feature information saved for {feature_id}"
            
        except Exception as e:
            return f"Error saving feature information: {str(e)}"
    
    def update_description_display(self, feature_choice: str):
        """Update the name and description text boxes when a feature is selected"""
        name, description = self.load_description(feature_choice)
        return gr.update(value=name), gr.update(value=description)
    
    def create_interface(self):
        """Create the Gradio interface"""
        with gr.Blocks(title="Feature Visualization") as interface:
            gr.Markdown("# Feature Visualization Tool")
            gr.Markdown("Select a feature to view its top activations and corresponding video clips. You can also add descriptions for each feature.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Get features and format them for display
                    features = self._scan_available_features()
                    feature_choices = self._format_feature_choices(features)
                    
                    feature_dropdown = gr.Dropdown(
                        choices=feature_choices,
                        label="Select Feature",
                        value=feature_choices[0] if feature_choices else None
                    )
                    
                    activation_dropdown = gr.Dropdown(
                        choices=[],
                        label="Top Activations",
                        value=None
                    )
                    
                    # Feature description section
                    gr.Markdown("### Feature Information")
                    feature_name_textbox = gr.Textbox(
                        label="Feature Name",
                        placeholder="Enter a short name for this feature...",
                        lines=1,
                        max_lines=1
                    )
                    
                    description_textbox = gr.Textbox(
                        label="Feature Description",
                        placeholder="Enter a detailed description for this feature...",
                        lines=4,
                        max_lines=10
                    )
                    
                    with gr.Row():
                        save_button = gr.Button("Save Feature Info", variant="primary") # TODO: add interactive=False as an arg to disable the save feature
                        clear_button = gr.Button("Clear", variant="secondary")
                    
                    status_message = gr.Textbox(
                        label="Status",
                        interactive=False,
                        visible=False
                    )
                
                with gr.Column(scale=2):
                    gr.Markdown("### Video Clips (100 frames, starts from activation frame)")
                    video_gallery = gr.Gallery(
                        label="Camera Videos",
                        show_label=True,
                        elem_id="video_gallery",
                        columns=2,
                        height="600",
                        object_fit="contain",
                        format="mp4"  # Specify that we're displaying videos
                    )
            
            # Set up event handlers
            feature_dropdown.change(
                fn=self.update_activations,
                inputs=[feature_dropdown],
                outputs=[activation_dropdown]
            )
            
            feature_dropdown.change(
                fn=self.update_description_display,
                inputs=[feature_dropdown],
                outputs=[feature_name_textbox, description_textbox]
            )
            
            activation_dropdown.change(
                fn=self.update_videos,
                inputs=[feature_dropdown, activation_dropdown],
                outputs=[video_gallery]
            )
            
            save_button.click(
                fn=self.save_description,
                inputs=[feature_dropdown, feature_name_textbox, description_textbox],
                outputs=[status_message]
            ).then(
                fn=lambda: gr.update(visible=True),
                outputs=[status_message]
            )
            
            clear_button.click(
                fn=lambda: ("", ""),
                outputs=[feature_name_textbox, description_textbox]
            )
            
            # Initialize with first feature if available
            if self.available_features:
                interface.load(
                    fn=self.update_activations,
                    inputs=[feature_dropdown],
                    outputs=[activation_dropdown]
                )
                
                interface.load(
                    fn=self.update_description_display,
                    inputs=[feature_dropdown],
                    outputs=[feature_name_textbox, description_textbox]
                )
        
        return interface


def find_results_dir() -> Path:
    """Find the feature analysis results directory, checking both possible locations.
    
    Returns:
        Path: Path to the existing results directory
        
    Raises:
        FileNotFoundError: If neither location contains the results directory
    """
    # Check both possible locations
    local_path = Path("examples/features_huggingface_space/feature_analysis_results")
    gradio_path = Path("feature_analysis_results")
    
    if local_path.exists():
        return local_path
    elif gradio_path.exists():
        return gradio_path
    else:
        raise FileNotFoundError(
            f"Could not find feature_analysis_results directory in either:\n"
            f"  - {local_path.absolute()}\n"
            f"  - {gradio_path.absolute()}"
        )

def launch_ui(datasets, share: bool = False, port: int = 7860):
    """
    Launch the feature visualization UI
    
    Args:
        results_dir: Path to feature_analysis_results directory
        datasets: MultiLeRobotDataset instance
        share: Whether to create a public link
        port: Port to run the server on
    """
    # Find the correct results directory
    results_path = find_results_dir()
    print(f"Using results directory: {results_path}")
    
    ui = FeatureVisualizationUI(results_path, datasets)
    interface = ui.create_interface()
    
    interface.launch(
        share=share, 
        server_port=port,
        server_name="0.0.0.0"
    )


# Example usage:
if __name__ == "__main__":
    # Example of how to use this UI
    # You would replace these with your actual paths and dataset
    
    dataset = make_dataset_without_config(
        repo_id='[villekuosmanen/pick_coffee_prop_20Hz,villekuosmanen/pick_coffee_prop_20Hz_alt]',
        action_delta_indices=list(range(100)),
        observation_delta_indices=None,
    )
    
    launch_ui(dataset, share=True)
    
    print("To use this UI:")
    print("1. Import this module")
    print("2. Create your MultiLeRobotDataset instance") 
    print("3. Call launch_ui(results_dir, datasets)")
    print()
    print("Example:")
    print("from feature_viz_ui import launch_ui")
    print("launch_ui('python/mech_interp/feature_analysis_results', your_dataset)")

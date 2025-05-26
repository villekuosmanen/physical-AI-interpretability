import subprocess

def encode_video_ffmpeg(frames, output_filename, fps, pix_fmt_in="bgr24"):
    """
    Encodes a list of numpy frames into a video using ffmpeg.

    Args:
        frames (List[np.ndarray]): List of frames (H, W, C) as numpy arrays (BGR format).
        output_filename (str): Path to save the output video.
        fps (int): Frames per second for the output video.
        pix_fmt_in (str): Input pixel format for ffmpeg (e.g., "bgr24" for OpenCV images).
    """
    if not frames:
        print(f"No frames to encode for {output_filename}.")
        return

    height, width, channels = frames[0].shape
    if channels != 3:
        print(f"Error: Frames must be 3-channel (BGR). Got {channels} channels.")
        return

    command = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',  # Frame size
        '-pix_fmt', pix_fmt_in,     # Input pixel format
        '-r', str(fps),             # Frames per second
        '-i', '-',                  # Input comes from stdin
        '-an',                      # No audio
        '-vcodec', 'libx264',       # Output video codec
        '-pix_fmt', 'yuv420p',      # Output pixel format for broad compatibility
        '-crf', '23',               # Constant Rate Factor (quality, 18-28 is good range)
        output_filename
    ]

    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for frame in frames:
            process.stdin.write(frame.tobytes())
        
        stdout, stderr = process.communicate() # Close stdin, wait for process to finish
        
        if process.returncode != 0:
            print(f"Error encoding video {output_filename}:")
            print(f"FFmpeg stdout:\n{stdout.decode(errors='ignore')}")
            print(f"FFmpeg stderr:\n{stderr.decode(errors='ignore')}")
        else:
            print(f"Successfully encoded video: {output_filename}")
            
    except FileNotFoundError:
        print("Error: ffmpeg command not found. Please ensure ffmpeg is installed and in your PATH.")
    except Exception as e:
        print(f"An unexpected error occurred during video encoding for {output_filename}: {e}")

"""
=============================================================================
create_video.py - 이미지 시퀀스를 영상으로 변환
=============================================================================

Memory-safe 배치 처리로 대량의 이미지도 메모리 부족 없이 영상 생성.

사용법:
  # 기본 (30fps, CRF 18)
  python create_video.py -i output -o timelapse.mp4
  
  # 해상도 변경 (1080p)
  python create_video.py -i output -o timelapse.mp4 --width 1920

옵션:
  -i, --input   : 입력 이미지 폴더 (필수)
  -o, --output  : 출력 영상 파일 (기본: output.mp4)
  --fps         : 초당 프레임 수 (기본: 30)
  --crf         : 품질 0-51, 낮을수록 고품질 (기본: 18)
  --width       : 영상 가로 해상도 (세로는 비율 유지)
  --batch       : 배치당 이미지 수 (기본: 200)
  --ext         : 이미지 확장자 (기본: jpg)
=============================================================================
"""
import os
import subprocess
import argparse
from glob import glob
import tempfile
import shutil

def get_images(input_dir, ext='jpg'):
    """Get sorted list of images (recursive)"""
    patterns = [f"**/*.{ext}", f"**/*.{ext.upper()}"]
    images = []
    for pattern in patterns:
        images.extend(glob(os.path.join(input_dir, pattern), recursive=True))
    return sorted(images)

def create_chunk_video(image_list, output_file, fps=30, crf=18, width=None):
    """Create a video from a list of images using concat demuxer"""
    # Create temporary list file
    list_file = output_file + ".txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for img in image_list:
            # Use forward slashes and absolute path
            abs_path = os.path.abspath(img).replace("\\", "/")
            f.write(f"file '{abs_path}'\n")
            f.write(f"duration {1/fps}\n")
    
    # ffmpeg command with lower memory usage settings
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", str(crf),
        "-preset", "medium",
        "-r", str(fps),
        "-movflags", "+faststart"
    ]
    
    if width:
        cmd.extend(["-vf", f"scale={width}:-2"])
        
    cmd.append(output_file)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        os.remove(list_file)
    except: pass
    
    if result.returncode != 0:
        print(f"Error creating {output_file}: {result.stderr}")
        return False
    return True

def concat_videos(video_list, output_file):
    """Concatenate multiple videos into one"""
    list_file = output_file + "_concat.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for vid in video_list:
            abs_path = os.path.abspath(vid).replace("\\", "/")
            f.write(f"file '{abs_path}'\n")
    
    # Use stream copy for fast concatenation (no re-encoding)
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-c", "copy",  # No re-encoding, just copy streams
        output_file
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        os.remove(list_file)
    except: pass
    
    if result.returncode != 0:
        print(f"Error concatenating videos: {result.stderr}")
        return False
    return True

def create_video_chunked(input_dir, output_file, fps=30, crf=18, batch_size=200, ext='jpg', width=None, image_list=None):
    """Create video by processing images in chunks to avoid memory issues"""
    if image_list:
        images = image_list
    else:
        images = get_images(input_dir, ext)
        
    if not images:
        print("No images found!")
        return
    
    total = len(images)
    print(f"Found {total} images. Processing in batches of {batch_size}...")
    
    # Create temp directory for chunk videos
    temp_dir = tempfile.mkdtemp(prefix="ffmpeg_chunks_")
    chunk_videos = []
    
    try:
        # Process in chunks
        for i in range(0, total, batch_size):
            chunk_images = images[i:i+batch_size]
            chunk_num = i // batch_size + 1
            total_chunks = (total + batch_size - 1) // batch_size
            
            chunk_file = os.path.join(temp_dir, f"chunk_{chunk_num:04d}.mp4")
            print(f"Processing chunk {chunk_num}/{total_chunks} ({len(chunk_images)} images)...")
            
            if create_chunk_video(chunk_images, chunk_file, fps, crf, width):
                chunk_videos.append(chunk_file)
            else:
                print(f"Failed to create chunk {chunk_num}")
                return
        
        # Concatenate all chunks
        print(f"\nConcatenating {len(chunk_videos)} chunks...")
        if concat_videos(chunk_videos, output_file):
            print(f"\n✓ Video created: {output_file}")
            # Get file size
            size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"  Size: {size_mb:.1f} MB")
            print(f"  Duration: {total/fps:.1f} seconds @ {fps} fps")
        else:
            print("Failed to concatenate videos")
    
    finally:
        # Cleanup temp directory
        print("Cleaning up temporary files...")
        shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    parser = argparse.ArgumentParser(description="Create video from images with chunked processing")
    parser.add_argument("--input", "-i", required=True, help="Input folder with images")
    parser.add_argument("--output", "-o", default="output.mp4", help="Output video file")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    parser.add_argument("--crf", type=int, default=18, help="Quality (0-51, lower=better, default: 18)")
    parser.add_argument("--batch", type=int, default=200, help="Images per batch (default: 200)")
    parser.add_argument("--ext", default="jpg", help="Image extension (default: jpg)")
    parser.add_argument("--width", type=int, help="Target video width (e.g. 1920)")
    
    args = parser.parse_args()
    
    create_video_chunked(
        input_dir=args.input,
        output_file=args.output,
        fps=args.fps,
        crf=args.crf,
        batch_size=args.batch,
        ext=args.ext,
        width=args.width
    )

if __name__ == "__main__":
    main()

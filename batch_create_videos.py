import os
import argparse
from datetime import datetime
from create_video import create_video_chunked

def get_date_folders(path):
    """Get sorted list of YYYY-MM-DD folders in path"""
    folders = []
    if os.path.exists(path):
        for d in os.listdir(path):
            if os.path.isdir(os.path.join(path, d)):
                try:
                    datetime.strptime(d, "%Y-%m-%d")
                    folders.append(d)
                except ValueError:
                    continue
    return sorted(folders)

def main():
    parser = argparse.ArgumentParser(description="Batch create videos for all sites in output directory")
    parser.add_argument("--output_root", default="output", help="Root output directory containing site folders")
    parser.add_argument("--width", type=int, help="Target video width (e.g. 1920)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--crf", type=int, default=18, help="CRF quality (lower is better)")
    parser.add_argument("--batch", type=int, default=200, help="Batch size for processing")
    
    args = parser.parse_args()
    
    base_output_dir = os.path.abspath(args.output_root)
    
    if not os.path.exists(base_output_dir):
        print(f"Error: Output directory not found at {base_output_dir}")
        return

    # Get all subdirectories in output (assuming these are the sites)
    sites = [d for d in os.listdir(base_output_dir) if os.path.isdir(os.path.join(base_output_dir, d))]
    
    if not sites:
        print("No site directories found in output folder.")
        return

    print(f"Found {len(sites)} sites: {', '.join(sites)}")

    for site in sites:
        site_path = os.path.join(base_output_dir, site)
        
        # Determine start and end dates from folder structure
        date_folders = get_date_folders(site_path)
        if not date_folders:
            print(f"Skipping {site}: No valid date folders found.")
            continue
            
        start_d = date_folders[0]
        end_d = date_folders[-1]
        
        # Naming convention: {site}_{start}~{end}_{res}_{time}.mp4
        res_str = f"{args.width}p" if args.width else "Original"
        now_str = datetime.now().strftime("%H%M%S")
        video_filename = f"{site}_{start_d}~{end_d}_{res_str}_{now_str}.mp4"
        
        # Output path is inside the project folder
        output_video_path = os.path.join(site_path, video_filename)
        
        print(f"\nProcessing site: {site}")
        print(f"  Range: {start_d} ~ {end_d}")
        print(f"  Output: {output_video_path}")
        
        # Call the video creation function
        create_video_chunked(
            input_dir=site_path,
            output_file=output_video_path,
            fps=args.fps,
            crf=args.crf,
            batch_size=args.batch,
            ext="jpg",
            width=args.width
        )
        print(f"Finished processing {site}")

if __name__ == "__main__":
    main()

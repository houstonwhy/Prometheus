import os
from pathlib import Path
import requests
from bs4 import BeautifulSoup

# Create a directory to save the downloaded videos
workdir = '/data1/yyb/PlatonicGen/workdir/dir3d_gallery'
os.makedirs(workdir, exist_ok=True)


for i in range(27):
    # Target webpage URL
    url = f"https://imlixinyang.github.io/director3d-page/gallery_{i}.html"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all <video> tags and their <source> tags
    videos = soup.find_all('video')

    # Extract and download video links
    for video in videos:
        source = video.find('source')
        if source and 'data-src' in source.attrs:
            video_url = source['data-src'].strip()

            # Construct the full URL
            full_video_url = f"https://imlixinyang.github.io/director3d-page/{video_url}"

            # Get the video file name
            video_filename = os.path.join(workdir, 'videos', os.path.basename(video_url))

            # Download the video
            print(f"Downloading {full_video_url} to {video_filename}")
            video_response = requests.get(full_video_url)
            with open(video_filename, 'wb') as f:
                f.write(video_response.content)

    print("All videos have been downloaded.")
import urllib.request
import os

os.makedirs('temp_medsam2', exist_ok=True)
files = ['logo.png', '1.png', '4.png', '6.png', '0.mp4', '2.mp4', '3.mp4', '5.mp4', '7.mp4', '8.mp4', '9.mp4']

for fn in files:
    url = f'https://medsam2.github.io/static/{fn}'
    dest = f'temp_medsam2/{fn}'
    try:
        urllib.request.urlretrieve(url, dest)
        size = os.path.getsize(dest)
        print(f'Downloaded {fn} ({size} bytes)')
    except Exception as e:
        print(f'Error downloading {fn}: {e}')

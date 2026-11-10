import urllib.request
import re
from urllib.parse import urljoin

url = 'https://medsam2.github.io/'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        html = resp.read().decode('utf-8', errors='ignore')
    print('Fetched medsam2.github.io successfully, length:', len(html))
    media = re.findall(r'src=["\']([^"\']+\.(?:png|jpg|jpeg|gif|mp4|webm))["\']', html, re.I)
    print('Found media count:', len(media))
    for m in media:
        full_url = urljoin(url, m)
        print('  ', full_url)
except Exception as e:
    print('Error:', e)

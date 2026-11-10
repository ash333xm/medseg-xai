from PIL import Image
import numpy as np

im = Image.open('temp_medsam2/1.png')
print('1.png dimensions:', im.size)

# 1.png is 2047 x 2411. Let's see what regions exist
# Let's save a downsampled thumbnail of 1.png to inspect
thumb = im.resize((512, int(512 * im.size[1] / im.size[0])), Image.Resampling.LANCZOS)
thumb.save('temp_medsam2/1_thumb.png')
print('1_thumb.png saved:', thumb.size)

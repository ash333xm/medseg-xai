from PIL import Image
import numpy as np

# Let's inspect 6.png
im6 = Image.open('temp_medsam2/6.png').convert('RGB')
print('6.png size:', im6.size)

# Save 4 quadrants
w, h = im6.size
half_w, half_h = w // 2, h // 2
im6.crop((0, 0, half_w, half_h)).save('temp_medsam2/6_q1.png')
im6.crop((half_w, 0, w, half_h)).save('temp_medsam2/6_q2.png')
im6.crop((0, half_h, half_w, h)).save('temp_medsam2/6_q3.png')
im6.crop((half_w, half_h, w, h)).save('temp_medsam2/6_q4.png')
print('Quadrants saved for 6.png!')

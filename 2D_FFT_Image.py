from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt

image = Image.open("images.jpg").convert("L")
A = np.asarray(image)

ny, nx = A.shape

f = np.fft.fft2(A)
f_shift = np.fft.fftshift(f)

kmax_x = nx//8
kmax_y = ny//8

# print(kmax_x,kmax_y)

mask = np.zeros_like(f_shift)

cx, cy = nx//2, ny//2

mask[cy - kmax_y : cy + kmax_y, cx - kmax_x : cx + kmax_x] = True

f_shift = np.where(mask, f_shift, 0)

f_invs_shift = np.fft.ifftshift(f_shift)
A_recons = np.fft.ifft2(f_invs_shift)

A_recons = np.abs(A_recons)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(A,cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(A_recons, cmap="gray")
plt.axis("off")

plt.show()
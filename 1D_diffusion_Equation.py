import numpy as np
import matplotlib.pyplot as plt

n = 512
l = 2*np.pi
x = np.linspace(0,l,n, endpoint=False)
dx = l/n 
c = 1.0
d = 0.1

time = [0.0, 2.0, 4.0, 10.0]

x0 = np.pi 
sigma = 0.2

u0 = np.exp(-0.5 + 0.2*np.cos(4*x))

k = np.fft.fftfreq(n,d=dx)

u0_hat = np.fft.fft(u0)

sol = []
for t in time:
    factor = np.exp((-1j * c * k - d * (k**2)) * t)
    u_hat_t = u0_hat * factor
    u_t = np.fft.ifft(u_hat_t)
    sol.append(np.real(u_t))

plt.figure()
for i,t in enumerate(time):
    plt.subplot(2,2,i+1)
    plt.plot(x,sol[i])
    if i == 0:
        plt.plot(x,u0)
    plt.xlim(0,l)
    plt.legend()
    
plt.tight_layout()
plt.show()
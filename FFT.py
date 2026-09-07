def fft(x):
    N = len(x)
    if N <= 1:
        return x
    
    ev = fft(x[0::2])
    od = fft(x[1::2])

    result = [0] * N
    for i in range (N//2):
        twiddle = np.exp(-2j * np.pi * i / N) * od[i]
        result[i] = ev[i] + twiddle
        result[i + N//2] = ev[i] - twiddle
    return result

def ifft(x):
    N = len(x)
    
    x_conj = [np.conjugate(x) for x in x]
    
    x = fft(x_conj)

    return [np.conjugate(val)/N for val in x]

N = 256
x = np.linspace(0, np.pi, N, endpoint=False)
f = np.exp(-x) * np.cos(5*x)

fk = fft(list(f))

f_recons = ifft(fk)

plt.figure()
plt.plot(x,f)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x,np.real(f_recons), "--")
plt.legend()
plt.show()
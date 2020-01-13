import numpy as np

k=4
b = np.ones((1, k-1))[0]
m =  np.diag(b, 1)
m[-1,0]=1

print(m)

evalues=np.linalg.eigvals(m)
print("eigenvalues are")
print(evalues)

th_evalues=[np.cos(2*np.pi*(k-j)/k)+1j*np.sin(2*np.pi*(k-j)/k) for j in range(0,k,1)]

x=np.sort(evalues)-np.sort(th_evalues)
x_index=np.abs(x) <1e-14
x[x_index]=0

print("Deviation from theoretical prediction is")
print(x)


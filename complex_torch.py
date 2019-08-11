import torch

class Complex:
    def __init__(self, x=None, y=None, r=None, theta=None):
        if (x is not None) and (y is not None) and (r is None) and (theta is None):
            self.real = x
            self.imag = y
            self.abs = torch.sqrt(x**2 + y**2)
            self.angle = torch.atan2(y, x)
        elif (x is None) and (y is None) and (r is not None) and (theta is not None):
            self.abs = torch.abs(r)
            self.real = r * torch.cos(theta)
            self.imag = r * torch.sin(theta)
            self.angle = torch.atan2(self.imag, self.real)
    
    def castComplex(self, w):
        if isinstance(w, int) or isinstance(w, float):
            return self.__class__(x=torch.ones_like(self.real)*w, y=torch.zeros_like(self.imag))
    
    def __add__(self, w):
        if not isinstance(w, self.__class__):
            w = self.castComplex(w)
        return self.__class__(x=self.real + w.real, y=self.imag + w.imag)

    def __radd__(self, w):
        if not isinstance(w, self.__class__):
            w = self.castComplex(w)
        return self.__class__(x=w.real + self.real, y=w.imag + self.imag)
    
    def __sub__(self, w):
        if not isinstance(w, self.__class__):
            w = self.castComplex(w)
        return self.__class__(x=self.real - w.real, y=self.imag - w.imag)

    def __rsub__(self, w):
        if not isinstance(w, self.__class__):
            w = self.castComplex(w)
        return self.__class__(x=w.real - self.real, y=w.imag - self.imag)
    
    def __mul__(self, w):
        if not isinstance(w, self.__class__):
            w = self.castComplex(w)
        return self.__class__(r=self.abs * w.abs, theta=self.angle + w.angle)

    def __rmul__(self, w):
        if not isinstance(w, self.__class__):
            w = self.castComplex(w)
        return self.__class__(r=w.abs * self.abs, theta=w.angle + self.angle)
    
    def __truediv__(self, w):
        if not isinstance(w, self.__class__):
            w = self.castComplex(w)
        return self.__class__(r=self.abs / w.abs, theta=self.angle - w.angle)

    def __rtruediv__(self, w):
        if not isinstance(w, self.__class__):
            w = self.castComplex(w)
        return self.__class__(r=w.abs / self.abs, theta=w.angle - self.angle)
    
    def __pow__(self, n):
        return self.__class__(r=torch.pow(self.abs, n), theta=self.angle*n)
    
    def mm(self, w):
        return self.__class__(x=torch.mm(self.real, w.real)-torch.mm(self.imag, w.imag), y=torch.mm(self.real, w.imag)+torch.mm(self.imag, w.real))
    
    def conjugate(self):
        return self.__class__(x=self.real, y=-self.imag)
    
    def transpose(self):
        return self.__class__(x=self.real.t(), y=self.imag.t())

    def to(self, device):
        return self.__class__(x=self.real.to(device), y=self.imag.to(device))

def mm(z, w):
    return Complex(x=torch.mm(z.real, w.real)-torch.mm(z.imag, w.imag), y=torch.mm(z.real, w.imag)+torch.mm(z.imag, w.real))

def printComp(z, mode="xy", n_row=10, n_col=30, unit="i"):
    if mode is "xy":
        for i in range(min(z.real.size(0), n_row)):
            for j in range(min(z.real.size(1), n_col)):
                print("{:.3f}{:+.3f}{}".format(z.real[i][j], z.imag[i][j], unit), end=" ")
            print()
    elif mode is "phasor":
        for i in range(min(z.real.size(0), n_row)):
            for j in range(min(z.real.size(1), n_col)):
                print("{:.3f}∠{:.3f}".format(z.abs[i][j], z.angle[i][j]), end=" ")
            print()
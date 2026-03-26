import math

def distancia(p1, p2):
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )

def dedo_estendido(base, ponta):
    return ponta.y < base.y - 0.02

def dedo_dobrado(base, ponta):
    return ponta.y > base.y + 0.02
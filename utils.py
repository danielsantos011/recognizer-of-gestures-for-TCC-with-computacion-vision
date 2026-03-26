import math

def distancia(p1, p2):
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )

def dedo_dobrado(base, ponta):
    """
    Retorna True se o dedo estiver dobrado
    """
    return ponta.y > base.y

def dedo_estendido(base, ponta):
    """
    Retorna True se o dedo estiver estendido
    """
    return ponta.y < base.y
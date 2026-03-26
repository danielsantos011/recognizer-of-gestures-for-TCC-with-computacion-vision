from utils import dedo_estendido, dedo_dobrado, distancia

IND_BASE, IND_TIP = 5, 8
MED_BASE, MED_TIP = 9, 12
ANE_BASE, ANE_TIP = 13, 16
MIN_BASE, MIN_TIP = 17, 20
POL_BASE, POL_TIP = 2, 4


def letra_a(lm):
    dedos = (
        dedo_dobrado(lm[IND_BASE], lm[IND_TIP]) and
        dedo_dobrado(lm[MED_BASE], lm[MED_TIP]) and
        dedo_dobrado(lm[ANE_BASE], lm[ANE_TIP]) and
        dedo_dobrado(lm[MIN_BASE], lm[MIN_TIP])
    )
    polegar_fora = distancia(lm[POL_TIP], lm[IND_BASE]) > 0.06
    return dedos and polegar_fora


def letra_b(lm):
    dedos = (
        dedo_estendido(lm[IND_BASE], lm[IND_TIP]) and
        dedo_estendido(lm[MED_BASE], lm[MED_TIP]) and
        dedo_estendido(lm[ANE_BASE], lm[ANE_TIP]) and
        dedo_estendido(lm[MIN_BASE], lm[MIN_TIP])
    )
    polegar_dentro = distancia(lm[POL_TIP], lm[IND_BASE]) < 0.04
    return dedos and polegar_dentro


def letra_c(lm):
    d_ind = distancia(lm[IND_BASE], lm[IND_TIP])
    d_med = distancia(lm[MED_BASE], lm[MED_TIP])
    curvo = 0.04 < d_ind < 0.08 and 0.04 < d_med < 0.08
    polegar_oposto = distancia(lm[POL_TIP], lm[IND_TIP]) < 0.06
    return curvo and polegar_oposto
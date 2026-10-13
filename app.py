"""Laboratorio virtual: Muestreo e Inferencia Estadística aplicada a experimentos de IA.

Ejecutar:  streamlit run app.py
"""
import io
import math
from datetime import date

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Muestreo e Inferencia · Laboratorio virtual", layout="wide")

# Columnas de la bitácora (mismo orden que el archivo exportado de la guía)
COLS = [
    "N_registro", "Fecha", "Módulo", "Condición", "param_N", "param_S", "param_E", "param_confianza",
    "res_Z", "res_n_calculado", "res_n_entero", "Decisión/Interpretación", "param_N_i", "param_S_i",
    "param_n_total", "res_N_i·S_i", "res_peso_Neyman", "res_n_i_Neyman", "res_n_i_proporcional",
    "param_mu", "param_sigma", "param_R", "param_semilla", "param_n", "res_media_de_medias",
    "res_DE_empirica", "res_EE_teorico", "res_diferencia", "param_datos", "param_mu0", "param_alfa",
    "param_alternativa", "res_media", "res_s", "res_EE", "res_t", "res_gl", "res_p", "param_A", "param_B",
    "res_media_dif", "res_s_d", "param_réplicas", "param_réplicas_datos", "param_k", "res_SS_entre",
    "res_SS_dentro", "res_MS_entre", "res_MS_dentro", "res_F", "res_eta2",
]

if "bitacora" not in st.session_state:
    st.session_state.bitacora = []


def registrar(registros):
    """Agrega uno o varios registros a la bitácora."""
    if isinstance(registros, dict):
        registros = [registros]
    for r in registros:
        r = dict(r)
        r["N_registro"] = len(st.session_state.bitacora) + 1
        r["Fecha"] = date.today().isoformat()
        st.session_state.bitacora.append(r)


def bitacora_df():
    df = pd.DataFrame(st.session_state.bitacora)
    if df.empty:
        return pd.DataFrame(columns=COLS)
    extra = [c for c in df.columns if c not in COLS]
    return df.reindex(columns=COLS + extra)


def z_de(conf):
    return stats.norm.ppf(1 - (1 - conf) / 2)


def parse_serie(texto):
    return np.array([float(v) for v in texto.replace(";", ",").replace("\n", ",").split(",") if v.strip()])


def decision(p, alfa):
    return "Rechazar H0" if p < alfa else "No rechazar H0"


# ---------------------------------------------------------------- cálculos
def n_media_finita(N, S, E, conf):
    Z = z_de(conf)
    return Z, S**2 / (E**2 / Z**2 + S**2 / N)


def n_media_infinita(S, E, conf):
    Z = z_de(conf)
    return Z, Z**2 * S**2 / E**2


def n_proporcion(N, P, E, conf):
    Z = z_de(conf)
    PQ = P * (1 - P)
    return Z, Z**2 * PQ * N / (E**2 * (N - 1) + Z**2 * PQ)


def n_replicas(s, E, conf):
    """n = t² s² / E², iterando porque t depende de n-1 grados de libertad."""
    n = max(2.0, (z_de(conf) * s / E) ** 2)
    for _ in range(100):
        t = stats.t.ppf(1 - (1 - conf) / 2, max(1, math.ceil(n) - 1))
        nuevo = t**2 * s**2 / E**2
        if abs(nuevo - n) < 1e-9:
            break
        n = nuevo
    return t, n


def neyman(tabla, n_total):
    Ni = tabla["N_i"].to_numpy(float)
    Si = tabla["S_i"].to_numpy(float)
    w = Ni * Si
    out = tabla.copy()
    out["N_i·S_i"] = w
    out["Peso Neyman"] = w / w.sum()
    out["n_i Neyman"] = n_total * w / w.sum()
    out["n_i proporcional"] = n_total * Ni / Ni.sum()
    return out


def simular(tipo, mu, sigma, p, R, semilla, n):
    rng = np.random.default_rng(int(semilla))
    if tipo == "Media":
        est = rng.normal(mu, sigma, (int(R), int(n))).mean(axis=1)
        ee = sigma / math.sqrt(n)
    else:
        est = rng.binomial(int(n), p, int(R)) / n
        ee = math.sqrt(p * (1 - p) / n)
    return est, ee


def t_una_muestra(x, mu0, alfa, alt_):
    r = stats.ttest_1samp(x, mu0, alternative=alt_)
    return dict(media=x.mean(), s=x.std(ddof=1), EE=x.std(ddof=1) / math.sqrt(len(x)),
                t=float(r.statistic), gl=len(x) - 1, p=float(r.pvalue))


def anova(grupos):
    todos = np.concatenate(grupos)
    gm = todos.mean()
    k, N = len(grupos), len(todos)
    ss_e = sum(len(g) * (g.mean() - gm) ** 2 for g in grupos)
    ss_d = sum(((g - g.mean()) ** 2).sum() for g in grupos)
    ms_e, ms_d = ss_e / (k - 1), ss_d / (N - k)
    F = ms_e / ms_d
    return dict(k=k, N=N, SS_entre=ss_e, SS_dentro=ss_d, MS_entre=ms_e, MS_dentro=ms_d, F=F,
                gl=(k - 1, N - k), p=float(stats.f.sf(F, k - 1, N - k)), eta2=ss_e / (ss_e + ss_d))


def f_varianzas(a, b):
    va, vb = a.var(ddof=1), b.var(ddof=1)
    if va >= vb:
        F, gl = va / vb, (len(a) - 1, len(b) - 1)
    else:
        F, gl = vb / va, (len(b) - 1, len(a) - 1)
    return F, gl, min(1.0, 2 * stats.f.sf(F, *gl))


# ---------------------------------------------------------------- registros
def reg_tamano(cond, tipo, N, S, E, conf, P=None):
    if tipo == "Media · población finita":
        Z, n = n_media_finita(N, S, E, conf)
    elif tipo == "Media · población infinita":
        Z, n = n_media_infinita(S, E, conf)
        N = None
    else:
        Z, n = n_proporcion(N, P, E, conf)
    r = {"Módulo": "Planificar muestreo", "Condición": cond, "param_N": N, "param_S": S, "param_E": E,
         "param_confianza": conf, "res_Z": round(Z, 4), "res_n_calculado": round(n, 2),
         "res_n_entero": math.ceil(n)}
    if P is not None:
        r["param_P"] = P
    return r


def regs_neyman(cond, tabla, n_total):
    res = neyman(tabla, n_total)
    return [{"Módulo": "Muestreo estratificado · Neyman", "Condición": f"{cond} · {f['Estrato']}",
             "param_N_i": f["N_i"], "param_S_i": f["S_i"], "param_n_total": n_total,
             "res_N_i·S_i": round(f["N_i·S_i"], 4), "res_peso_Neyman": round(f["Peso Neyman"], 4),
             "res_n_i_Neyman": round(f["n_i Neyman"], 1), "res_n_i_proporcional": round(f["n_i proporcional"], 1)}
            for _, f in res.iterrows()]


def reg_sim(cond, tipo, mu, sigma, p, R, semilla, n):
    est, ee = simular(tipo, mu, sigma, p, R, semilla, n)
    de = est.std(ddof=1)
    r = {"Módulo": "Simular muestreo", "Condición": cond, "param_R": R, "param_semilla": semilla, "param_n": n,
         "res_media_de_medias": round(est.mean(), 4), "res_DE_empirica": round(de, 4),
         "res_EE_teorico": round(ee, 4), "res_diferencia": round(de - ee, 4)}
    if tipo == "Media":
        r.update(param_mu=mu, param_sigma=sigma)
    else:
        r["param_P"] = p
    return r


def reg_t1(cond, x, mu0, alfa, alt_):
    o = t_una_muestra(x, mu0, alfa, alt_)
    return {"Módulo": "Inferencia", "Condición": cond, "Decisión/Interpretación": decision(o["p"], alfa),
            "param_datos": ",".join(f"{v:g}" for v in x), "param_mu0": mu0, "param_alfa": alfa,
            "param_alternativa": alt_, "res_media": round(o["media"], 5), "res_s": round(o["s"], 5),
            "res_EE": round(o["EE"], 5), "res_t": round(o["t"], 4), "res_gl": o["gl"], "res_p": round(o["p"], 5)}


def reg_pareada(cond, a, b, alfa, alt_="two-sided"):
    d = a - b
    r = stats.ttest_rel(a, b, alternative=alt_)
    return {"Módulo": "Inferencia", "Condición": cond, "Decisión/Interpretación": decision(r.pvalue, alfa),
            "param_alfa": alfa, "param_alternativa": alt_, "res_t": round(float(r.statistic), 4),
            "res_gl": len(d) - 1, "res_p": round(float(r.pvalue), 4),
            "param_A": ",".join(f"{v:.2f}" for v in a), "param_B": ",".join(f"{v:.2f}" for v in b),
            "res_media_dif": round(d.mean(), 4), "res_s_d": round(d.std(ddof=1), 4)}


def reg_dos_medias(cond, a, b, alfa, welch, alt_="two-sided"):
    r = stats.ttest_ind(a, b, equal_var=not welch, alternative=alt_)
    gl = float(r.df) if hasattr(r, "df") else len(a) + len(b) - 2
    return {"Módulo": "Inferencia · " + ("t Welch" if welch else "t varianza agrupada"), "Condición": cond,
            "Decisión/Interpretación": decision(r.pvalue, alfa), "param_alfa": alfa, "param_alternativa": alt_,
            "res_t": round(float(r.statistic), 4), "res_gl": round(gl, 2), "res_p": float(f"{r.pvalue:.3g}"),
            "param_A": ",".join(f"{v:g}" for v in a), "param_B": ",".join(f"{v:g}" for v in b),
            "res_media_dif": round(a.mean() - b.mean(), 4)}


def reg_f(cond, a, b, alfa):
    F, gl, p = f_varianzas(a, b)
    return {"Módulo": "Inferencia · F varianzas", "Condición": cond,
            "Decisión/Interpretación": decision(p, alfa) + (" (varianzas distintas)" if p < alfa else " (varianzas homogéneas)"),
            "param_alfa": alfa, "res_gl": f"{gl[0]}, {gl[1]}", "res_p": round(p, 4), "res_F": round(F, 3),
            "param_A": ",".join(f"{v:g}" for v in a), "param_B": ",".join(f"{v:g}" for v in b)}


def regs_anova(cond, nombres, grupos, alfa):
    o = anova(grupos)
    out = [{"Módulo": "Inferencia · ANOVA", "Condición": f"{cond} · {nom}", "res_media": round(g.mean(), 2),
            "res_s": round(g.std(ddof=1), 3), "param_réplicas": len(g),
            "param_réplicas_datos": ",".join(f"{v:g}" for v in g)} for nom, g in zip(nombres, grupos)]
    out.append({"Módulo": "Inferencia · ANOVA", "Condición": f"{cond} · ANOVA de una vía", "param_N": o["N"],
                "Decisión/Interpretación": decision(o["p"], alfa), "param_alfa": alfa,
                "res_gl": f"{o['gl'][0]}, {o['gl'][1]}", "res_p": float(f"{o['p']:.3g}"), "param_k": o["k"],
                "res_SS_entre": round(o["SS_entre"], 3), "res_SS_dentro": round(o["SS_dentro"], 3),
                "res_MS_entre": round(o["MS_entre"], 3), "res_MS_dentro": round(o["MS_dentro"], 4),
                "res_F": round(o["F"], 2), "res_eta2": round(o["eta2"], 3)})
    return out


# ---------------------------------------------------------------- datos de la guía
NEYMAN_GUIA = pd.DataFrame({"Estrato": ["Premium", "Freemium", "Invitados"],
                            "N_i": [100000, 500000, 50000], "S_i": [0.15, 0.05, 0.20]})
AZUFRE = "0.112, 0.118, 0.115, 0.119"
COL_A = "1.46, 2.22, 2.84, 1.97, 1.13, 2.35"
COL_B = "1.42, 2.38, 2.67, 1.80, 1.09, 2.25"
AUG = {
    "A. Sin aumento": "86.4, 85.8, 86.9, 85.5, 86.2, 85.7, 86.1, 85.6, 86.8, 85.9",
    "B. Aumento moderado": "88.0, 88.5, 87.9, 88.3, 88.1, 88.4, 87.8, 88.2, 88.6, 88.1",
    "C. Aumento intenso": "87.2, 86.9, 87.4, 87.3, 87.0, 87.5, 87.1, 87.3, 87.2, 87.4",
}


def cargar_experimentos_guia():
    regs = [reg_tamano("Exp1 basal · Media población finita", "Media · población finita", 40000, 8, 1.0, 0.95)]
    regs[0]["Decisión/Interpretación"] = "n=245 radiografías; diapositiva reporta ≈242 (discrepancia aritmética)"
    for E in [0.5, 0.75, 1.0, 1.5, 2.0]:
        regs.append(reg_tamano(f"Exp2 Serie A · E={E}", "Media · población finita", 40000, 8, E, 0.95))
    for c in [0.90, 0.95, 0.99]:
        regs.append(reg_tamano(f"Exp2 Serie B · confianza={c:.0%}".replace("%", " %"),
                               "Media · población finita", 40000, 8, 1.0, c))
    regs += regs_neyman("Exp3", NEYMAN_GUIA, 3000)
    for n in [10, 25, 50, 100, 500]:
        regs.append(reg_sim(f"Exp4 · Media n={n}", "Media", 10.0, 2.0, None, 1000, 1234, n))
    x = parse_serie(AZUFRE)
    regs.append(reg_t1("Exp5 · t una muestra · 95 %", x, 0.123, 0.05, "two-sided"))
    regs.append(reg_t1("Exp5 · t una muestra · 99 %", x, 0.123, 0.01, "two-sided"))
    regs.append(reg_pareada("Exp6 · t pareada", parse_serie(COL_A), parse_serie(COL_B), 0.05))
    g = {k: parse_serie(v) for k, v in AUG.items()}
    regs += regs_anova("Libre", list(g), list(g.values()), 0.05)
    regs.append(reg_f("Libre · F A vs C", g["A. Sin aumento"], g["C. Aumento intenso"], 0.05))
    regs.append(reg_dos_medias("Libre · Welch B vs C", g["B. Aumento moderado"], g["C. Aumento intenso"], 0.05, True))
    registrar(regs)


# ---------------------------------------------------------------- interfaz
st.sidebar.title("Laboratorio virtual")
st.sidebar.caption("Muestreo e Inferencia Estadística aplicada a experimentos de IA")
modulo = st.sidebar.radio("Módulo de trabajo", [
    "Planificar muestreo", "Muestreo estratificado · Neyman", "Simular muestreo", "Inferencia", "Bitácora y descarga"])
st.sidebar.divider()
st.sidebar.metric("Mediciones registradas", len(st.session_state.bitacora))
st.sidebar.caption("Secuencia: predecir → modificar variables → simular → registrar → analizar → interpretar")

st.title(modulo)


def caja_prediccion(clave):
    return st.text_area("Predicción (antes de modificar los parámetros)", key=f"pred_{clave}", height=70)


def boton_registrar(registros, clave, prediccion=""):
    if st.button("Registrar medición", type="primary", key=f"btn_{clave}"):
        regs = registros if isinstance(registros, list) else [registros]
        if prediccion:
            for r in regs:
                r["Predicción"] = prediccion
        registrar(regs)
        st.success(f"Registrado. Total en bitácora: {len(st.session_state.bitacora)}")


# --- Planificar muestreo
if modulo == "Planificar muestreo":
    tipo = st.selectbox("Tipo de cálculo", ["Media · población finita", "Media · población infinita",
                                             "Proporción · población finita", "Número de réplicas"])
    pred = caja_prediccion("plan")
    c1, c2, c3, c4 = st.columns(4)
    conf = c4.select_slider("Confianza", [0.90, 0.95, 0.99], value=0.95, format_func=lambda v: f"{v:.0%}")
    if tipo == "Número de réplicas":
        s = c1.number_input("s de muestreo (estudio piloto)", value=1.0, min_value=0.0001, format="%.4f")
        E = c2.number_input("Error máximo E", value=0.5, min_value=0.0001, format="%.4f")
        t, n = n_replicas(s, E, conf)
        m1, m2, m3 = st.columns(3)
        m1.metric("t crítico", f"{t:.4f}")
        m2.metric("n calculado", f"{n:.2f}")
        m3.metric("n entero (réplicas)", math.ceil(n))
        reg = {"Módulo": "Planificar muestreo", "Condición": f"Réplicas · s={s}, E={E}, {conf:.0%}",
               "param_S": s, "param_E": E, "param_confianza": conf, "res_Z": round(t, 4),
               "res_n_calculado": round(n, 2), "res_n_entero": math.ceil(n)}
    else:
        P = None
        N = None
        if tipo != "Media · población infinita":
            N = c1.number_input("Tamaño de población N", value=40000 if tipo.startswith("Media") else 120000, min_value=1)
        if tipo.startswith("Media"):
            S = c2.number_input("Desviación estándar S", value=8.0, min_value=0.0001)
            E = c3.number_input("Error máximo E", value=1.0, min_value=0.0001)
        else:
            S = None
            P = c2.number_input("Proporción esperada P", value=0.008, min_value=0.0001, max_value=0.9999, format="%.4f")
            E = c3.number_input("Error máximo E (absoluto)", value=0.003, min_value=0.0001, format="%.4f")
        reg = reg_tamano(f"{tipo} · N={N}, {'S=' + str(S) if S else 'P=' + str(P)}, E={E}, {conf:.0%}",
                         tipo, N, S, E, conf, P)
        m1, m2, m3 = st.columns(3)
        m1.metric("Z", f"{reg['res_Z']:.4f}")
        m2.metric("n calculado", f"{reg['res_n_calculado']:.2f}")
        m3.metric("n entero", reg["res_n_entero"])

        st.subheader("Sensibilidad de n frente a E")
        Es = np.linspace(E * 0.4, E * 2.5, 80)
        filas = []
        for c in [0.90, 0.95, 0.99]:
            for e in Es:
                if tipo == "Media · población finita":
                    nn = n_media_finita(N, S, e, c)[1]
                elif tipo == "Media · población infinita":
                    nn = n_media_infinita(S, e, c)[1]
                else:
                    nn = n_proporcion(N, P, e, c)[1]
                filas.append({"E": e, "n": nn, "Confianza": f"{c:.0%}"})
        curva = alt.Chart(pd.DataFrame(filas)).mark_line().encode(
            x=alt.X("E:Q", title="Error tolerado E"), y=alt.Y("n:Q", title="Tamaño de muestra n"),
            color=alt.Color("Confianza:N"))
        punto = alt.Chart(pd.DataFrame([{"E": E, "n": reg["res_n_calculado"]}])).mark_point(
            size=120, filled=True, color="red").encode(x="E:Q", y="n:Q")
        st.altair_chart(curva + punto, use_container_width=True)
    reg["Condición"] = st.text_input("Nombre de la condición para la bitácora", reg["Condición"])
    boton_registrar(reg, "plan", pred)

# --- Neyman
elif modulo == "Muestreo estratificado · Neyman":
    pred = caja_prediccion("ney")
    tabla = st.data_editor(NEYMAN_GUIA, num_rows="dynamic", use_container_width=True, key="tabla_ney")
    n_total = st.number_input("n total", value=3000, min_value=1)
    tabla = tabla.dropna()
    if len(tabla) and (tabla["N_i"] * tabla["S_i"]).sum() > 0:
        res = neyman(tabla, n_total)
        st.dataframe(res.style.format({"N_i·S_i": "{:,.1f}", "Peso Neyman": "{:.1%}",
                                       "n_i Neyman": "{:,.1f}", "n_i proporcional": "{:,.1f}"}),
                     use_container_width=True)
        larga = res.melt(id_vars="Estrato", value_vars=["n_i Neyman", "n_i proporcional"],
                         var_name="Asignación", value_name="n_i")
        st.altair_chart(alt.Chart(larga).mark_bar().encode(
            x=alt.X("Estrato:N"), xOffset="Asignación:N", y="n_i:Q", color="Asignación:N"),
            use_container_width=True)
        st.caption("Neyman asigna n_i ∝ N_i·S_i: los estratos grandes y/o más variables reciben más observaciones.")
        cond = st.text_input("Nombre de la condición para la bitácora", "Neyman")
        boton_registrar(regs_neyman(cond, tabla, n_total), "ney", pred)

# --- Simulación
elif modulo == "Simular muestreo":
    pred = caja_prediccion("sim")
    tipo = st.radio("Estimador", ["Media", "Proporción"], horizontal=True)
    c = st.columns(5)
    mu = sigma = p = None
    if tipo == "Media":
        mu = c[0].number_input("μ", value=10.0)
        sigma = c[1].number_input("σ", value=2.0, min_value=0.0001)
    else:
        p = c[0].number_input("p poblacional", value=0.3, min_value=0.001, max_value=0.999)
    R = c[2].number_input("Réplicas R", value=1000, min_value=10, step=100)
    semilla = c[3].number_input("Semilla", value=1234, step=1)
    n = c[4].select_slider("Tamaño de muestra n", [5, 10, 25, 50, 100, 250, 500, 1000], value=10)
    est, ee = simular(tipo, mu, sigma, p, R, semilla, n)
    m = st.columns(4)
    m[0].metric("Media de las estimaciones", f"{est.mean():.4f}")
    m[1].metric("DE empírica", f"{est.std(ddof=1):.4f}")
    m[2].metric("EE teórico", f"{ee:.4f}")
    m[3].metric("Diferencia", f"{est.std(ddof=1) - ee:+.4f}")
    st.altair_chart(alt.Chart(pd.DataFrame({"estimación": est})).mark_bar().encode(
        x=alt.X("estimación:Q", bin=alt.Bin(maxbins=40), title=f"{tipo.lower()} muestral"),
        y=alt.Y("count()", title="Frecuencia")).properties(
        title=f"Distribución muestral ({tipo.lower()}, n={n}, R={R}, semilla={semilla})"), use_container_width=True)
    reg = reg_sim(f"Exp4 · {tipo} n={n}", tipo, mu, sigma, p, R, semilla, n)
    reg["Condición"] = st.text_input("Nombre de la condición para la bitácora", reg["Condición"])
    boton_registrar(reg, "sim", pred)

# --- Inferencia
elif modulo == "Inferencia":
    prueba = st.selectbox("Prueba", ["t de una muestra", "t pareada", "t dos medias (varianza agrupada)",
                                     "t de Welch", "F para varianzas", "ANOVA de una vía"])
    pred = caja_prediccion("inf")
    c1, c2 = st.columns(2)
    conf = c1.select_slider("Confianza", [0.90, 0.95, 0.99], value=0.95, format_func=lambda v: f"{v:.0%}")
    alfa = round(1 - conf, 2)
    alt_map = {"bilateral": "two-sided", "unilateral menor (<)": "less", "unilateral mayor (>)": "greater"}
    alt_ = alt_map[c2.selectbox("Alternativa", list(alt_map))] if prueba not in ("F para varianzas", "ANOVA de una vía") else "two-sided"

    try:
        if prueba == "t de una muestra":
            x = parse_serie(st.text_input("Datos (separados por comas)", AZUFRE))
            mu0 = st.number_input("μ0 (valor de referencia)", value=0.123, format="%.4f")
            reg = reg_t1(f"t una muestra · {conf:.0%}".replace("%", " %"), x, mu0, alfa, alt_)
            gl = len(x) - 1
            tcrit = stats.t.ppf(1 - alfa / 2, gl)
            m = st.columns(5)
            m[0].metric("Media", f"{reg['res_media']:.4f}")
            m[1].metric("s", f"{reg['res_s']:.5f}")
            m[2].metric("EE", f"{reg['res_EE']:.5f}")
            m[3].metric("t", f"{reg['res_t']:.4f}")
            m[4].metric("p", f"{reg['res_p']:.4f}")
            st.info(f"gl = {gl} · t crítico bilateral = {tcrit:.3f} · Decisión (α = {alfa}): **{reg['Decisión/Interpretación']}**")
            datos = pd.DataFrame({"valor": x, "grupo": "muestra"})
            graf = alt.Chart(datos).mark_circle(size=90).encode(x="grupo:N", y=alt.Y("valor:Q", scale=alt.Scale(zero=False)))
            ref = alt.Chart(pd.DataFrame({"y": [mu0]})).mark_rule(color="red", strokeDash=[4, 4]).encode(y="y:Q")
            st.altair_chart(graf + ref, use_container_width=True)
        elif prueba in ("t pareada", "t dos medias (varianza agrupada)", "t de Welch", "F para varianzas"):
            a = parse_serie(st.text_input("Serie A", COL_A))
            b = parse_serie(st.text_input("Serie B", COL_B))
            if prueba == "t pareada":
                reg = reg_pareada("t pareada", a, b, alfa, alt_)
                m = st.columns(5)
                m[0].metric("Media de diferencias", f"{reg['res_media_dif']:.4f}")
                m[1].metric("s_d", f"{reg['res_s_d']:.4f}")
                m[2].metric("t", f"{reg['res_t']:.4f}")
                m[3].metric("gl", reg["res_gl"])
                m[4].metric("p", f"{reg['res_p']:.4f}")
                st.dataframe(pd.DataFrame({"A": a, "B": b, "d = A − B": a - b}), use_container_width=True)
            elif prueba == "F para varianzas":
                reg = reg_f("F varianzas", a, b, alfa)
                m = st.columns(3)
                m[0].metric("F = s²mayor / s²menor", f"{reg['res_F']:.3f}")
                m[1].metric("gl", reg["res_gl"])
                m[2].metric("p (bilateral)", f"{reg['res_p']:.4f}")
                st.warning("La prueba F es muy sensible a la falta de normalidad: verifique supuestos (Shapiro-Wilk).")
            else:
                welch = prueba == "t de Welch"
                reg = reg_dos_medias(prueba, a, b, alfa, welch, alt_)
                m = st.columns(4)
                m[0].metric("Diferencia de medias", f"{reg['res_media_dif']:.4f}")
                m[1].metric("t", f"{reg['res_t']:.4f}")
                m[2].metric("gl", reg["res_gl"])
                m[3].metric("p", f"{reg['res_p']:.4g}")
            st.info(f"Decisión (α = {alfa}): **{reg['Decisión/Interpretación']}**")
            datos = pd.concat([pd.DataFrame({"valor": a, "grupo": "A"}), pd.DataFrame({"valor": b, "grupo": "B"})])
            st.altair_chart(alt.Chart(datos).mark_circle(size=90).encode(
                x="grupo:N", y=alt.Y("valor:Q", scale=alt.Scale(zero=False)), color="grupo:N"), use_container_width=True)
        else:
            texto = st.text_area("Un grupo por línea:  Nombre: v1, v2, v3 ...",
                                 "\n".join(f"{k}: {v}" for k, v in AUG.items()), height=120)
            nombres, grupos = [], []
            for i, linea in enumerate(l for l in texto.splitlines() if l.strip()):
                nom, _, vals = linea.rpartition(":") if ":" in linea else (f"Grupo {i + 1}", "", linea)
                nombres.append(nom.strip() or f"Grupo {i + 1}")
                grupos.append(parse_serie(vals))
            regs = regs_anova("ANOVA", nombres, grupos, alfa)
            o = anova(grupos)
            st.dataframe(pd.DataFrame({
                "Fuente": ["Entre grupos", "Dentro (error)", "Total"],
                "SS": [o["SS_entre"], o["SS_dentro"], o["SS_entre"] + o["SS_dentro"]],
                "gl": [o["gl"][0], o["gl"][1], o["N"] - 1],
                "MS": [o["MS_entre"], o["MS_dentro"], None],
                "F": [o["F"], None, None], "p": [o["p"], None, None]}), use_container_width=True)
            st.info(f"F crítico = {stats.f.ppf(1 - alfa, *o['gl']):.3f} · η² = {o['eta2']:.3f} · "
                    f"Decisión (α = {alfa}): **{decision(o['p'], alfa)}**")
            datos = pd.concat([pd.DataFrame({"valor": g, "grupo": n}) for n, g in zip(nombres, grupos)])
            st.altair_chart(alt.Chart(datos).mark_circle(size=70).encode(
                x="grupo:N", y=alt.Y("valor:Q", scale=alt.Scale(zero=False)), color="grupo:N"), use_container_width=True)
            reg = regs
        if isinstance(reg, dict):
            reg["Condición"] = st.text_input("Nombre de la condición para la bitácora", reg["Condición"])
        boton_registrar(reg, "inf", pred)
    except (ValueError, ZeroDivisionError) as e:
        st.error(f"Revise los datos ingresados: {e}")

# --- Bitácora
else:
    df = bitacora_df()
    c1, c2, c3 = st.columns(3)
    if c1.button("Cargar experimentos de la guía (Exp 1–6 + libre)"):
        cargar_experimentos_guia()
        st.rerun()
    if c2.button("Borrar última medición") and st.session_state.bitacora:
        st.session_state.bitacora.pop()
        st.rerun()
    if c3.button("Vaciar bitácora"):
        st.session_state.bitacora = []
        st.rerun()
    vista = df.dropna(axis=1, how="all")
    for c in vista.select_dtypes("object").columns:  # columnas mixtas (p. ej. gl "2, 27")
        vista[c] = vista[c].astype("string")
    st.dataframe(vista, use_container_width=True, height=500)
    if not df.empty:
        d1, d2 = st.columns(2)
        d1.download_button("Descargar CSV", df.to_csv(index=False).encode("utf-8-sig"),
                           "experimentos_muestreo_inferencia.csv", "text/csv")
        buf = io.BytesIO()
        df.to_excel(buf, index=False, sheet_name="Bitacora")
        d2.download_button("Descargar Excel", buf.getvalue(), "experimentos_muestreo_inferencia.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Laboratorio virtual: Muestreo e Inferencia Estadística aplicada a experimentos de IA

Aplicación en Streamlit para el laboratorio de muestreo e inferencia estadística. Sigue la secuencia de la guía: predecir → modificar variables → simular → registrar → analizar → interpretar.

**Autor:** Juan Manuel Márquez Jiménez

## Módulos

| Módulo | Qué calcula |
|---|---|
| Planificar muestreo | Tamaño de muestra para una media (población finita o infinita), para una proporción y número de réplicas. Incluye la gráfica de sensibilidad de n frente a E para 90 %, 95 % y 99 %. |
| Muestreo estratificado · Neyman | Asignación n_i ∝ N_i·S_i y comparación con la asignación proporcional a N_i. |
| Simular muestreo | Distribución muestral de la media o de la proporción (R réplicas, semilla fija): DE empírica frente al error estándar teórico. |
| Inferencia | t de una muestra, t pareada, t con varianza agrupada, t de Welch, F para varianzas y ANOVA de una vía, con p-valor, decisión y tamaño del efecto. |
| Bitácora y descarga | Registro de mediciones y descarga en CSV o Excel. El botón "Cargar experimentos de la guía" registra los experimentos 1 a 6 y el experimento libre. |

## Ejecutar en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

En Windows también se puede abrir con doble clic en `iniciar_app.bat`. La aplicación queda en http://localhost:8501.

## Ecuaciones implementadas

- Media, población finita: n = S² / (E²/Z² + S²/N)
- Media, población infinita: n = Z²S² / E²
- Proporción, población finita: n = Z²PQN / (E²(N−1) + Z²PQ)
- Neyman: n_i = n · N_i S_i / Σ N_j S_j
- Réplicas: n = t² s² / E² (t con n−1 grados de libertad, calculado por iteración)
- Pruebas t, F y ANOVA con `scipy.stats`

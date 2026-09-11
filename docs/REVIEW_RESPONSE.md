# Evaluación de las observaciones — 6 de septiembre de 2026

Las observaciones mejoran la presentación de los resultados. He incorporado los problemas comprobados y corregido tres interpretaciones que iban más allá de lo que permiten los datos.

| Observación | Evaluación y cambio |
|---|---|
| Cero aceptaciones no demuestra una buena política | Correcto. La política es inoperante en el panel OOF de trametinib y afatinib. Se añade un diagnóstico por fold de radio, límites de clase, rango predicho y margen disponible. |
| La regla nunca puede aceptar, ni con datos perfectos | No es correcto en general. Las clases son semirrectas: un intervalo puede quedar dentro de una cola aunque su radio supere el hueco entre clases. Con residuos menores puede aceptar; palbociclib ya tiene una aceptación técnica por cada familia molecular en los OOF guardados. |
| Primera aceptación con cobertura nominal 65–69% / 77–79% | No verificable con los artefactos originales: sólo contienen el radio al 90%, no la distribución de residuos de calibración de cada fold. No se publican esos porcentajes como resultados. Se añade el cálculo exacto para futuras ejecuciones que sí guardan esos residuos. |
| El radio identifica el techo de ruido del ensayo | No. Mezcla error del modelo, ruido de medición y heterogeneidad. La no cobertura marginal de un intervalo tampoco equivale al error de clasificación entre casos seleccionados. |
| Lineage tiene empates en el top-10 | Correcto. La lista determinista por ModelID estaba definida y da 5; su sensibilidad al desempate no estaba suficientemente destacada. El rango es 5–6 y la expectativa uniforme 5.2. Se actualizan tablas, gráficos y narrativa. |
| El intervalo ponderado está dominado por ceros | Correcto en esencia, pero el IC de K=5 corresponde a 48 pares: 32 son numéricamente cero y 16 proceden de dos targets. Los 10 escenarios no uniformes entre K=2 y K=5 no son diez réplicas independientes. El intervalo tiene ancho no nulo; su extremo superior es numéricamente cero. |
| Gemcitabina muestra una penalización por complejidad | Correcto. Diferencia XGBoost−ElasticNet −0.1028, IC condicional [−0.1779, −0.0224]. Se destaca en el README, un gráfico propio y el seminario. El intervalo es exploratorio, sin ajuste por multiplicidad ni incertidumbre de reentrenamiento. |
| Los outcomes intervienen en la estratificación | Correcto. La asignación de folds usa cuartiles de rangos del outcome completo. Se declara en los manifiestos y el código; no se presenta como asignación prospectiva ciega al outcome. |
| Abrir con lo que el diseño permitió establecer | Incorporado. El resultado positivo es un proceso reproducible capaz de detectar una elección de modelado no respaldada y definir una prueba experimental concreta, sin atribuirse ahorro económico o eficacia que no se han medido. |

## Qué se pudo cuantificar ahora

Con las predicciones OOF y la regla de margen congeladas, XGBoost necesita multiplicar los radios originales por **0.5431** en trametinib o **0.8034** en afatinib para que aparezca al menos una aceptación técnica. Equivale a reducir esos radios un **45.7% / 19.7%**. Son escalas de radio, no coberturas nominales y no una recomendación para modificar la política. Otros controles de la API pueden seguir bloqueando esas decisiones.

La comparación debe mantenerse dentro de cada fold y familia. Los radios OOF de XGBoost son 0.2463–0.2911 y 0.1768–0.2554; los bundles ajustados finales tienen radios diferentes, 0.3318 y 0.3045. No deben mezclarse.

En el PoC, la probabilidad de que seis extracciones de targets con reemplazo omitan los dos targets activos es `(4/6)^6 = 8.78%`. Es una explicación estructural de la masa cerca de cero en el bootstrap, no una prueba alternativa seleccionada después de ver el resultado. Se conservan el estimando y el intervalo primarios.

## Evidencia reproducible

- [Diagnóstico de intervalos por fold](../reports/pierre_fabre_review/interval_feasibility.csv).
- [Sensibilidad al radio](../reports/pierre_fabre_review/interval_radius_sensitivity.csv).
- [Resultados de presupuesto con empates](../reports/pierre_fabre_review/screening_budget_summary.csv).
- [Diferencias pareadas de AUROC](../reports/pierre_fabre_review/paired_auroc_differences.csv).
- [Revisión científica completa](PIERRE_FABRE_REVIEW.md).
- El complemento del PoC incluye las tablas originales de pesos, pares y resultados, con `audit_weighted.py` y `audit_output/weighted_structure.json`.

La política de producción no se ha relajado. No se han reentrenado los modelos ni se han inventado residuos, nuevos experimentos o niveles de cobertura.

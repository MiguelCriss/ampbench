# Fase 0 — Inventario y triage

Fecha: 2026-11-13  
Fuente: carpeta `MODELOS/` (proyecto de modelado de amplificador de potencia / tesis).

## Resumen ejecutivo

| Categoría | Acción | Volumen aprox. |
|-----------|--------|----------------|
| Código Python (núcleo) | **REWRITE** (mantener lógica) | ~10 archivos |
| Scripts de plot | **REWRITE** | 3 archivos |
| Datos crudos de curvas | **KEEP** (selección) | ~10 `.txt` pequeños |
| Dataset I/Q procesado | **KEEP** `dataset.txt`; splits regenerables | ~1.5–3 MB |
| Artefactos (pkl/h5/pred) | **DROP** del repo (gitignore / `artifacts/`) | ~15+ MB |
| PDFs datasheets fabricantes | **DROP** (no redistribuir) | ~9+ MB |
| Excel intermedios | **DROP** o local-only | ~25 MB |
| `__pycache__` | **DROP** | — |

**Destino propuesto del repo:** `rf-pa-ml-toolkit` (nombre a confirmar en Fase 1).

---

## 1. KEEP — conservar (con posible rename en Fase 1–2)

### 1.1 Código fuente (lógica valiosa → se reescribe, no se tira)

| Archivo | Rol | Notas |
|---------|-----|-------|
| `ANN/ANN_new.py` | Entrenamiento ANN (Keras) | Script monolítico; → `scripts/train_ann.py` |
| `ANN/funciones_ANN.py` | Leer datos, normalizar, memoria, loss NMSE | Núcleo ANN; → `src/.../ann/` |
| `metodos_tradicionales/funciones.py` | I/O, métricas, plots, utilidades | Archivo grande (~29 KB); partir en módulos |
| `metodos_tradicionales/funciones_modelos.py` | MPM / Hammerstein / Wiener | Núcleo modelos clásicos |
| `metodos_tradicionales/models.py` | Entry point entrenamiento modelos | → `scripts/train_traditional.py` |
| `metodos_tradicionales/modelo_construido.py` | Inferencia con coeficientes guardados | Útil; limpiar |
| `dataset/gen_dat_MATLAB.py` | Split 70/15/15 → train/valid/test | Renombrar (no es MATLAB); KEEP lógica |
| `dataset/gen_dat_PAcomercial.py` | Curvas PA → dataset / plots | Fusionar con `gen_new.py` |
| `dataset/gen_new.py` | Interpolación / generación | Duplica mucho con `gen_dat_PAcomercial.py` |
| `dataset/1 data TGA2595/graficar.py` | Plot curvas TGA2595 | Integrar en plotting package |
| `graf1.py` | Time series ANN vs original | Depende de `.pkl` → reescribir |
| `graf2.py` | PSD comparación modelos | Idem |
| `graf3.py` | AM/AM–AM/PM comparación | Idem |

### 1.2 Datos crudos (prioridad: TGA2595 = caso 5G principal)

| Archivo | Decisión |
|---------|----------|
| `dataset/1 data TGA2595/vout_vs_vin.txt` | **KEEP** → `data/raw/tga2595/` |
| `dataset/1 data TGA2595/fase_vs_vin.txt` | **KEEP** |
| `dataset/1 data TGA2595/Gain_vs_Pin.txt` | **KEEP** |
| `dataset/1 data TGA2595/AMPM_vs_Pout.txt` | **KEEP** |
| `dataset/2 data K-HPA10W-1721-SSPA/vout_vs_vin.txt` | **KEEP** (segundo PA) |
| `dataset/2 data K-HPA10W-1721-SSPA/fase_vs_vin.txt` | **KEEP** |
| `dataset/2 data K-HPA10W-1721-SSPA/Errores.txt` | **KEEP** (notas de error; revisar redacción) |
| `dataset/datos_originales.txt` | **KEEP** (curvas digitadas compactas) |

### 1.3 Datos procesados (señal I/Q para entrenar)

| Archivo | Decisión |
|---------|----------|
| `dataset/dataset.txt` (~1.5 MB, 24576 filas, t/I/Q) | **KEEP** → `data/processed/` (dataset principal) |
| `dataset/train.txt` / `valid.txt` / `test.txt` | **KEEP opcional** o regenerar con script de split; recomendación Fase 2: regenerar + documentar, o versionar si el split debe ser fijo |

### 1.4 Exportes pequeños útiles (revisar en Fase 2)

| Archivo | Decisión |
|---------|----------|
| `ANN/weights_and_biases.txt` | **KEEP** como ejemplo de export ADS/FDD (no sustituye `.h5`) |
| `eq.txt` / `ANN/eq_ANN.txt` | **KEEP** en `docs/equations/` o `artifacts/exports/` documentados |
| `coeficientes.pkl` / `coeficientes.mat` | **KEEP** solo si se documenta el modelo (N,M); si no, regenerar |

---

## 2. DROP — no subir a GitHub (borrar del tree público o gitignore)

### 2.1 Basura / regenerable automático

| Archivo / patrón | Motivo |
|------------------|--------|
| `**/__pycache__/**` | Bytecode |
| `*.pyc` | Bytecode |

### 2.2 Artefactos de entrenamiento / predicción (regenerables)

| Archivo | Tamaño aprox. | Motivo |
|---------|---------------|--------|
| `modelo_entrenado.h5` | 56 KB | Peso; regenerable; duplicado de keras |
| `modelo_entrenado_keras.h5` | 56 KB | Duplicado |
| `timeseries.pkl` | 576 KB | Artefacto plot |
| `timeseries_sinM.pkl` | 576 KB | Artefacto plot |
| `datos_salida.pkl` | 576 KB | Artefacto plot |
| `datos_ann.pkl`, `datos_ham.pkl`, `datos_wie.pkl`, `datos_mpm.pkl` | ~2 KB c/u | Artefactos plot |
| `psd.pkl`, `psd_ann.pkl`, `psd_ann_nuevo.pkl`, `psd_hamm.pkl`, `psd_wie.pkl`, `psd_mpm.pkl` | ~384 KB c/u | Artefactos PSD |
| `predicciones_MPM.txt` | ~3 MB | Predicción larga |
| `predicciones_Hammerstein.txt` | ~3 MB | Idem |
| `predicciones_Wiener.txt` | ~3 MB | Idem |
| `ANN/sim_ann.txt` | ~1 MB | Simulación/export largo |

**Política:** carpeta `artifacts/` en `.gitignore`. Opcional: un `.h5` pequeño de demo más adelante si hace falta para Alignerr (Fase 6).

### 2.3 No redistribuir (copyright / datasheets)

| Archivo | Motivo |
|---------|--------|
| `dataset/1 data TGA2595/TGA2595 datasheet amplificador 5g.pdf` | Datasheet fabricante |
| `dataset/1 data TGA2595/Gain_vs_Pin_grafica.pdf` | Figura/datasheet |
| `dataset/2 data K-HPA10W-1721-SSPA/ReliaSat_-K-HPA10W-1721-SSPA-4.pdf` | Datasheet |
| `dataset/3 data Q-HPA2W-3743/Q-HPA2W-3743.pdf` | Datasheet |
| `dataset/4 data HPA11201721QF/HPA11201721QF.pdf` | Datasheet |

En README: citar fabricante + enlace público al datasheet, **no** adjuntar el PDF.

### 2.4 Excel / CSV pesados (intermedios o duplicados)

| Archivo | Tamaño | Motivo |
|---------|--------|--------|
| `dataset/datos_simulink_QAM256 definitivo.xlsx` | ~23 MB | Intermedio Simulink; no versionar |
| `dataset/*/datos*.xlsx`, `data amp*.xlsx` | ~0.2–0.3 MB c/u | Preferir `.txt` digitados; Excel local-only |
| `amplifier_data.csv` | ~3.8 MB | Revisar si = derivado de `dataset.txt`; si sí → DROP |
| `PA_datos.csv` | ~1.5 MB | Probable ADS/export; confirmar uso |
| `PA_datos.ds` | ~3 KB | Formato Keysight ADS; no útil en repo Python puro |

### 2.5 Datos de PAs 3 y 4 sin curvas `.txt` listas

| Carpeta | Decisión |
|---------|----------|
| `dataset/3 data Q-HPA2W-3743/` (solo xlsx+pdf) | **DROP** del repo público por ahora (o extraer `.txt` en Fase 2 si se necesitan) |
| `dataset/4 data HPA11201721QF/` (solo xlsx+pdf) | **DROP** igual |

---

## 3. REWRITE — problemas a corregir en fases siguientes

| Problema | Dónde | Fase |
|----------|-------|------|
| `sys.path` hack / imports frágiles | `ANN_new.py`, `funciones_modelos.py` | 3 |
| Paths hardcodeados (`dataset/...`, `*.pkl` en cwd) | Casi todos los scripts | 3–4 |
| Duplicación `leer` vs `read_data` | ANN vs tradicionales | 3 |
| `gen_new.py` ≈ `gen_dat_PAcomercial.py` | dataset/ | 2 |
| Nombre engañoso `gen_dat_MATLAB.py` | solo hace split | 2 |
| `funciones.py` hace demasiadas cosas (I/O + métricas + plots) | tradicionales | 3–4 |
| Comentarios/redacción en español mezclado; typos | varios | 3–5 |
| Plots con LaTeX obligatorio (`graf1`) | graf1.py | 4 |
| Estilo plot muy “tesis” (colores extremos) | gen_*.py | 4 |
| Sin `requirements.txt` / empaquetado | — | 1 |
| Sin README profesional | — | 5 |
| Modelos tradicionales importan `from funciones import *` | models.py | 3 |

---

## 4. Mapa lógico del proyecto (para Fase 1)

```
Curvas fabricante (txt)
    → gen dataset / interpolación
        → dataset.txt (I/Q o mag/fase)
            → split train/valid/test
                → ANN (Keras) ──→ métricas NMSE + plots
                → MPM / Hammerstein / Wiener ──→ métricas + plots
                    → (opcional) export ecuaciones / pesos para ADS
```

**Núcleo académico alineado con la entrevista:** pipeline de datos → modelo (ANN + baselines) → juicio por NMSE y curvas AM/AM–AM/PM.

---

## 5. Decisiones pendientes (confirmar antes de borrar)

1. **¿Nombre del repo?** sugerido: `rf-pa-ml-toolkit`
2. **¿Versionar `train/valid/test.txt` o solo `dataset.txt` + script de split?**
3. **¿Incluir PAs 2–4 o solo TGA2595 al inicio?**
4. **¿Conservar Excel/PDF en una carpeta local `_local_only/` fuera de git, o eliminar del workspace?**
5. **¿Idioma del código/README?** EN (GitHub/Alignerr) recomendado; comentarios ES opcionales

---

## 6. Criterio de cierre Fase 0

- [x] Inventario completo de archivos actuales  
- [x] Clasificación KEEP / DROP / REWRITE  
- [ ] Confirmación del usuario en decisiones pendientes  
- [ ] Listo para ejecutar Fase 1 (esqueleto + `.gitignore` según este triage)

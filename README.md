# 32 GB de VRAM gratis (y fácil)

Un LLM de 35B sirviéndose desde **dos Tesla T4 (32 GB de VRAM en total)** con las **30 horas
semanales de GPU gratuita de Kaggle**, publicado con **Web UI y API compatible con OpenAI** por
un túnel de Cloudflare.

**La idea, en una frase:** Kaggle regala 30 h/semana de `GPU T4 x2`; un notebook descarga
llama.cpp y el modelo, lo sirve y abre un túnel, y ya tienes un LLM propio con URL pública —
sin tarjeta, sin compilar y sin pagar.

## Qué te llevas

| | |
|---|---|
| Hardware | 2 × Tesla T4 = 32 GB de VRAM (16 GB por tarjeta; `nvidia-smi` reporta 15360 MiB usables por tarjeta) |
| Cuota | 30 h/semana por cuenta verificada (Kaggle a veces da más según demanda); resetea el sábado a las 00:00 UTC |
| Sesión | 12 h seguidas como máximo; después hay que volver a arrancar |
| Salida | Web UI en `https://<algo>.trycloudflare.com/` y API en `.../v1` |
| Coste | 0 |

## Cómo funciona

```text
        navegador  ·  cualquier cliente OpenAI
                     │ https
                     ▼
        Cloudflare quick tunnel  ──►  llama-server  (127.0.0.1:8080)
                                            │
                              ┌─────────────┴─────────────┐
                        Web UI  /                    API  /v1
                                            │
                            GGUF en /kaggle/tmp + binario oficial de llama.cpp con CUDA
```

El notebook hace cuatro cosas, en este orden: baja el binario oficial de llama.cpp (sin
compilar), baja el modelo, arranca `llama-server` repartiendo las capas entre las dos T4, y
abre el túnel.

## Antes de empezar: la cuenta de Kaggle

1. Crear cuenta (con el correo verificado).
2. **Verificar el teléfono** — *Settings → Phone Verification*. Es el requisito que desbloquea
   la GPU: sin SMS, `GPU T4 x2` no aparece en *Accelerator*. A algunas cuentas Kaggle les pide
   además una verificación de identidad (documento/rostro); si te sale, hazla, pero la que abre
   la GPU es la del teléfono.
3. Mirar la cuota disponible en *Settings → Quotas*.

## Pasos

1. **Code → New Notebook**.
2. En el panel derecho, *Settings*:
   - **Accelerator → GPU T4 x2** (la sesión se reinicia al cambiarlo)
   - **Internet → On** (el notebook descarga de GitHub y de Hugging Face; sin esto no hace nada)
3. Pegar el contenido de [`notebook/kaggle_llm.py`](notebook/kaggle_llm.py) en una celda y
   ejecutarla (o importar [`notebook/kaggle_llm.ipynb`](notebook/kaggle_llm.ipynb) desde
   *File → Import Notebook*).
4. Esperar la primera descarga: ~23 GB entre modelo y proyector.
5. Al terminar imprime la URL:

```text
Web UI   : https://xxxx.trycloudflare.com/
Base URL : https://xxxx.trycloudflare.com/v1
```

La Web UI se abre en el navegador; la API la consume cualquier cliente OpenAI-compatible.

## Ajustes rápidos (arriba del código)

| Variable | Para qué |
|---|---|
| `MODEL_URL` | Cualquier GGUF de Hugging Face; da igual el link de la página (`/blob/`) o el de descarga (`/resolve/`) |
| `MMPROJ_URL`, `MMPROJ_STATE` | Proyector multimodal (visión); `False` si no lo quieres |
| `MODO` | `"webui"` (Web UI + API) o `"api"` (solo `/v1`, sin Web UI) |
| `API_KEY_ON` | `""` = automático, `True` = siempre con clave, `False` = nunca |
| `ARGS_BASE` | Flags de `llama-server`: contexto, reparto entre tarjetas, tipo de KV, decodificación especulativa, sampling |
| `LLAMA_RELEASE` | Fijar un release concreto de llama.cpp si el último no arranca en tu máquina |

Por defecto: Qwen3.6-35B-A3B en Q4_K_M (22.7 GB), contexto de 64k, capas repartidas entre las
dos T4 (`--split-mode layer`, `--tensor-split 1,1`), KV en `q8_0`, flash-attention activada y
decodificación especulativa por n-gramas.

## Apagar

**Stop / Interrupt** en el notebook abre un menú:

1. Parar el modelo y **dejar el túnel con su URL** (para cambiar flags o modelo y volver a
   arrancar sin que cambie la dirección).
2. Parar modelo y túnel.

Cerrar la pestaña del navegador **no** para nada y sigue gastando cuota.

## Lo que esto NO es

- No es un clúster: las dos T4 van partidas **por capas**; no hay entrenamiento distribuido.
- La URL del túnel es **pública y efímera**: quien tenga el link entra. Con `MODO = "webui"` y
  sin clave no hay autenticación de ningún tipo; si el túnel se cae, la URL cambia.
- Kaggle no está pensado para servir aplicaciones: su política restringe el tráfico saliente por
  túneles y hay cuentas suspendidas por hacerlo. Es tu cuenta y es tu riesgo.
- Solo hay 30 h/semana y 12 h por sesión; agotada la cuota, el siguiente arranque falla.
- Nada de cifras de rendimiento aquí: este repo explica el montaje, no mide tokens/s.

## Detalles

Apagado, uso de la API, cuotas y errores típicos: [`docs/detalles.md`](docs/detalles.md).

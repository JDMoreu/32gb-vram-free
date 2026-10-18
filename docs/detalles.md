# Detalles

Complemento del [README](../README.md). Nada de aquí hace falta para el camino corto.

## Cuenta y verificación

- **Correo**: al registrarse.
- **Teléfono** (*Settings → Phone Verification*): obligatorio; sin SMS no aparece la opción de
  GPU en *Accelerator*.
- **Identidad** (documento/rostro): algunas cuentas la ven además; si aparece, complétala.
- Kaggle permite **una cuenta por persona** y un teléfono no vale para varias.

## Cuota y sesión

| | |
|---|---|
| Cuota | 30 h/semana (puede ser más según demanda y recursos) |
| Reset | sábado a las 00:00 UTC |
| Sesión continua | 12 h como máximo |
| Sesión interactiva sin actividad | se cierra a los 60 min |
| Dónde verla | *Settings → Quotas*, y la barra del editor |

La cuota se cobra por **tiempo con la sesión arrancada**, no por uso real de GPU: una sesión
abierta sin hacer nada la consume igual. Se para a mano (*Active Events → Stop*) o con el menú
del notebook.

Cada sesión nueva vuelve a bajar los ~23 GB, porque `/kaggle/tmp` es efímero. Si lo vas a hacer
a menudo, sube el GGUF como Dataset de Kaggle, adjúntalo y enlázalo antes de ejecutar la celda:

```bash
ln -s /kaggle/input/<tu-dataset>/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf /kaggle/tmp/
```

El script lo detecta por tamaño, verifica el SHA256 y se salta la descarga.

## Usar la API

Es compatible con OpenAI, así que sirve cualquier cliente que acepte un `base_url`.

```bash
curl https://xxxx.trycloudflare.com/v1/models

curl https://xxxx.trycloudflare.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3.6-35B-A3B-MTP-GGUF","messages":[{"role":"user","content":"Hola"}]}'
```

Si activas la clave (`API_KEY_ON = True`, o `MODO = "api"`, que la activa sola), añade
`-H "Authorization: Bearer <clave>"`. La clave la imprime el notebook al arrancar; si la quieres
fija, escribe `--api-key TU_CLAVE` dentro de `ARGS_BASE`.

## Menú de apagado

Stop/Interrupt en el notebook:

1. **Para el modelo y deja Cloudflare**: libera la VRAM y conserva la URL del túnel, para cambiar
   `ARGS_BASE` o el modelo y volver a arrancar sin que cambie la dirección.
2. **Para todo**: modelo y túnel.

## Si algo falla

| Síntoma | Qué mirar |
|---|---|
| No aparece `GPU T4 x2` | Falta la verificación del teléfono |
| No descarga nada, error de red | *Internet: On* en Settings del notebook |
| `no kernel image` en el log | El build no trae kernels para la T4 (sm_75): fija `LLAMA_RELEASE` a un release anterior |
| La descarga del GGUF se corta | Se reanuda sola; vuelve a ejecutar la celda |
| El server no arranca con ese bloque de flags | El script reintenta solo, primero sin los `--spec-ngram-mod*` y luego sin ningún `--spec*` |
| El túnel no publica URL | Reintenta; mira `~/.local/state/llm-kaggle/cloudflared.log` |
| No cabe el modelo / OOM | Baja de cuantización (`UD-Q4_K_S`, `UD-IQ4_XS`) o el contexto |
| La URL cambió sola | El túnel se cayó; al recrearlo cambia el subdominio |

Los logs quedan en `~/.local/state/llm-kaggle/` (`llama-server.log`, `cloudflared.log`).

## Aviso de uso

Kaggle no ofrece un servicio pensado para publicar aplicaciones; su política restringe el
tráfico saliente por túneles y hay cuentas suspendidas por usarlo así. Este repo documenta cómo
funciona la técnica; no garantiza que tu cuenta sobreviva a hacerlo.

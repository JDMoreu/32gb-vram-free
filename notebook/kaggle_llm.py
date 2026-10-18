import atexit, hashlib, json, os, platform, re, secrets, shlex, shutil, signal, subprocess, sys
import tarfile, time, urllib.error, urllib.request
from pathlib import Path
from urllib.parse import unquote

# ============================== CONFIGURACIÓN ==============================

# --- Qué modelo descargar y desplegar ---
MODEL_URL = "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/main/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
ALIAS     = "Qwen3.6-35B-A3B-MTP-GGUF"

# --- Ficheros extra: solo para los modelos que los traen; deja "" en los que no ---
# MTP_URL:    cabezal de predicción múltiple. Se pasa solo como modelo draft del spec
#             (--spec-draft-model; en binarios antiguos, -md).
# MMPROJ_URL: proyector multimodal (modelos con visión/audio). Se pasa solo con --mmproj.
# Los dos valen como link de la página (/blob/) o de descarga (/resolve/).
MTP_URL    = ""
MMPROJ_URL = "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/main/mmproj-BF16.gguf"

# Interruptores: aunque el link esté puesto, solo se carga si está en True. Son booleanos,
# así puedes dejar los links guardados y encender/apagar sin borrarlos.
MTP_STATE    = False
MMPROJ_STATE = True

# Del link salen solos el nombre del fichero, el repo y el SHA256 (de SHA256SUMS del repo o de
# lfs.sha256 en la API de Hugging Face); si ninguno de los dos lo publica, se comprueba solo el
# tamaño. El link de la página del fichero (/blob/) también vale: se convierte solo a descarga.

# --- API key ---
# ""    = automático: con key en modo "api" (el túnel publica /v1 y la key lo protege)
#         y sin key en modo "webui" (así la página web entra directa, sin pegar nada).
# True  = siempre con key   ·   False = nunca
# Si quieres una key fija, escribe --api-key TU_CLAVE dentro de ARGS_BASE.
API_KEY_ON = False

# --- Cómo montar el server ---
# ""      = pregunta antes de montarlo
# "api"   = solo API (/v1), sin Web UI
# "webui" = API + Web UI, para probar desde el navegador
MODO = "webui"

# --- Con qué flags se ejecuta el modelo (equivalente a tu .bat, pero sin los ^) ---
# {threads} se sustituye por THREADS. --model, --alias, --host, --port y --api-key
# se añaden solos si no los escribes aquí.
#
# Los --spec-* (decodificación especulativa) van aquí como uno más. Si el server no arranca
# con el bloque tal cual, se reintenta solo quitando primero los --spec-ngram-mod* y luego
# todo lo --spec*, sin que tengas que escribir nada más.
ARGS_BASE = """
    --jinja
    --spec-draft-n-max 2
    --fit off
    --gpu-layers all
    --split-mode layer
    --tensor-split 1,1
    --threads {threads}
    --reasoning on
    --ctx-size 65536
    --flash-attn on
    --cache-type-k q8_0
    --cache-type-v q8_0
    --parallel 1
    --kv-unified
    --temp 0.6
    --top-p 0.95
    --top-k 20
    --min-p 0.0
    --presence_penalty 0.0
    --repeat-penalty 1.0
    --spec-type ngram-mod
    --spec-ngram-mod-n-match 24
    --spec-ngram-mod-n-min 48
    --spec-ngram-mod-n-max 64
"""

THREADS = max(2, os.cpu_count() or 4)   # valor de {threads}
PORT    = 8080                          # solo se usa si no pones --port en ARGS_BASE

# --- Túnel ---
CF_HOST_HEADER = True   # necesario para que la Web UI funcione a través del quick tunnel

# --- llama.cpp: binario oficial (casi nunca hay que tocarlo) ---
LLAMA_RELEASE = ""      # "" = último release; o "b11696" para fijar uno concreto
TAGS_ATRAS    = 25      # cuántos releases anteriores probar si el último no sirve
FALLBACK_TAG  = "b11696"
CUDA_BUILDS   = {"13": ["13.4", "12.8"], "12": ["12.8"]}   # según lo que aguante el driver

# ===========================================================================


HOME      = Path.home()
WORK_DIR  = HOME / ".local/state/llm-kaggle"
PRE_DIR   = HOME / ".local/opt/llama-prebuilt"
CF_BIN    = HOME / ".local/bin/cloudflared"
LLAMA_LOG = WORK_DIR / "llama-server.log"
CF_LOG    = WORK_DIR / "cloudflared.log"
CF_URL_FILE = WORK_DIR / "cloudflared.url"  # URL persistente para reutilizar el mismo túnel


def pick_data_dir():
    """Directorio con sitio para el GGUF: el scratch de Kaggle si existe."""
    for cand in ("/kaggle/tmp", "/kaggle/temp"):
        if Path(cand).is_dir():
            return Path(cand)
    return Path("/tmp")


DATA_DIR = pick_data_dir()


def parsear_url(url):
    """(repo, nombre del fichero) a partir del link del modelo.

    Vale tanto el link de descarga (/resolve/) como el de la página del fichero (/blob/),
    y aguanta parámetros tipo ?download=true. Si no es un link de Hugging Face, deja
    repo = None y el nombre que se pueda sacar de la ruta.
    """
    u = url.strip().replace("/blob/", "/resolve/")
    ruta = u.split("?", 1)[0].split("#", 1)[0]
    m = re.match(r"^https?://huggingface\.co/([^/]+/[^/]+)/(?:resolve|raw)/([^/]+)/(.+)$", ruta)
    if m:
        return m.group(1), unquote(m.group(3))
    return None, (unquote(ruta).rstrip("/").split("/")[-1] or "model.gguf")


def fichero_local(url):
    """Nombre con el que se guarda un link en el scratch (None si el link está vacío)."""
    return Path(parsear_url(url)[1]).name if url.strip() else None


def ruta_local(url):
    n = fichero_local(url)
    return (DATA_DIR / n) if n else None


MODEL_REPO, MODEL_NOMBRE = parsear_url(MODEL_URL)
MODEL = DATA_DIR / Path(MODEL_NOMBRE).name
MTP_LOCAL = ruta_local(MTP_URL)
MMPROJ_LOCAL = ruta_local(MMPROJ_URL)


def _extras_de_config():
    """Los ficheros extra que se van a usar: link puesto Y su interruptor en True."""
    puestos = []
    for etiqueta, url, ruta, activo in (("MTP", MTP_URL, MTP_LOCAL, MTP_STATE),
                                        ("MMPROJ", MMPROJ_URL, MMPROJ_LOCAL, MMPROJ_STATE)):
        if not activo:
            continue
        if not url.strip() or not ruta:
            say(f"  AVISO: {etiqueta}_STATE = True pero {etiqueta}_URL está vacío; se ignora.")
            continue
        puestos.append((etiqueta, url, ruta))
    return puestos


def extras():
    """[(etiqueta, url, ruta en disco)] de los ficheros extra en uso."""
    return EXTRAS


# (EXTRAS y EXTRAS_NOMBRES se calculan más abajo, ya definido say(), que el aviso lo usa.)


for d in (WORK_DIR, DATA_DIR, CF_BIN.parent):
    d.mkdir(parents=True, exist_ok=True)

LIB_DIRS = []   # directorios con librerías del binario

_proc = None    # proceso de llama-server
_cf = None      # proceso de cloudflared
_help = None    # cache de llama-server --help


def say(m=""):
    print(m, flush=True)


def _read_proc_cmdline(pid):
    """Devuelve la línea de comandos de un proceso Linux, o cadena vacía."""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        return raw.replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except (OSError, ProcessLookupError):
        return ""


def _find_managed_pids(kind):
    """Encuentra servidores huérfanos de ejecuciones anteriores en el puerto configurado."""
    found = set()
    proc_root = Path("/proc")
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return found

    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid():
            continue
        cmdline = _read_proc_cmdline(pid)
        if not cmdline:
            continue
        args = cmdline.split()

        if kind == "llama":
            is_llama = any(Path(arg).name == "llama-server" for arg in args)
            has_port = any(args[i] == "--port" and i + 1 < len(args)
                           and args[i + 1] == str(PORT)
                           for i in range(len(args)))
            if is_llama and has_port:
                found.add(pid)
        elif kind == "cloudflared":
            is_cloudflared = any(Path(arg).name == "cloudflared" for arg in args)
            is_tunnel = "tunnel" in args
            targets_server = f"http://127.0.0.1:{PORT}" in args
            if is_cloudflared and is_tunnel and targets_server:
                found.add(pid)
    return found


def _send_signal_to_process(pid, sig):
    """Envía señal al grupo del proceso si es su líder; si no, solo al proceso."""
    try:
        pgid = os.getpgid(pid)
        if pgid == pid:
            os.killpg(pgid, sig)
        else:
            os.kill(pid, sig)
    except ProcessLookupError:
        pass
    except PermissionError as exc:
        say(f"  No se pudo señalizar PID {pid}: {exc}")


def _pid_exists(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _stop_pids(pids, known_proc=None, label="servicio"):
    """Termina procesos gestionados, espera su salida y fuerza el cierre si hace falta."""
    pids = set(pids)
    if known_proc is not None and known_proc.poll() is None:
        pids.add(known_proc.pid)
    if not pids:
        if known_proc is not None:
            try:
                known_proc.wait(timeout=0.1)
            except (OSError, subprocess.TimeoutExpired, ChildProcessError):
                pass
        return

    say(f"Deteniendo {label}...")
    for pid in pids:
        _send_signal_to_process(pid, signal.SIGTERM)

    deadline = time.monotonic() + 8
    def still_running(pid):
        if known_proc is not None and pid == known_proc.pid:
            return known_proc.poll() is None
        return _pid_exists(pid)

    while time.monotonic() < deadline and any(still_running(pid) for pid in pids):
        time.sleep(0.2)

    for pid in {pid for pid in pids if still_running(pid)}:
        _send_signal_to_process(pid, signal.SIGKILL)

    if known_proc is not None:
        try:
            known_proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                known_proc.kill()
                known_proc.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                pass
        except (OSError, ChildProcessError):
            pass


def stop_llama_server():
    """Detiene solo llama-server para liberar el modelo; no toca Cloudflare."""
    global _proc
    pids = _find_managed_pids("llama")
    _stop_pids(pids, _proc, "llama-server (modelo y VRAM/RAM)")
    _proc = None
    say("llama-server detenido. Cloudflare permanece activo y conserva su URL.")


def stop_cloudflare():
    """Detiene solo el túnel de Cloudflare."""
    global _cf
    pids = _find_managed_pids("cloudflared")
    _stop_pids(pids, _cf, "Cloudflare")
    _cf = None
    CF_URL_FILE.unlink(missing_ok=True)
    say("Túnel de Cloudflare detenido.")


def cleanup_processes():
    """Apaga llama-server y cloudflared, incluidos procesos huérfanos del puerto actual."""
    stop_llama_server()
    stop_cloudflare()
    say("Servidor y túnel detenidos.")


# Respaldo para cierre normal del kernel/proceso. La limpieza principal va en finally.
atexit.register(cleanup_processes)


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, check=False, **kw)


# Ficheros extra en uso. Se calcula aquí abajo, después de say(), porque el aviso lo usa.
EXTRAS = _extras_de_config()
EXTRAS_NOMBRES = {etiqueta for etiqueta, _, _ in EXTRAS}


def get_secret(name):
    v = os.environ.get(name)
    if v:
        return v.strip()
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name).strip()
    except Exception:
        return None


def free_gb(path):
    sv = getattr(os, "statvfs", None)
    if not sv:
        return None
    try:
        st = sv(str(path))
        return st.f_bavail * st.f_frsize / 1e9
    except Exception:
        return None


def runtime_env():
    env = os.environ.copy()
    libs = [str(d) for d in LIB_DIRS] + [env.get("LD_LIBRARY_PATH", "")]
    env["LD_LIBRARY_PATH"] = ":".join(x for x in libs if x)
    return env


def http_get(url, timeout=60, api=False, token=None):
    headers = {"User-Agent": "kaggle-llm/1.0"}
    if api:
        headers["Accept"] = "application/vnd.github+json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout)


def http_head_size(url, token=None):
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "kaggle-llm/1.0",
                                              **({"Authorization": f"Bearer {token}"} if token else {})})
        with urllib.request.urlopen(req, timeout=30) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except Exception:
        return None


# ---------------------- Modelo: URL y SHA256 ----------------------
def hf_url(repo, filename):
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


def sha_from_repo(repo, nombre):
    """Busca el hash del fichero en los SHA256SUMS que publica el repo de Hugging Face."""
    if not repo:
        return None
    objetivo = Path(nombre).name
    for name in ("SHA256SUMS", "SHA256SUMS.txt", "sha256sums.txt"):
        try:
            with http_get(hf_url(repo, name), 30, token=HF_TOKEN) as r:
                texto = r.read().decode("utf-8", "replace")
        except Exception:
            continue
        for linea in texto.splitlines():
            partes = linea.split()
            if len(partes) >= 2 and Path(partes[-1].lstrip("*")).name == objetivo:
                if re.fullmatch(r"[0-9a-fA-F]{64}", partes[0].strip()):
                    return partes[0].strip().lower()
    return None


def hf_api_meta(repo, nombre):
    """(tamaño, sha256) del fichero según la API de Hugging Face.

    El sha256 sale del campo lfs.sha256, que es el hash del CONTENIDO.
    Ojo: el ETag que devuelve el link de descarga NO sirve para esto — es el blob
    de git del repo, no el SHA-256 del fichero.
    """
    if not repo:
        return None, None
    try:
        with http_get(f"https://huggingface.co/api/models/{repo}?blobs=true",
                      30, token=HF_TOKEN) as r:
            data = json.load(r)
    except Exception:
        return None, None
    objetivo = Path(nombre).name
    for s in data.get("siblings", []):
        if Path(s.get("rfilename", "")).name != objetivo:
            continue
        lfs = s.get("lfs") or {}
        sha = str(lfs.get("sha256") or "").strip().lower()
        size = s.get("size") or lfs.get("size")
        return (int(size) if size else None,
                sha if re.fullmatch(r"[0-9a-f]{64}", sha) else None)
    return None, None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_src(url=None):
    """(url, sha256 esperado, tamaño esperado) de un .gguf del que se sabe el link.

    El repo y el nombre salen del propio link; el hash se busca solo: SHA256SUMS del repo y,
    si ahí no aparece, lfs.sha256 de la API de Hugging Face."""
    url = (url if url is not None else MODEL_URL).strip().replace("/blob/", "/resolve/")
    if not url:
        raise RuntimeError("Falta el link del modelo: pon MODEL_URL (y MTP_URL/MMPROJ_URL si tocan).")
    repo, nombre = parsear_url(url)
    api_size, api_sha = hf_api_meta(repo, nombre)

    sha = (sha_from_repo(repo, nombre) or "") if repo else ""
    origen = "SHA256SUMS del repo"
    if not sha:
        sha, origen = api_sha or "", "lfs.sha256 de la API de Hugging Face"
    if sha:
        say(f"  SHA256 de {Path(nombre).name} ({origen}): {sha[:16]}…")
    else:
        say(f"  {Path(nombre).name}: ni SHA256SUMS ni la API publican hash; "
            "solo se comprobará el tamaño descargado.")

    return url, sha, (http_head_size(url, HF_TOKEN) or api_size)


def descargar_gguf(etiqueta, url, destino):
    """Descarga (o reutiliza) un .gguf y lo verifica por SHA256 o, si no hay hash, por tamaño.

    Devuelve True si el fichero quedó listo en disco.
    """
    url, sha_esperado, esperado = ensure_src(url)
    ok = destino.parent / (destino.name + ".ok")

    if destino.exists() and ok.exists():
        say(f"      {etiqueta}: ya descargado y verificado.")
        return True
    if destino.exists() and esperado and destino.stat().st_size == esperado:
        say(f"      {etiqueta}: ya completo en disco; salto la descarga.")
    else:
        say(f"      {etiqueta}: descargando {destino.name}"
            + (f" ({esperado / 1e9:.1f} GB)" if esperado else "")
            + ", se reanuda si se corta...")
        cmd = ["curl", "-L", "-f", "-sS", "-C", "-", "--retry", "5", "--retry-delay", "5",
               "-o", str(destino), url]
        if HF_TOKEN:
            cmd[1:1] = ["-H", f"Authorization: Bearer {HF_TOKEN}"]
        p = subprocess.Popen(cmd)
        while p.poll() is None:
            time.sleep(20)
            if destino.exists():
                b = destino.stat().st_size
                pct = f" ({100 * b / esperado:.0f}%)" if esperado else ""
                say(f"        {b / 1e9:.1f} GB{pct}")
        if p.returncode not in (0, 33):   # 33 = "ya estaba completo" (range no satisfecho)
            raise RuntimeError(f"curl falló con código {p.returncode} al bajar {destino.name}")

    if sha_esperado:
        say(f"      {etiqueta}: verificando SHA256...")
        real = sha256(destino)
        if real != sha_esperado:
            destino.unlink(missing_ok=True)
            ok.unlink(missing_ok=True)
            raise RuntimeError(f"SHA256 de {destino.name} no coincide (esperado "
                               f"{sha_esperado[:16]}…, obtenido {real[:16]}…). "
                               "Descartado; vuelve a ejecutar.")
    ok.write_text("ok")
    return True


def ensure_model():
    """El modelo y, solo si están activados, sus ficheros extra (MTP, MMPROJ)."""
    say(f"[2/4] Modelo: {MODEL.name}")
    descargar_gguf("Modelo", MODEL_URL, MODEL)
    for etiqueta, url, destino in extras():
        descargar_gguf(etiqueta, url, destino)


# ---------------------- llama.cpp: binario oficial ----------------------
def latest_tag():
    if LLAMA_RELEASE:
        return LLAMA_RELEASE
    try:
        with http_get("https://github.com/ggml-org/llama.cpp/releases/latest", 30) as r:
            m = re.search(r"/tag/(b\d+)", r.geturl())
            if m:
                return m.group(1)
    except Exception as e:
        say(f"  (redirección de /releases/latest falló: {type(e).__name__})")
    try:
        with http_get("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest", 30, api=True) as r:
            tag = json.load(r).get("tag_name", "")
            if re.fullmatch(r"b\d+", tag):
                return tag
    except Exception as e:
        say(f"  (API de GitHub falló: {type(e).__name__})")
    return None


def driver_cuda():
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", sh(["nvidia-smi"]).stdout)
    return (int(m.group(1)), int(m.group(2))) if m else None


def cuda_versions():
    dc = driver_cuda()
    say(f"  Driver soporta CUDA: {dc[0]}.{dc[1]}" if dc
        else "  No se pudo leer la versión de CUDA del driver.")
    return CUDA_BUILDS["13"] if (dc and dc >= (13, 4)) else CUDA_BUILDS["12"]


def safe_extract(archive, dest):
    with tarfile.open(archive, "r:gz") as tf:
        for flt in ("data", "tar", None):
            try:
                if flt is None or not hasattr(tarfile, "data_filter"):
                    tf.extractall(dest)
                else:
                    tf.extractall(dest, filter=flt)
                return
            except Exception as e:
                if flt is None:
                    raise
                say(f"  (extracción con filtro '{flt}' falló: {e}; reintentando)")


def fetch_asset(url, name):
    dest = WORK_DIR / name
    try:
        with http_get(url, 600) as r, open(dest, "wb") as o:
            shutil.copyfileobj(r, o)
    except urllib.error.HTTPError as e:
        dest.unlink(missing_ok=True)
        if e.code == 404:
            return None
        raise
    return dest


def find_server():
    if PRE_DIR.exists():
        for p in PRE_DIR.rglob("llama-server"):
            if p.is_file():
                return p
    return None


def check_server(srv):
    for f in srv.parent.glob("*"):
        if f.is_file():
            f.chmod(f.stat().st_mode | 0o111)
    LIB_DIRS[:] = sorted({str(srv.parent)} | {str(p.parent) for p in PRE_DIR.rglob("*.so*")})
    r = sh([str(srv), "--version"], env=runtime_env())
    out = (r.stdout + r.stderr).strip()
    if r.returncode != 0:
        say("  No se ejecuta:\n  " + out[-500:].replace("\n", "\n  "))
        return False
    say("  OK: " + (out.splitlines()[-1] if out else "llama-server"))
    return True


def flag_supported(srv, flag):
    """Comprueba en --help si esta versión conoce el flag (evita romper con builds viejos)."""
    global _help
    if _help is None:
        r = sh([str(srv), "--help"], env=runtime_env())
        _help = r.stdout + r.stderr
    return flag in _help


def ensure_llama():
    srv = find_server()
    if srv and check_server(srv):
        say("[1/4] Binario de llama.cpp ya descargado.")
        return srv

    say("[1/4] Descargando llama.cpp oficial (sin compilar)...")
    versions = cuda_versions()

    tag = latest_tag()
    n0 = int((tag or FALLBACK_TAG)[1:])
    say(f"  Último release: {tag or 'desconocido (usando ' + FALLBACK_TAG + ')'}")
    tags = [f"b{n0 - i}" for i in range(TAGS_ATRAS)]
    if not tag and FALLBACK_TAG not in tags:
        tags.insert(0, FALLBACK_TAG)

    base = "https://github.com/ggml-org/llama.cpp/releases/download"
    for t in tags:
        for ver in versions:
            main_name = f"llama-{t}-bin-ubuntu-cuda-{ver}-x64.tar.gz"
            rt_name = f"cudart-llama-{t}-bin-ubuntu-cuda-{ver}-x64.tar.gz"
            main = fetch_asset(f"{base}/{t}/{main_name}", main_name)
            if not main:
                continue
            say(f"  Descargado {main_name}")
            shutil.rmtree(PRE_DIR, ignore_errors=True)
            PRE_DIR.mkdir(parents=True, exist_ok=True)
            safe_extract(main, PRE_DIR)
            main.unlink(missing_ok=True)
            rt = fetch_asset(f"{base}/{t}/{rt_name}", rt_name)
            if rt:
                say(f"  Descargado {rt_name}")
                safe_extract(rt, PRE_DIR)
                rt.unlink(missing_ok=True)
            srv = find_server()
            if srv and check_server(srv):
                say(f"  Usando release {t} con CUDA {ver}")
                return srv
            say(f"  El build {t} / CUDA {ver} no sirve en esta máquina; probando otro...")
    raise RuntimeError("No se pudo obtener un binario oficial de llama.cpp que funcione aquí. "
                       "Revisa los mensajes anteriores (acceso a github.com, versión de driver CUDA).")


# ---------------------- Servidor ----------------------
def server_alive():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
            return r.status == 200
    except urllib.error.HTTPError as e:
        return e.code in (401, 403)
    except Exception:
        return False


def _quitar_spec(args, quitar_ngram):
    """Copia de args sin nada de --spec*, o sin los --spec-ngram-mod* dejando el resto.

    Un token que empieza por -- es un flag; el siguiente, si no empieza por --, es su valor.
    """
    salida, i = [], 0
    while i < len(args):
        t = args[i]
        valor = args[i + 1] if (i + 1 < len(args) and not args[i + 1].startswith("--")) else None
        paso = 2 if valor else 1
        if not t.startswith("--spec"):
            salida += args[i:i + paso]
        elif quitar_ngram and not t.startswith("--spec-ngram-mod"):
            if t == "--spec-type" and valor:
                tipos = [x for x in valor.split(",") if x != "ngram-mod"]
                if tipos:
                    salida += [t, ",".join(tipos)]
            else:
                salida += args[i:i + paso]
        i += paso
    return salida


def bloques_de_args():
    """ARGS_BASE (ya con {threads} sustituido) y sus variantes, por si la primera no arranca.

    Las variantes se generan solas, no hay que escribirlas:
        1) tal cual                        (spec completo: draft-mtp + ngram)
        2) sin los --spec-ngram-mod*        (solo draft-mtp)
        3) sin ningún --spec*               (sin decodificación especulativa)
    Si en ARGS_BASE no pones ningún --spec*, hay un único intento.
    """
    base = shlex.split(ARGS_BASE.replace("^", " ").replace("{threads}", str(THREADS)))
    if "MTP" in EXTRAS_NOMBRES:
        # El MTP va como modelo draft del spec, dentro del propio bloque, para que las
        # variantes de abajo lo quiten igual que el resto de --spec*.
        base += ["--spec-draft-model", str(MTP_LOCAL)]
    intentos = [base]
    if any(t.startswith("--spec") for t in base):
        for variante in (_quitar_spec(base, True), _quitar_spec(base, False)):
            if variante and variante != base and variante not in intentos:
                intentos.append(variante)
    return intentos


def descripcion_spec(args):
    if "--spec-type" not in args:
        return "sin decodificación especulativa"
    tipo = f"spec {arg_valor(args, '--spec-type')}"
    if any(f in args for f in ("--spec-draft-model", "--model-draft", "-md")):
        tipo += " + MTP como draft"
    return tipo


def puerto_configurado():
    """Si pusiste --port en ARGS_BASE, ese manda sobre PORT."""
    for args in bloques_de_args():
        if "--port" in args:
            i = args.index("--port")
            if i + 1 < len(args) and args[i + 1].isdigit():
                return int(args[i + 1])
    return PORT


def usar_api_key(modo):
    """¿Se arranca llama-server con --api-key?

    API_KEY_ON = "" -> automático: con key en modo "api", sin key en modo "webui"
    (para que la página web entre directa, sin pegar nada).
    """
    v = API_KEY_ON
    if isinstance(v, str) and not v.strip():
        return modo == "api"
    return bool(v)


def arg_valor(args, flag, default=None):
    if flag in args:
        i = args.index(flag)
        if i + 1 < len(args):
            return args[i + 1]
    return default


def build_cmd(srv, bloque, modo):
    args = list(bloque)
    if "--spec-draft-model" in args and not flag_supported(srv, "--spec-draft-model"):
        # binario antiguo: el mismo draft con el alias clásico
        args = ["-md" if t == "--spec-draft-model" else t for t in args]
    if "--model" not in args:
        args += ["--model", str(MODEL)]
    if "--alias" not in args:
        args += ["--alias", ALIAS]
    if "MMPROJ" in EXTRAS_NOMBRES and MMPROJ_LOCAL.exists() and "--mmproj" not in args:
        args += ["--mmproj", str(MMPROJ_LOCAL)]
    if "--host" not in args:
        args += ["--host", "127.0.0.1"]
    if "--port" not in args:
        args += ["--port", str(PORT)]
    if usar_api_key(modo) and "--api-key" not in args:
        args += ["--api-key", API_KEY]
    if modo == "api":
        if flag_supported(srv, "--no-webui"):
            args += ["--no-webui"]
        else:
            say("  (esta versión de llama-server no conoce --no-webui; la Web UI queda accesible)")
    return [str(srv)] + args


def start_server(srv, modo):
    global _proc
    if server_alive():
        say("llama-server ya responde.")
        return True
    # Si se interrumpió la celda mientras el modelo aún cargaba, reutilizar ese proceso.
    if _proc is not None and _proc.poll() is None:
        say("llama-server sigue cargando; esperando al proceso existente...")
        t0 = time.time()
        while time.time() - t0 < 900:
            time.sleep(5)
            if server_alive():
                say("llama-server listo.")
                return True
            if _proc.poll() is not None:
                break
    bloques = bloques_de_args()
    for i, bloque in enumerate(bloques, 1):
        cmd = build_cmd(srv, bloque, modo)
        say(f"[3/4] Iniciando llama-server "
            f"(intento {i}/{len(bloques)}: {descripcion_spec(bloque)})...")
        with open(LLAMA_LOG, "w", encoding="utf-8") as log:
            _proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True,
                                     env=runtime_env())
        t0 = time.time()
        while time.time() - t0 < 900:
            time.sleep(5)
            if _proc.poll() is not None:
                break
            if server_alive():
                say("llama-server listo.")
                return True
        log_tail = "\n".join(LLAMA_LOG.read_text(errors="replace").splitlines()[-25:])
        say("Falló. Últimas líneas del log:\n" + log_tail)
        if "no kernel image" in log_tail:
            say("Este binario no trae kernels para tu GPU (T4, sm_75). "
                "Prueba a fijar LLAMA_RELEASE a un release más antiguo.")
            return False
        if not usar_api_key(modo):
            say("  (sin --api-key: la Web UI y /v1 quedan abiertos a quien tenga la URL)")
        if _proc.poll() is None:
            _proc.terminate()
    return False


# ---------------------- Cloudflare Quick Tunnel ----------------------
def ensure_cloudflared():
    if CF_BIN.exists():
        return
    say("Descargando cloudflared...")
    arch = {"x86_64": "amd64", "amd64": "amd64",
            "aarch64": "arm64", "arm64": "arm64"}.get(platform.machine().lower())
    if not arch:
        raise RuntimeError("Arquitectura no compatible para cloudflared")
    url = f"https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-{arch}"
    with urllib.request.urlopen(url, timeout=120) as r, open(CF_BIN, "wb") as o:
        shutil.copyfileobj(r, o)
    CF_BIN.chmod(0o755)
    v = sh([str(CF_BIN), "--version"]).stdout.strip()
    if not v:
        CF_BIN.unlink(missing_ok=True)
        raise RuntimeError("cloudflared descargado pero no se ejecuta.")
    say(v)


def tunnel_alive():
    """Comprueba el túnel actual incluso si pertenece a una ejecución anterior del notebook."""
    if _cf is not None and _cf.poll() is None:
        return True
    return bool(_find_managed_pids("cloudflared"))


def _saved_tunnel_url():
    """Recupera la URL guardada o la lee del log existente sin truncarlo."""
    if CF_URL_FILE.exists():
        candidate = CF_URL_FILE.read_text(errors="replace").strip()
        if re.fullmatch(r"https://[a-z0-9-]+\.trycloudflare\.com", candidate):
            return candidate
    log_text = CF_LOG.read_text(errors="replace") if CF_LOG.exists() else ""
    matches = re.findall(r"https://[a-z0-9-]+\.trycloudflare\.com", log_text)
    if matches:
        url = matches[-1]
        CF_URL_FILE.write_text(url)
        return url
    return None


def start_tunnel():
    """Reutiliza el túnel vivo y su URL; solo crea uno nuevo si el anterior ya no existe."""
    global _cf
    if tunnel_alive():
        url = _saved_tunnel_url()
        if url:
            say(f"Reutilizando el túnel de Cloudflare existente: {url}")
            return url
        # El proceso existe, pero no se ha encontrado su URL; esperamos a que el log la publique.
        for _ in range(10):
            time.sleep(1)
            url = _saved_tunnel_url()
            if url:
                say(f"Reutilizando el túnel de Cloudflare existente: {url}")
                return url
        raise RuntimeError("Hay un túnel de Cloudflare activo, pero no se pudo recuperar su URL. "
                           "No se creará otro túnel para evitar cambiar la dirección.")

    cmd = [str(CF_BIN), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{PORT}"]
    if CF_HOST_HEADER:
        cmd += ["--http-host-header", f"localhost:{PORT}"]
    CF_URL_FILE.unlink(missing_ok=True)
    with open(CF_LOG, "w", encoding="utf-8") as log:
        _cf = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 90:
        time.sleep(2)
        text = CF_LOG.read_text(errors="replace") if CF_LOG.exists() else ""
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", text)
        if m:
            url = m.group(0)
            CF_URL_FILE.write_text(url)
            return url
        if _cf.poll() is not None:
            break
    raise RuntimeError("No se obtuvo URL del túnel. Log:\n"
                       + CF_LOG.read_text(errors="replace")[-1500:])


def check_public(url, modo):
    """Comprobación desde fuera: la API siempre; la raíz solo si hay Web UI."""
    try:
        cabeceras = {"Authorization": f"Bearer {API_KEY}"} if usar_api_key(modo) else {}
        req = urllib.request.Request(url + "/v1/models", headers=cabeceras)
        with urllib.request.urlopen(req, timeout=20) as r:
            say(f"Prueba pública /v1/models: HTTP {r.status}")
    except Exception as e:
        say(f"Prueba pública falló ({type(e).__name__}: {e}). "
             "Puede tardar unos segundos; reintenta con curl.")
    if modo == "webui":
        try:
            with urllib.request.urlopen(url + "/", timeout=20) as r:
                say(f"Prueba pública Web UI  /: HTTP {r.status}")
        except Exception as e:
            say(f"Prueba pública de la Web UI falló ({type(e).__name__}: {e}).")


def resumen(url, modo):
    say("")
    say("=" * 70)
    say("SERVIDOR LISTO")
    say(f"Modelo   : {ALIAS}  ({MODEL.name})")
    if extras():
        say("Extra    : " + ", ".join(f"{e} ({d.name})" for e, _, d in extras()))
    if modo == "webui":
        say(f"Web UI   : {url}/")
        if not usar_api_key(modo):
            say("           (entra directa, sin pegar nada; quien tenga la URL puede usarla)")
    else:
        say("Web UI   : desactivada (--no-webui)")
    say(f"Base URL : {url}/v1")
    if usar_api_key(modo):
        say(f"API Key  : {API_KEY}")
        say(f'Prueba   : curl {url}/v1/models -H "Authorization: Bearer {API_KEY}"')
    else:
        say("API Key  : ninguna")
        say(f"Prueba   : curl {url}/v1/models")
    say("=" * 70)


# ---------------------- La pregunta del principio ----------------------
def ask_modo():
    m = (MODO or os.environ.get("LLM_MODO", "")).strip().lower()
    if m in ("api", "a", "1", "solo-api"):
        return "api"
    if m in ("webui", "web", "ui", "w", "2", "pagina", "página"):
        return "webui"

    say("¿Cómo monto llama-server?")
    say("  1) Solo API      -> solo /v1, sin Web UI (OpenWebUI, curl, tus apps)")
    say("  2) API + Web UI  -> además sirve la página web, para probar desde el navegador")
    while True:
        try:
            r = input("Elige 1 o 2 [2]: ").strip().lower()
        except KeyboardInterrupt:
            say("  (interrumpido; uso API + Web UI)")
            return "webui"
        except Exception as e:
            say(f"  (aquí no puedo preguntar: {type(e).__name__}; uso API + Web UI)")
            say('  Si querías solo API, pon MODO = "api" arriba.')
            return "webui"
        if r in ("", "2", "webui", "web", "ui", "w"):
            return "webui"
        if r in ("1", "api", "a", "solo-api"):
            return "api"
        say("  Responde 1 o 2.")


HF_TOKEN = get_secret("HF_TOKEN") or ""
API_KEY = get_secret("LLAMA_API_KEY") or secrets.token_urlsafe(24)


# ---------------------- Principal ----------------------
def ask_shutdown_action():
    """Menú de apagado al pulsar Stop/Interrupt en el notebook."""
    say("\n¿Qué quieres detener?")
    say("  1) Detener llama-server y liberar el modelo; dejar Cloudflare activo y conservar la URL")
    say("  2) Detener llama-server y Cloudflare")
    while True:
        try:
            choice = input("Elige 1 o 2 [1]: ").strip()
        except (EOFError, KeyboardInterrupt):
            say("No se pudo leer la opción; por seguridad se detendrán ambos servicios.")
            return "2"
        if choice in ("", "1"):
            return "1"
        if choice == "2":
            return "2"
        say("Opción no válida. Escribe 1 o 2.")


def main():
    global PORT
    say("=" * 70)
    say(f"Modelo   : {MODEL.name}")
    say(f"Repo     : {MODEL_REPO or '(link directo, sin repo de Hugging Face)'}")
    say("Extra    : " + (", ".join(f"{e} -> {d.name}" for e, _, d in extras())
                         if extras() else "ninguno"))
    libre = free_gb(DATA_DIR)
    say(f"Datos en : {DATA_DIR}" + (f"  ({libre:.0f} GB libres)" if libre else ""))
    say("=" * 70)

    modo = ask_modo()
    PORT = puerto_configurado()
    say(f"Montaje  : {'solo API (sin Web UI)' if modo == 'api' else 'API + Web UI'}  ·  "
        f"puerto {PORT}  ·  {'con API key' if usar_api_key(modo) else 'sin API key'}")
    say("")

    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def _handle_sigterm(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        # Si se interrumpe durante la carga o cuando el servidor ya está listo,
        # se ofrecen las dos opciones de apagado.
        while True:
            try:
                srv = ensure_llama()
                ensure_model()
                if not start_server(srv, modo):
                    raise RuntimeError("llama-server no arrancó con ninguna configuración.")
                say("[4/4] Comprobando/abriendo túnel de Cloudflare...")
                ensure_cloudflared()
                url = start_tunnel()
                check_public(url, modo)
                resumen(url, modo)
                say("Pulsa Stop/Interrupt para mostrar el menú de apagado.\n")
                n = 0

                while True:
                    time.sleep(30)
                    n += 1
                    if not server_alive():
                        say("llama-server no responde; reiniciando...")
                        if not start_server(srv, modo):
                            say("No se pudo reiniciar llama-server; el túnel seguirá activo.")
                    if not tunnel_alive():
                        say("El túnel se cayó; creando un túnel nuevo (la URL cambiará)...")
                        url = start_tunnel()
                        resumen(url, modo)
                    if n % 2 == 0:
                        say(f"[{time.strftime('%H:%M:%S')}] activo - {url}")

            except KeyboardInterrupt:
                action = ask_shutdown_action()
                if action == "1":
                    stop_llama_server()
                    say("\nPuedes cambiar ARGS_BASE y volver a ejecutar la celda.")
                    saved_url = _saved_tunnel_url()
                    if saved_url and tunnel_alive():
                        say(f"La URL de Cloudflare se conserva: {saved_url}")
                    else:
                        say("Cloudflare no estaba activo; en el siguiente arranque se deberá crear un túnel.")
                    return
                if action == "2":
                    cleanup_processes()
                    say("Todos los servicios quedaron detenidos.")
                    return
    except Exception as e:
        say(f"ERROR: {type(e).__name__}: {e}")
        cleanup_processes()
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
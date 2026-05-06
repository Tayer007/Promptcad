# PromptCAD

STL → CadQuery code generation using a trained model based on Qwen2 1.5B.

---

# English

## Running Locally

### Prerequisites

- **Python 3.10** — [Download](https://www.python.org/downloads/release/python-31011/)
- **Git**
- **NVIDIA GPU — Turing architecture or newer** (RTX 20-series and above, 2018+).

> **CPU-only inference:** By default the app requires a CUDA GPU. If you want to run on CPU anyway (very slow — expect several minutes per inference), make the following changes in `inference.py`:
> 1. Change `device_map='cuda'` to `device_map='cpu'` (line 129)
> 2. Change `.cuda()` to `.cpu()` on both lines inside the `chamfer_distance` function (lines 101–102)

### 1. Clone the repository

```bash
git clone https://github.com/Tayer007/Promptcad.git
cd Promptcad
```

### 2. Run the setup script

**Windows:**
```
setup.bat
```

**Mac / Linux:**
```bash
bash setup.sh
```

This creates a `cad_env` virtual environment and installs all dependencies (PyTorch CPU + project requirements).

> For GPU support, after setup replace the PyTorch installation inside `cad_env` with the CUDA version matching your driver from [pytorch.org](https://pytorch.org/get-started/locally/).

### 3. Run the app

**Windows:**
```
cad_env\Scripts\activate
set HF_REPO=fourat25/mesh-to-cadquery-qwen
python app.py
```

**Mac / Linux:**
```bash
source cad_env/bin/activate
export HF_REPO=fourat25/mesh-to-cadquery-qwen
python app.py
```

Open `http://localhost:5000`

> The model (~3 GB) is downloaded automatically from HuggingFace on the first inference run. This only happens once.

---

## Running on HPC (Singularity)

### Prerequisites

- Singularity / Apptainer available on the cluster
- A GPU compute node **without NVSwitch/NVLink fabric** (see warning below)
- Internet access on the login node (not required on compute nodes)

> **NVSwitch / NVLink warning:** Nodes with NVSwitch or NVLink fabric (typically nodes with 8+ GPUs connected via fabric, e.g. DGX/HGX systems) require the NVIDIA Fabric Manager to be accessible inside the container. Singularity cannot access the Fabric Manager by default, causing CUDA to fail with `Error 802: system not yet initialized`. Always request a node without NVSwitch — typically nodes with 4 or fewer GPUs on standard PCIe.

### 1. Pull the container (login node)

```bash
singularity pull promptcad.sif docker://fourat98/promptcad:latest
```

### 2. Download the model (login node)

The compute node has no internet access, so download the model first on the login node:

```bash
bash download_model.sh
```

This saves the model to `$HOME/.cache/huggingface`. Only needs to be done once.

### 3. Allocate a GPU compute node

Request an interactive session with GPU resources:

```bash
salloc -N1 -w <compute-node> --gres=gpu:1 --mem=32G -c 32 --time=4:00:00
srun --pty bash
```

> Replace `<compute-node>` with your target node name. Adjust `--mem` and `-c` (cores) to what your cluster permits.

### 4. Set up the SSH tunnel (from your local machine)

**Only after the SLURM job is running** — SSH to compute nodes is blocked without an active job. Open a new terminal on your local machine:

```bash
ssh -L 5000:localhost:5000 -J <username>@<hpc-login> <username>@<compute-node>
```

> This uses a jump host (`-J`) to tunnel directly to the compute node, bypassing login node firewall restrictions.
>
> **Important:** Always include the `-L` flag — without it the port is not forwarded and the browser cannot reach the app.
>
> Port `5000` is the port Flask listens on inside the container.
>
> - **If port 5000 is already in use on your local machine:** change only the left-hand port in the tunnel — Flask inside the container stays on 5000 and no code change is needed. Example: `ssh -L 8080:localhost:5000 -J ...` → open `http://localhost:8080`.
> - **If port 5000 is blocked on the HPC compute node:** you need to change it in both the tunnel and the code. Edit the last line of `app.py` (`port=5000`) to your chosen port, rebuild the Docker image, re-pull the `.sif`, and update the tunnel accordingly.

### 5. Run the app (on the compute node)

In the `srun --pty bash` session, run the app:

```bash
bash run_app.sh
```

> **Note:** Environment variables do not persist across sessions. If `run_app.sh` is not available, set the variables manually and run singularity directly — see the script contents for reference.

Once Flask prints `Running on http://0.0.0.0:5000`, open `http://localhost:5000` in your browser.

---
---

# Deutsch

## Lokale Ausführung

### Voraussetzungen

- **Python 3.10** — [Download](https://www.python.org/downloads/release/python-31011/)
- **Git**
- **NVIDIA GPU — Turing-Architektur oder neuer** (RTX 20-Serie und neuer, ab 2018).

> **CPU-only Inferenz:** Standardmäßig erfordert die App eine CUDA-fähige GPU. Wer dennoch auf der CPU ausführen möchte (sehr langsam — mehrere Minuten pro Inferenz), muss folgende Änderungen in `inference.py` vornehmen:
> 1. `device_map='cuda'` zu `device_map='cpu'` ändern (Zeile 129)
> 2. `.cuda()` zu `.cpu()` in beiden Zeilen der Funktion `chamfer_Distance` ändern (Zeilen 101–102)

### 1. Repository klonen

```bash
git clone https://github.com/Tayer007/Promptcad.git
cd Promptcad
```

### 2. Setup-Skript ausführen

**Windows:**
```
setup.bat
```

**Mac / Linux:**
```bash
bash setup.sh
```

Dieses Skript erstellt eine virtuelle Umgebung (`cad_env`) und installiert alle Abhängigkeiten (PyTorch CPU + Projektanforderungen).

> Für GPU-Unterstützung kann PyTorch nach der Installation durch die passende CUDA-Version ersetzt werden. Die entsprechende Version ist unter [pytorch.org](https://pytorch.org/get-started/locally/) verfügbar.

### 3. App starten

**Windows:**
```
cad_env\Scripts\activate
set HF_REPO=fourat25/mesh-to-cadquery-qwen
python app.py
```

**Mac / Linux:**
```bash
source cad_env/bin/activate
export HF_REPO=fourat25/mesh-to-cadquery-qwen
python app.py
```

Anschließend im Browser öffnen: `http://localhost:5000`

> Das Modell (~3 GB) wird beim ersten Aufruf automatisch von HuggingFace heruntergeladen. Dies geschieht nur einmalig.

---

## Ausführung auf einem HPC-Cluster (Singularity)

### Voraussetzungen

- Singularity / Apptainer auf dem Cluster verfügbar
- GPU-Rechenknoten **ohne NVSwitch/NVLink-Fabric** (siehe Hinweis unten)
- Internetzugang auf dem Login-Knoten (auf dem Rechenknoten nicht erforderlich)

> **NVSwitch / NVLink Hinweis:** Knoten mit NVSwitch oder NVLink-Fabric (typischerweise Knoten mit 8 oder mehr GPUs, z. B. DGX/HGX-Systeme) erfordern den NVIDIA Fabric Manager im Container. Singularity kann standardmäßig nicht auf den Fabric Manager zugreifen, was zu einem CUDA-Fehler `Error 802: system not yet initialized` führt. Es sollte immer ein Knoten ohne NVSwitch verwendet werden — typischerweise Knoten mit 4 oder weniger GPUs über Standard-PCIe.

### 1. Container herunterladen (Login-Knoten)

```bash
singularity pull promptcad.sif docker://fourat98/promptcad:latest
```

### 2. Modell herunterladen (Login-Knoten)

Da der Rechenknoten keinen Internetzugang hat, muss das Modell zunächst auf dem Login-Knoten heruntergeladen werden:

```bash
bash download_model.sh
```

Das Modell wird in `$HOME/.cache/huggingface` gespeichert. Dieser Schritt muss nur einmalig durchgeführt werden.

### 3. GPU-Rechenknoten reservieren

Interaktive Sitzung mit GPU-Ressourcen anfordern:

```bash
salloc -N1 -w <rechenknoten> --gres=gpu:1 --mem=32G -c 32 --time=4:00:00
srun --pty bash
```

> `<rechenknoten>` durch den gewünschten Knotennamen ersetzen. `--mem` und `-c` (Kerne) entsprechend den Cluster-Limits anpassen.

### 4. SSH-Tunnel einrichten (vom lokalen Rechner)

**Erst nachdem der SLURM-Job läuft** — SSH auf Rechenknoten ist ohne aktiven Job gesperrt. Neues Terminal auf dem lokalen Rechner öffnen:

```bash
ssh -L 5000:localhost:5000 -J <benutzername>@<hpc-login> <benutzername>@<rechenknoten>
```

> Dieser Befehl verwendet einen Jump-Host (`-J`), um direkt auf den Rechenknoten zu tunneln und dabei Firewall-Einschränkungen des Login-Knotens zu umgehen.
>
> **Wichtig:** Das `-L`-Flag darf nicht vergessen werden — ohne es wird der Port nicht weitergeleitet.
>
> Port `5000` ist der Port, auf dem Flask im Container lauscht.
>
> - **Falls Port 5000 auf dem lokalen Rechner bereits belegt ist:** Nur die linke Seite des Tunnels ändern — keine Code-Änderung notwendig. Beispiel: `ssh -L 8080:localhost:5000 -J ...` → Browser öffnen unter `http://localhost:8080`.
> - **Falls Port 5000 auf dem Rechenknoten blockiert ist:** Änderungen sowohl im Tunnel als auch im Code erforderlich. Die letzte Zeile in `app.py` (`port=5000`) anpassen, Docker-Image neu bauen, `.sif` neu herunterladen und Tunnel entsprechend aktualisieren.

### 5. App starten (auf dem Rechenknoten)

In der `srun --pty bash`-Sitzung die App starten:

```bash
bash run_app.sh
```

> **Hinweis:** Umgebungsvariablen werden nicht sitzungsübergreifend gespeichert. Falls `run_app.sh` nicht verfügbar ist, können die Variablen manuell gesetzt und Singularity direkt gestartet werden — der Skriptinhalt dient als Referenz.

Sobald Flask `Running on http://0.0.0.0:5000` ausgibt, im Browser öffnen: `http://localhost:5000`

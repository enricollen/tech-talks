# Presentazione Model Context Protocol (MCP)

Questa repository contiene materiali dimostrativi completi per una presentazione sul **Model Context Protocol (MCP)**, un protocollo standardizzato per connettere strumenti AI, risorse e prompt a modelli di linguaggio.

## 📋 Contenuti

### Struttura del Progetto

```
2026/GDG Firenze - Model Context Protocol/
├── mcp_server/
│   ├── mcp_server.py      # Server MCP personalizzato per ricette
│   └── recipes.json       # Database ricette italiane
├── mcp_client/
│   ├── mcp_client.py      # Script esempi client MCP e agenti
│   └── mcp_client_ui.py   # Interfaccia Gradio multi-agente
├── mcp_presentation.ipynb # Notebook Jupyter per presentazione
├── requirements.txt       # Dipendenze Python
├── .env                   # Variabili d'ambiente (da creare)
└── README.md
```

## 🚀 Setup Iniziale

### 1. Installazione Dipendenze

```bash
pip install -r requirements.txt
```

### 2. Configurazione Variabili d'Ambiente

Crea un file `.env` nella root del progetto:

```env
OPENAI_API_KEY=sk-...
DOCKER_GATEWAY_TOKEN=...
HUGGINGFACE_API_TOKEN=hf_...
```

**Come ottenere i token:**

- `OPENAI_API_KEY`: da [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- `DOCKER_GATEWAY_TOKEN`: viene generato automaticamente quando avvii il Docker MCP Gateway (vedi sotto)
- `HUGGINGFACE_API_TOKEN`: da [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (opzionale)

### 3. Avvio Docker MCP Gateway

Il Docker MCP Gateway permette di connettere più server MCP tramite un singolo endpoint:

```bash
docker mcp gateway run --port 8080 --transport streaming
```

**Nota:** Copia il Bearer token dall'output del comando e aggiungilo al file `.env` come `DOCKER_GATEWAY_TOKEN`.

### 4. Avvio Server MCP Locale

In un terminale separato, avvia il server MCP personalizzato per le ricette:

```bash
mcp run mcp_server/mcp_server.py --transport=streamable-http
```

Il server sarà disponibile su `http://localhost:8000/mcp`.

## 📚 Struttura della Presentazione

### 1. Docker MCP Gateway (`mcp_presentation.ipynb` - Sezione 1)

Dimostra come:

- Connettersi al Docker MCP Gateway
- Autenticarsi con Bearer token
- Elencare strumenti disponibili da più server MCP
- Gestire errori di connessione

**Concetti chiave:**

- Gateway come punto di accesso unificato
- Autenticazione con header HTTP
- Aggregazione di strumenti da fonti multiple

### 2. Server MCP Personalizzato (`mcp_server/mcp_server.py`)

Implementazione di un server MCP usando FastMCP che espone:

**Tools (Strumenti):**

- `list_recipes()` - elenca ricette italiane disponibili
- `get_recipe_instructions(recipe_name)` - ottiene istruzioni dettagliate

**Resources (Risorse):**

- `guide://usage` - guida all'uso del server

**Prompts:**

- `suggest_recipe(occasion, dietary_preference)` - genera prompt per suggerimenti personalizzati

**Avvio:**

```bash
mcp run mcp_server/mcp_server.py --transport=streamable-http
```

### 3. MCP Inspector

Strumento di debug web-based per server MCP.

**Installazione:**

```bash
npm install -g @modelcontextprotocol/inspector
```

**Avvio:**

```bash
mcp-inspector
```

Apri il browser su `http://localhost:5173` e connettiti al tuo server MCP per:

- Ispezionare strumenti disponibili
- Testare chiamate a strumenti
- Visualizzare risorse e prompt
- Debug in tempo reale

### 4. Client MCP (`mcp_client/mcp_client.py`)

Script Python completo con esempi di:

#### a) Connessione a Docker Gateway

```python
docker_client = BasicMCPClient(
    "http://localhost:8080/mcp",
    headers={
        "Authorization": f"Bearer {DOCKER_GATEWAY_TOKEN}",
        "Accept": "application/json, text/event-stream"
    }
)
```

#### b) Connessione a Server Locale

```python
local_client = BasicMCPClient("http://localhost:8000/mcp")
```

#### c) Connessione a FastMCP Cloud

```python
remote_client = BasicMCPClient("https://unnecessary-crimson-wildebeest.fastmcp.app/mcp")
```

#### d) Connessione a Hugging Face Space

```python
gradio_client = BasicMCPClient("https://hysts-mcp-flux-1-schnell.hf.space/gradio_api/mcp/")
```

**Esecuzione:**

```bash
python mcp_client/mcp_client.py
```

### 5. Agente Ibrido (`mcp_client/mcp_client.py` - `hybrid_mcp_agent()`)

Dimostra un agente che combina strumenti da **4 fonti MCP diverse**:

- Server locale (ricette)
- FastMCP remoto (meteo)
- Docker Gateway (trascrizione video YouTube)
- Hugging Face Space (generazione immagini)

**Problema dimostrato:** Tool Overload

- Troppi strumenti confondono l'agente
- Performance degradate
- Difficoltà nel routing delle richieste

### 6. Sistema Multi-Agente con Triage (`mcp_client/mcp_client.py` - `multiagent_triage_example()`)

**Soluzione Best Practice** al problema del Tool Overload:

**Architettura:**

```
TriageAgent (router)
    ├── RecipeAgent (strumenti ricette locali)
    ├── WeatherAgent (strumenti meteo FastMCP)
    ├── VideoAgent (strumenti trascrizione Docker Gateway)
    └── ImageAgent (strumenti generazione immagini Hugging Face)
```

**Vantaggi:**

- Ogni agente specializzato ha solo gli strumenti necessari
- Il TriageAgent instrada intelligentemente le richieste
- Scalabile: facile aggiungere nuovi agenti specializzati
- Performance migliorate: meno strumenti per agente

### 7. Interfaccia Gradio (`mcp_client/mcp_client_ui.py`)

Interfaccia web interattiva per il sistema multi-agente.

**Features:**

- Chat UI moderna e intuitiva
- Supporto per streaming delle risposte
- Visualizzazione tool calls in tempo reale
- Gestione errori con messaggi informativi
- Supporto per tutti e 4 gli agenti specializzati

**Avvio:**

```bash
python mcp_client/mcp_client_ui.py
```

Apri il browser su `http://localhost:7860`.

## 🎯 Flusso della Presentazione

### Parte 1: Fondamenti MCP

1. Introduzione al protocollo MCP
2. Architettura: Tools, Resources, Prompts
3. Docker MCP Gateway come aggregatore

### Parte 2: Implementazione

1. Creazione server MCP personalizzato (`mcp_server/mcp_server.py`)
2. Debug con MCP Inspector
3. Implementazione client MCP
4. Connessione a fonti multiple

### Parte 3: Pattern Avanzati

1. Agente ibrido con strumenti multipli
2. Problema del Tool Overload
3. Soluzione: Sistema Multi-Agente con Triage
4. Demo live con interfaccia Gradio (`mcp_client/mcp_client_ui.py`)

## 🔧 Esempi di Utilizzo

### Eseguire il Notebook di Presentazione

```bash
jupyter notebook mcp_presentation.ipynb
```

**Nota:** Esegui il notebook dalla root del progetto, non dalle sottocartelle.

Esegui le celle in sequenza per vedere:

- Connessione a Docker Gateway
- Test strumenti da server locale
- Agente ibrido in azione
- Sistema multi-agente con triage

### Eseguire Script Standalone

**Importante:** Esegui sempre dalla root del progetto per garantire che `.env` venga caricato correttamente.

```bash
# test docker gateway
python mcp_client/mcp_client.py  # decommentare docker_mcp_servers_gateway()

# test server locale
python mcp_client/mcp_client.py  # decommentare local_mcp_server()

# test agente ibrido
python mcp_client/mcp_client.py  # decommentare hybrid_mcp_agent()

# test sistema multi-agente (default)
python mcp_client/mcp_client.py
```

### Avviare Interfaccia Web

```bash
python mcp_client/mcp_client_ui.py
```

Prova query come:

- "come si fa il tiramisù?"
- "che tempo fa a milano?"
- "trascrivi questo video: [https://www.youtube.com/watch?v=](https://www.youtube.com/watch?v=)..."
- "genera un'immagine di un tramonto sulle montagne"

## 🏗️ Architettura Tecnica

### Stack Tecnologico

- **LlamaIndex**: Framework per agenti LLM e orchestrazione
- **FastMCP**: Framework per creare server MCP
- **Gradio**: Interfaccia web per demo interattive
- **Docker**: Containerizzazione e MCP Gateway

### Componenti MCP

**Tools (Strumenti):**

- Funzioni che l'agente può chiamare
- Schema JSON per parametri
- Esecuzione sincrona o asincrona

**Resources (Risorse):**

- Contenuti statici o dinamici
- URI-based access
- Supporto per diversi MIME types

**Prompts:**

- Template riutilizzabili
- Parametrizzabili
- Context-aware

## 🔗 Server MCP Utilizzati

### 1. Server Locale (Ricette)

- **URL:** `http://localhost:8000/mcp`
- **Tools:** `list_recipes`, `get_recipe_instructions`
- **Tipo:** Custom FastMCP server
- **Uso:** Dimostra creazione server personalizzato

### 2. FastMCP Cloud (Meteo)

- **URL:** `https://unnecessary-crimson-wildebeest.fastmcp.app/mcp`
- **Tools:** `get_weather`
- **Tipo:** Server remoto pubblico
- **Uso:** Dimostra integrazione servizi cloud

### 3. Docker Gateway (Video Transcription)

- **URL:** `http://localhost:8080/mcp`
- **Tools:** `get_transcript` (YouTube)
- **Tipo:** Aggregatore di server MCP in Docker
- **Uso:** Dimostra orchestrazione enterprise

### 4. Hugging Face Space (Image Generation)

- **URL:** `https://hysts-mcp-flux-1-schnell.hf.space/gradio_api/mcp/`
- **Tools:** `FLUX_1_schnell_infer`, `FLUX_1_schnell_get_seed`
- **Tipo:** Gradio app su Hugging Face
- **Uso:** Dimostra integrazione AI models

## 📝 Modifiche e Personalizzazione

### Aggiungere un Nuovo Agente Specializzato

1. Crea o connetti a un nuovo server MCP
2. Carica gli strumenti specifici
3. Crea un nuovo `ReActAgent` con system prompt appropriato
4. Aggiungi l'agente al `can_handoff_to` del TriageAgent
5. Aggiorna il system prompt del TriageAgent con la descrizione
6. Aggiungi l'agente all'`AgentWorkflow`

### Aggiungere Nuove Ricette

Modifica `mcp_server/recipes.json` seguendo il formato esistente:

```json
{
  "Nome Ricetta": {
    "ingredients": ["ingrediente 1", "ingrediente 2"],
    "instructions": "Passaggi dettagliati..."
  }
}
```

Riavvia il server MCP per applicare le modifiche.

## 📚 Risorse Aggiuntive

- **MCP Specification:** [https://spec.modelcontextprotocol.io/](https://spec.modelcontextprotocol.io/)
- **FastMCP Documentation:** [https://github.com/jlowin/fastmcp](https://github.com/jlowin/fastmcp)
- **LlamaIndex MCP Tools:** [https://developers.llamaindex.ai/python/examples/tools/mcp/](https://developers.llamaindex.ai/python/examples/tools/mcp/)
- **Docker MCP Gateway:** [https://github.com/docker/mcp-gateway](https://github.com/docker/mcp-gateway)
- **MCP Inspector:** [https://github.com/modelcontextprotocol/inspector](https://github.com/modelcontextprotocol/inspector)


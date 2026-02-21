# Teams Bot Setup Guide (All Free)

## Prerequisites (Create These Accounts)

### 1. Azure Account (Free — for Bot Registration ONLY)
1. Go to [portal.azure.com](https://portal.azure.com)
2. Click **Start free** → create account with your email
3. You get **$200 free credit + 12 months of free services**
4. The Bot registration itself is **always free** (no credit needed)

### 2. Groq API Key (Free)
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up with Google/GitHub
3. Go to **API Keys** → **Create API Key**
4. Copy the key → paste in `.env` as `GROQ_API_KEY=gsk_...`

### 3. ngrok Account (Free)
1. Go to [ngrok.com](https://ngrok.com)
2. Sign up (free plan is enough)
3. Go to **Your Authtoken** page
4. Copy the token → paste in `.env` as `NGROK_AUTHTOKEN=...`

---

## Step 1: Register Azure Bot

1. Go to [portal.azure.com](https://portal.azure.com)
2. Search for **"Azure Bot"** → click **Create**
3. Fill in:
   - **Bot handle**: `meeting-agent-bot`
   - **Subscription**: Free Trial
   - **Resource group**: Create new → `meeting-agent-rg`
   - **Pricing**: **F0 (Free)**
   - **Microsoft App ID**: **Create new** (multi-tenant)
4. Click **Create** → wait for deployment
5. Go to the resource → **Configuration**:
   - Copy **Microsoft App ID** → paste in `.env` as `MICROSOFT_APP_ID=`
   - Click **Manage Password** → **New client secret** → copy value → `.env` as `MICROSOFT_APP_PASSWORD=`
6. Go to **Channels** → **Microsoft Teams** → Enable

## Step 2: Configure Bot Messaging Endpoint

1. Run `docker-compose up` (this starts ngrok)
2. Open [localhost:4040](http://localhost:4040) → copy the **ngrok HTTPS URL**
3. Back in Azure Portal → **Azure Bot** → **Configuration**:
   - Set **Messaging endpoint** to: `https://YOUR-NGROK-URL.ngrok-free.app/api/messages`
   - Click **Apply**

## Step 3: Create Teams App Package

1. Edit `manifest/manifest.json`:
   - Replace `{{MICROSOFT_APP_ID}}` with your actual App ID
2. Create placeholder icons (or use any 192x192 and 32x32 PNGs):
   ```
   manifest/color.png    (192x192)
   manifest/outline.png  (32x32)
   ```
3. Zip the manifest folder:
   ```powershell
   Compress-Archive -Path manifest\* -DestinationPath meeting-agent-app.zip
   ```

## Step 4: Install in Teams

1. Open **Microsoft Teams** (desktop or web)
2. Go to **Apps** → **Manage your apps** → **Upload an app**
3. Choose **Upload a custom app** → select `meeting-agent-app.zip`
4. Click **Add** → the bot appears in your Teams sidebar

## Step 5: Test Auto-Join

1. Start a Teams meeting (or schedule one)
2. Once the meeting starts, the bot should automatically join
3. Check the Docker logs: `docker-compose logs -f app`
4. You should see:
   ```
   Teams meeting started: <meeting-id>
   Pipeline initialised for meeting <meeting-id>
   Recognised [en] (85%): Hello everyone...
   ```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Bot doesn't appear in Teams | Check App ID matches in manifest.json and Azure Portal |
| Bot doesn't join meetings | Ensure ngrok is running and messaging endpoint is set |
| No transcription | Check Docker logs — Whisper model may still be downloading |
| ngrok tunnel expired | Free ngrok URLs change on restart — update Azure Bot endpoint |

## Architecture (After Migration)

```
Teams Meeting
    │
    ▼
ngrok tunnel → FastAPI (:8000) → Bot Framework
    │
    ▼
AudioStreamHandler
    │
    ├──→ WhisperRealtimeTranscriber (local, free)
    │        → real-time segments → database
    │
    └──→ ChunkManager → MinIO → WhisperWorker
                                    → batch transcription → database
    │
    ▼
Groq API (Llama 3, free) → Meeting Analysis Reports
```

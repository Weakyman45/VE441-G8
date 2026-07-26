package cn.edu.sjtu.voiceshop

import android.util.Base64
import android.util.Log
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import org.json.JSONArray
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Talker client: connects to the local backend Realtime proxy
 * (default Qwen Omni via `/api/v1/qwen/realtime/ws?session_id=...`)
 * which runs Workers on the Session Engine.
 */
class RealtimeSession(
    private val backendBaseUrl: String,
    private val initialSessionId: String? = null,
    private val listener: Listener
) {
    interface Listener {
        fun onConnectionChanged(connected: Boolean)
        fun onSessionReady(sessionId: String)
        fun onTalkerReady(provider: String, inputAudioSampleRate: Int)
        fun onUserSpeechStarted()
        fun onUserSpeechStopped()
        fun onUserTranscript(text: String, isFinal: Boolean)
        fun onAssistantTextDelta(delta: String)
        fun onAssistantTextDone(full: String)
        fun onAssistantAudio(pcm16: ByteArray)
        fun onAssistantItemStarted(itemId: String)
        fun onStatus(message: String)
        fun onError(message: String)
    }

    private val client = OkHttpClient.Builder()
        .pingInterval(20, TimeUnit.SECONDS)
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .build()

    private var webSocket: WebSocket? = null
    private val open = AtomicBoolean(false)
    private val assistantBuffer = StringBuilder()
    private val truncatedAssistantItemIds = mutableSetOf<String>()
    @Volatile private var currentResponseItemId: String? = null
    @Volatile private var responseInProgress = false
    @Volatile private var ignoreNextAssistantDone = false
    @Volatile var sessionId: String? = null
        private set
    @Volatile var talkerProvider: String = "qwen"
        private set
    @Volatile var inputAudioSampleRate: Int = AudioCapture.QWEN_SAMPLE_RATE
        private set
    @Volatile private var talkerWsPath: String = "/api/v1/realtime/ws"
    @Volatile private var interruptionHandlingEnabled = true

    fun connect() {
        Thread({
            try {
                listener.onStatus("Creating shopping session…")
                val created = createBackendSession()
                sessionId = created.sessionId
                talkerWsPath = created.wsPath
                talkerProvider = created.provider
                inputAudioSampleRate = created.inputAudioSampleRate
                interruptionHandlingEnabled = created.interruptionHandlingEnabled
                listener.onSessionReady(created.sessionId)
                listener.onTalkerReady(created.provider, created.inputAudioSampleRate)
                listener.onStatus("Connecting ${created.providerLabel} Talker…")
                openWebSocketViaBackend(created.sessionId, created.wsPath)
            } catch (error: Exception) {
                Log.e(TAG, "Connect failed via $backendBaseUrl", error)
                listener.onError("Connect failed ($backendBaseUrl): ${error.message}")
                listener.onConnectionChanged(false)
            }
        }, "realtime-connect").start()
    }

    private data class BackendSession(
        val sessionId: String,
        val wsPath: String,
        val provider: String,
        val inputAudioSampleRate: Int,
        val interruptionHandlingEnabled: Boolean
    ) {
        val providerLabel: String
            get() = if (provider == "openai") "GPT Realtime" else "Qwen Omni"
    }

    private fun createBackendSession(): BackendSession {
        val url = URL("$backendBaseUrl/api/v1/session")
        val conn = (url.openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            doOutput = true
            connectTimeout = 8_000
            readTimeout = 12_000
            setRequestProperty("Content-Type", "application/json")
        }
        try {
            val requestBody = if (initialSessionId.isNullOrBlank()) {
                "{}"
            } else {
                JSONObject().put("session_id", initialSessionId).toString()
            }
            conn.outputStream.use { it.write(requestBody.toByteArray()) }
            val code = conn.responseCode
            val stream = if (code in 200..299) conn.inputStream else conn.errorStream
            val body = stream?.bufferedReader()?.use { it.readText() }.orEmpty()
            if (code !in 200..299) {
                throw IllegalStateException("session create HTTP $code: $body")
            }
            val obj = JSONObject(body)
            val sid = obj.optString("session_id")
            if (sid.isBlank()) throw IllegalStateException("missing session_id")
            val wsUrl = obj.optString("ws_url")
            val path = when {
                wsUrl.contains("?") -> wsUrl.substringBefore("?")
                wsUrl.isNotBlank() -> wsUrl
                else -> "/api/v1/realtime/ws"
            }
            val provider = obj.optString("talker").ifBlank {
                if (path.contains("/openai/")) "openai" else "qwen"
            }
            val sampleRate = obj.optInt(
                "input_audio_sample_rate",
                if (provider == "openai") AudioCapture.OPENAI_SAMPLE_RATE else AudioCapture.QWEN_SAMPLE_RATE
            )
            val interruptionHandling = obj.optBoolean("interruption_handling_enabled", true)
            return BackendSession(sid, path, provider, sampleRate, interruptionHandling)
        } finally {
            conn.disconnect()
        }
    }

    fun disconnect() {
        open.set(false)
        responseInProgress = false
        currentResponseItemId = null
        webSocket?.close(1000, "user disconnect")
        webSocket = null
        listener.onConnectionChanged(false)
    }

    fun isConnected(): Boolean = open.get() && webSocket != null

    /** Send a typed user message and ask the model to respond. */
    fun sendText(userText: String) {
        if (!isConnected()) {
            listener.onError("Not connected to GPT")
            return
        }
        val trimmed = userText.trim()
        if (trimmed.isEmpty()) return

        sendJson(JSONObject().apply {
            put("type", "conversation.item.create")
            put("item", JSONObject().apply {
                put("type", "message")
                put("role", "user")
                put("content", JSONArray().put(JSONObject().apply {
                    put("type", "input_text")
                    put("text", trimmed)
                }))
            })
        })
        sendJson(JSONObject().apply {
            put("type", "response.create")
        })
    }

    /** Append a PCM16 mono chunk (Base64 over the wire). */
    fun appendAudioPcm16(pcm: ByteArray) {
        if (!isConnected() || pcm.isEmpty()) return
        val b64 = Base64.encodeToString(pcm, Base64.NO_WRAP)
        sendJson(JSONObject().apply {
            put("type", "input_audio_buffer.append")
            put("audio", b64)
        })
    }

    /**
     * Full-duplex barge-in: cancel the current response and truncate unplayed audio
     * so the conversation history matches what the user actually heard.
     */
    fun interruptAssistant(playedMs: Int) {
        if (!isConnected()) return
        val itemId = currentResponseItemId
        val heardText = assistantBuffer.toString().trim()
        if (responseInProgress) {
            sendJson(JSONObject().apply { put("type", "response.cancel") })
        }
        if (!itemId.isNullOrBlank() && playedMs >= 0) {
            sendJson(JSONObject().apply {
                put("type", "conversation.item.truncate")
                put("item_id", itemId)
                put("content_index", 0)
                put("audio_end_ms", playedMs)
            })
            truncatedAssistantItemIds.add(itemId)
            ignoreNextAssistantDone = true
        }
        if (interruptionHandlingEnabled && heardText.isNotBlank()) {
            listener.onAssistantTextDone("$heardText $INTERRUPTED_MARKER")
        }
        responseInProgress = false
        currentResponseItemId = null
        assistantBuffer.clear()
    }

    private fun openWebSocketViaBackend(sessionId: String, wsPath: String) {
        val path = wsPath.trim().ifBlank { "/api/v1/realtime/ws" }
        val wsUrl = backendBaseUrl
            .replace("https://", "wss://")
            .replace("http://", "ws://")
            .trimEnd('/') + "$path?session_id=$sessionId"
        Log.i(TAG, "Opening Talker proxy $wsUrl")
        val request = Request.Builder().url(wsUrl).build()

        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                open.set(true)
                listener.onConnectionChanged(true)
                val label = if (talkerProvider == "openai") "GPT Realtime" else "Qwen Omni"
                listener.onStatus("$label connected (full-duplex)")
                // Backend seeds the Qwen session.update. Do not send a second
                // Android-side update: Qwen rejects some OpenAI-style fields.
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                runCatching { handleServerEvent(JSONObject(text)) }
                    .onFailure { Log.e(TAG, "Bad server event: $text", it) }
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                open.set(false)
                Log.e(TAG, "WebSocket failure", t)
                listener.onConnectionChanged(false)
                val detail = buildString {
                    append(t.message ?: "WebSocket error")
                    response?.let { resp ->
                        append(" (HTTP ").append(resp.code).append(' ').append(resp.message).append(')')
                        val body = runCatching { resp.body?.string() }.getOrNull().orEmpty()
                        if (body.isNotBlank()) append(": ").append(body.take(240))
                    }
                }
                listener.onError(detail)
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                open.set(false)
                listener.onConnectionChanged(false)
                listener.onStatus("Talker disconnected")
            }
        })
    }

    /** Qwen Omni Realtime session shape (PCM in 16 kHz / out 24 kHz). */
    private fun configureQwenSession() {
        sendJson(JSONObject().apply {
            put("type", "session.update")
            put("session", JSONObject().apply {
                put("modalities", JSONArray().put("text").put("audio"))
                put("voice", "Tina")
                put("instructions", shoppingInstructions())
                put("input_audio_format", "pcm")
                put("output_audio_format", "pcm")
                put("input_audio_transcription", JSONObject().apply {
                    put("model", "qwen3-asr-flash-realtime")
                    put("language", "en")
                })
                put("turn_detection", JSONObject().apply {
                    put("type", "server_vad")
                    // Match the working PC voice-test settings.
                    put("threshold", 0.4)
                    put("prefix_padding_ms", 300)
                    put("silence_duration_ms", 800)
                })
            })
        })
    }

    private fun sendJson(obj: JSONObject) {
        val socket = webSocket
        if (socket == null || !open.get()) return
        socket.send(obj.toString())
    }

    private fun handleServerEvent(event: JSONObject) {
        when (event.optString("type")) {
            "session.created", "session.updated" -> Unit

            "input_audio_buffer.speech_started" -> {
                listener.onUserSpeechStarted()
            }

            "input_audio_buffer.speech_stopped" -> {
                listener.onUserSpeechStopped()
            }

            "conversation.item.input_audio_transcription.delta" -> {
                val delta = event.optString("delta").ifBlank { event.optString("text") }
                if (delta.isNotBlank()) listener.onUserTranscript(delta, false)
            }

            "conversation.item.input_audio_transcription.completed" -> {
                val transcript = event.optString("transcript").ifBlank { event.optString("text") }
                if (transcript.isNotBlank()) listener.onUserTranscript(transcript, true)
            }

            "response.created" -> {
                responseInProgress = true
                assistantBuffer.clear()
            }

            "response.output_item.added", "response.output_item.created" -> {
                val item = event.optJSONObject("item")
                val itemId = item?.optString("id")?.takeIf { it.isNotBlank() }
                    ?: event.optString("item_id").takeIf { it.isNotBlank() }
                if (itemId != null) {
                    currentResponseItemId = itemId
                    listener.onAssistantItemStarted(itemId)
                }
            }

            "response.output_audio.delta", "response.audio.delta" -> {
                val b64 = event.optString("delta")
                if (b64.isNotBlank()) {
                    val pcm = Base64.decode(b64, Base64.DEFAULT)
                    if (pcm.isNotEmpty()) listener.onAssistantAudio(pcm)
                }
            }

            "response.output_text.delta", "response.text.delta",
            "response.output_audio_transcript.delta", "response.audio_transcript.delta" -> {
                val delta = event.optString("delta")
                if (delta.isNotEmpty()) {
                    assistantBuffer.append(delta)
                    listener.onAssistantTextDelta(delta)
                }
            }

            "response.output_text.done", "response.text.done" -> {
                val itemId = event.optString("item_id").takeIf { it.isNotBlank() }
                if (shouldIgnoreAssistantDone(itemId)) return
                val full = event.optString("text").ifBlank { assistantBuffer.toString() }
                if (full.isNotBlank()) listener.onAssistantTextDone(full)
                assistantBuffer.clear()
            }

            "response.output_audio_transcript.done", "response.audio_transcript.done" -> {
                val itemId = event.optString("item_id").takeIf { it.isNotBlank() }
                if (shouldIgnoreAssistantDone(itemId)) return
                val full = event.optString("transcript")
                    .ifBlank { event.optString("text") }
                    .ifBlank { assistantBuffer.toString() }
                if (full.isNotBlank()) listener.onAssistantTextDone(full)
                assistantBuffer.clear()
            }

            "response.done", "response.cancelled" -> {
                if (shouldIgnoreAssistantDone(null)) {
                    responseInProgress = false
                    return
                }
                if (assistantBuffer.isNotEmpty()) {
                    listener.onAssistantTextDone(assistantBuffer.toString())
                    assistantBuffer.clear()
                }
                responseInProgress = false
                // Keep item id until next response so late barge-in truncate still works briefly.
            }

            "error" -> {
                val message = event.optJSONObject("error")?.optString("message")
                    ?: event.optString("message").ifBlank { event.toString() }
                // Interrupt/cancel races often emit benign errors; log but surface to UI.
                Log.w(TAG, "Realtime error: $message")
                if (!message.contains("Cancellation failed", ignoreCase = true) &&
                    !message.contains("no active response", ignoreCase = true)
                ) {
                    listener.onError(message)
                }
            }
        }
    }

    private fun shouldIgnoreAssistantDone(itemId: String?): Boolean {
        if (itemId != null && truncatedAssistantItemIds.remove(itemId)) {
            assistantBuffer.clear()
            ignoreNextAssistantDone = false
            return true
        }
        if (ignoreNextAssistantDone) {
            assistantBuffer.clear()
            ignoreNextAssistantDone = false
            return true
        }
        return false
    }

    private fun shoppingInstructions(): String {
        return if (interruptionHandlingEnabled) {
            "$SHOPPING_INSTRUCTIONS\n\n$INTERRUPTION_INSTRUCTIONS"
        } else {
            SHOPPING_INSTRUCTIONS
        }
    }

    companion object {
        private const val TAG = "RealtimeSession"
        private const val INTERRUPTED_MARKER = "[INTERRUPTED]"
        private const val SHOPPING_INSTRUCTIONS =
            "You are VoiceShop++, a helpful retail shopping assistant. " +
                "Always reply in English only — never switch to Chinese or other languages. " +
                "Help the user clarify product category, budget, must-haves, nice-to-haves, " +
                "brand preferences, and constraints. " +
                "Ask concise clarifying questions when critical fields are missing. " +
                "Keep spoken replies short so the user can interrupt naturally."
        private const val INTERRUPTION_INSTRUCTIONS =
            "If a prior assistant message contains [INTERRUPTED], that marks where the " +
                "user cut you off; the user did not hear anything after that marker. " +
                "## Handling interruptions (barge-in)\n\n" +
                "The user can and will interrupt you mid-sentence. When that happens:\n\n" +
                "1. You did NOT finish speaking. The user only heard the words up to the " +
                "point where they cut in — never assume the rest was heard. Never claim " +
                "or imply you \"already mentioned\" or \"just said\" something that came " +
                "after the cut-off point.\n\n" +
                "2. Whatever the user says when interrupting is their most current and " +
                "highest-priority intent. Adopt it immediately and let it override your " +
                "previous plan or the answer you were in the middle of giving.\n\n" +
                "3. Be brief. Respond directly to what they now want. Do NOT re-read, " +
                "recap, or repeat the recommendations or details they already heard " +
                "before interrupting.\n\n" +
                "4. Only restate something from the unspoken part if it is essential to the " +
                "user's new request AND they were cut off before hearing it — and then " +
                "keep it to a single short clause, not a re-listing.\n\n" +
                "5. If the interruption is just a backchannel (\"uh-huh\", \"okay\", \"right\", " +
                "\"mm-hmm\", \"got it\") or clearly not addressed to you (background speech, " +
                "someone else talking), do NOT treat it as a new instruction. Continue " +
                "naturally from where you were.\n\n" +
                "Keep every reply short and spoken-style — one or two sentences."
    }
}

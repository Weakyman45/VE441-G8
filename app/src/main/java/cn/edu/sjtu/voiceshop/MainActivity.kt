package cn.edu.sjtu.voiceshop

import android.Manifest
import android.app.Activity
import android.app.AlertDialog
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Outline
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.media.AudioManager
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.provider.OpenableColumns
import android.speech.RecognizerIntent
import android.text.Editable
import android.text.InputType
import android.text.TextWatcher
import android.util.Base64
import android.util.Log
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.ViewOutlineProvider
import android.view.WindowInsetsController
import android.view.inputmethod.EditorInfo
import android.widget.CheckBox
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.HorizontalScrollView
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.Space
import android.widget.TextView
import android.widget.Toast
import java.io.ByteArrayOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import java.util.Locale
import java.util.concurrent.Executors
import org.json.JSONArray
import org.json.JSONObject

class MainActivity : Activity() {
    private enum class Stage {
        EXPRESS,
        CONTEXT,
        REFINE,
        RECOMMEND,
        DETAIL,
        COMPARE,
        RESULT
    }

    private enum class SortMode(val label: String) {
        MATCH("Best match"), PRICE("Lowest price"), RATING("Top reviews")
    }

    private enum class VoiceTarget { NEED, REQUIREMENT }

    private data class Laptop(
        val id: String,
        val name: String,
        val price: Int,
        val match: Int,
        val rating: Double,
        val reviewCount: String,
        val display: String,
        val performance: String,
        val battery: String,
        val weightKg: Double,
        val summary: String,
        val reviewSentiment: String,
        val weakness: String,
        val reasons: List<String>,
        val tradeOffs: List<String>,
        val color: Int,
        val platform: String = "Windows",
        val imageUrl: String = ""
    )

    private val ink = Color.rgb(30, 39, 50)
    private val muted = Color.rgb(99, 111, 126)
    private val accent = Color.rgb(247, 197, 20)
    private val accentDark = Color.rgb(116, 88, 0)
    private val paleYellow = Color.rgb(255, 248, 216)
    private val surface = Color.rgb(246, 248, 251)
    private val border = Color.rgb(216, 222, 230)
    private val blue = Color.rgb(45, 106, 139)
    private val paleBlue = Color.rgb(232, 245, 251)
    private val green = Color.rgb(38, 124, 82)
    private val paleGreen = Color.rgb(232, 247, 239)
    private val red = Color.rgb(166, 57, 57)
    private val paleRed = Color.rgb(254, 239, 239)

    private val fallbackLaptops = listOf(
        Laptop(
            "nova", "NovaBook Studio 14", 6899, 96, 4.8, "2,840 reviews",
            "14.5-inch 2.8K OLED, 100% DCI-P3",
            "Ryzen 7, 16GB RAM, 1TB SSD", "Up to 10 hours", 1.35,
            "Best display for design coursework",
            "Praised for screen accuracy and quiet fan behavior",
            "OLED can shorten battery life at high brightness",
            listOf("Within your ¥7,000 budget", "Color-accurate OLED display", "16GB memory handles Adobe and Figma", "Portable enough for campus"),
            listOf("No touch screen", "Fans are audible during long renders"),
            Color.rgb(242, 197, 76)
        ),
        Laptop(
            "zen", "ZenLite Pro 14", 6699, 93, 4.7, "1,960 reviews",
            "14-inch 2.5K IPS, anti-glare",
            "Core Ultra 5, 16GB RAM, 1TB SSD", "Up to 14 hours", 1.22,
            "Best portability and battery value",
            "Praised for portability, battery life, and value",
            "IPS blacks and contrast trail the OLED finalists",
            listOf("¥200 below the top pick", "Lightest finalist", "Long all-day battery", "Quiet in studio classes"),
            listOf("Display has lower contrast", "Integrated graphics limits complex 3D work"),
            Color.rgb(116, 155, 191)
        ),
        Laptop(
            "pixel", "MacBook Air 13 (M2)", 6999, 89, 4.7, "4,100 reviews",
            "13.6-inch Liquid Retina, 500 nits",
            "Apple M2, 16GB unified memory, 256GB SSD", "Up to 18 hours", 1.24,
            "Portable macOS option with long battery life",
            "Reviewers value battery life, build quality, and quiet operation",
            "Base storage is limited for large design project files",
            listOf("Long battery life", "Color-managed Retina display", "Very easy to carry"),
            listOf("Small canvas for design tools", "Base storage is only 256GB"),
            Color.rgb(126, 174, 142),
            "macOS"
        ),
        Laptop(
            "canvas", "CanvasPro 15", 7399, 91, 4.8, "1,480 reviews",
            "15.6-inch 3K OLED, 120Hz",
            "Core Ultra 7, 32GB RAM, 1TB SSD", "Up to 8 hours", 1.72,
            "Most performance and screen room",
            "Users praise its speed and large screen but mention fan noise",
            "Over budget and noticeably heavier for daily travel",
            listOf("Excellent large OLED canvas", "32GB memory for heavy creative work", "Strongest performance"),
            listOf("¥399 over budget", "Heaviest option", "Shortest battery estimate"),
            Color.rgb(176, 132, 190)
        ),
        Laptop(
            "flex", "StudioFlex Touch 14", 6988, 87, 4.5, "920 reviews",
            "14-inch 2.8K OLED touch display",
            "Core Ultra 5, 16GB RAM, 512GB SSD", "Up to 9 hours", 1.48,
            "Best for sketching directly on screen",
            "Reviewers like the pen experience but note weight and limited storage",
            "Touch hardware adds weight and storage is limited",
            listOf("Just within budget", "Touch and pen support", "High-contrast OLED"),
            listOf("512GB storage", "Heavier than non-touch models"),
            Color.rgb(208, 134, 112)
        )
    )

    // Loaded from assets/laptops.db (Amazon Reviews 2023 subset) when present;
    // otherwise falls back to the built-in curated catalog above.
    private val catalogColors = listOf(
        Color.rgb(242, 197, 76), Color.rgb(116, 155, 191), Color.rgb(126, 174, 142),
        Color.rgb(176, 132, 190), Color.rgb(208, 134, 112), Color.rgb(94, 160, 173)
    )
    // Route B: the catalog is fetched over HTTP from the backend
    // (backend/server.py, Amazon Reviews 2023 data). We start with the
    // built-in demo list so the UI is never empty, then replace it once the
    // network response arrives. If the backend is unreachable we keep the demo.
    @Volatile private var laptops: List<Laptop> = fallbackLaptops
    private var catalogSource: String = "Built-in demo (${fallbackLaptops.size}) - loading…"
    private val catalogExecutor = Executors.newSingleThreadExecutor()

    // Combine the typed need with image-derived keywords/category so the photo
    // actually influences the catalog search. Keywords go first so they land
    // within the backend's first-N search tokens.
    private fun buildSearchQuery(): String {
        val parts = ArrayList<String>()
        parts.addAll(imageSearchKeywords)
        if (imageCategory.isNotBlank()) parts.add(imageCategory)
        shoppingNeed.trim().takeIf { it.isNotBlank() }?.let { parts.add(it) }
        return parts.joinToString(" ").trim()
    }

    private fun loadCatalogAsync(query: String) {
        catalogExecutor.execute {
            val loaded = runCatching { fetchCatalog(query) }.getOrElse { error ->
                Log.e("VoiceShop", "Catalog fetch failed for '$query'", error)
                emptyList()
            }
            handler.post {
                if (isDestroyed) return@post
                if (loaded.isNotEmpty()) {
                    laptops = loaded
                    catalogSource = "Backend: ${loaded.size} products (q=\"$query\")"
                    Log.i("VoiceShop", catalogSource)
                } else {
                    catalogSource = "Built-in demo (${fallbackLaptops.size}) - backend unreachable at $BACKEND_BASE_URL"
                    Log.w("VoiceShop", catalogSource)
                }
                toast(catalogSource)
                render()
            }
        }
    }

    private fun fetchCatalog(query: String): List<Laptop> {
        val q = URLEncoder.encode(query.ifBlank { "laptop" }, "UTF-8")
        val url = URL("$BACKEND_BASE_URL/api/v1/search?q=$q&limit=60")
        val conn = (url.openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = 4000
            readTimeout = 6000
        }
        try {
            if (conn.responseCode != HttpURLConnection.HTTP_OK) return emptyList()
            val body = conn.inputStream.bufferedReader().use { it.readText() }
            val results = JSONObject(body).optJSONArray("results") ?: return emptyList()
            val list = ArrayList<Laptop>(results.length())
            for (i in 0 until results.length()) {
                results.optJSONObject(i)?.let { list.add(jsonToLaptop(it, i)) }
            }
            return list
        } finally {
            conn.disconnect()
        }
    }

    private fun jsonToLaptop(obj: JSONObject, index: Int): Laptop {
        val rating = obj.optDouble("rating", 0.0)
        val ratingNumber = obj.optInt("rating_number", 0)
        val store = obj.optString("store").substringBefore(" ").trim()
        val reviewCount = if (ratingNumber > 0) "%,d reviews".format(Locale.US, ratingNumber) else "No reviews yet"
        return Laptop(
            id = obj.optString("id").ifBlank { "item$index" },
            name = obj.optString("name").ifBlank { "Unnamed product" },
            price = obj.optInt("price", 0),
            match = (rating / 5.0 * 100).toInt().coerceIn(60, 99),
            rating = rating,
            reviewCount = reviewCount,
            display = obj.optString("display").orEmptyText(),
            performance = obj.optString("performance").orEmptyText(),
            battery = obj.optString("battery").orEmptyText(),
            weightKg = obj.optDouble("weight_kg", 0.0),
            summary = obj.optString("summary").ifNullOrBlank { "Product option${if (store.isNotBlank()) " from $store" else ""}" },
            reviewSentiment = obj.optString("review_sentiment").ifNullOrBlank { "No aggregated review summary available." },
            weakness = obj.optString("weakness").ifNullOrBlank { "No notable weaknesses in the catalog data." },
            reasons = jsonArrayToList(obj.optJSONArray("reasons")).ifEmpty { listOf("Matches your search") },
            tradeOffs = jsonArrayToList(obj.optJSONArray("trade_offs")).ifEmpty { listOf("Limited spec detail available for this item.") },
            color = catalogColors[index % catalogColors.size],
            platform = obj.optString("platform").ifNullOrBlank { "Windows" },
            imageUrl = obj.optString("image_url").trim()
        )
    }

    private fun jsonArrayToList(array: JSONArray?): List<String> {
        if (array == null) return emptyList()
        val out = ArrayList<String>(array.length())
        for (i in 0 until array.length()) {
            array.optString(i).takeIf { it.isNotBlank() }?.let { out.add(it.trim()) }
        }
        return out
    }

    private fun String?.orEmptyText(): String = if (this.isNullOrBlank()) "Not specified" else this

    private inline fun String?.ifNullOrBlank(fallback: () -> String): String =
        if (this.isNullOrBlank()) fallback() else this

    private var stage = Stage.EXPRESS
    private var shoppingNeed = ""
    private var selectedImageUri: String? = null
    private var selectedImageName: String? = null
    private var imageStyleSignal: String? = null
    private var uploadedImageId: String? = null
    // Concise, catalog-friendly keywords + category extracted from the image,
    // combined into the search query so results actually reflect the photo.
    private var imageSearchKeywords: List<String> = emptyList()
    private var imageCategory: String = ""
    // Product category the AI inferred for the current need (e.g. "运动鞋",
    // "coffee machine"). Drives which category-specific UI (specs, filters,
    // choices, icon) is shown so the app is not hardcoded to laptops.
    private var productCategory: String = ""
    private var imageUploadInFlight = false
    // Which engine produced the current analysis: "qwen"/"qwen-vl" (real LLM),
    // "fallback" (LLM unavailable → local rules), or null (not analyzed yet).
    private var imageLlmProvider: String? = null
    private var needLlmProvider: String? = null
    // Budget number extracted by the backend LLM (source of truth when the
    // must-have text carries no parseable number). null = unknown.
    private var analyzedBudget: Int? = null
    private var voiceTarget = VoiceTarget.NEED
    private val mustHaves = mutableListOf("Budget up to ¥7,000", "Design-studio use", "Strong display", "16GB RAM preferred")
    private val preferences = mutableListOf("Portable for campus", "Long battery life", "Silver or neutral style")
    private var osPreference = "No preference"
    private var touchPreference = "Not required"
    private var pendingConstraint = ""
    private val selectedIds = linkedSetOf<String>()
    private var sortMode = SortMode.MATCH
    private var filterUnderBudget = true
    private var filterOled = false
    private var filterPortable = false
    private var filterTopRated = false
    private var detailProductId = "nova"
    private var detailOrigin = Stage.RECOMMEND
    private var chosenProductId: String? = null
    private var finalConfirmed = false
    private var analysisProgress = 0
    private var analysisPaused = false
    // API-generated conversation summary shown on the "shopping brief" (CONTEXT) page.
    private var briefAnalysisText = ""
    private var briefAnalyzing = false
    private var briefAnalysisPaused = false
    private var briefAnalysisDone = false
    private var briefExtraInput = ""
    @Volatile private var briefAnalysisRun = 0
    private var briefAnalysisConn: HttpURLConnection? = null
    private val scrollPositions = mutableMapOf<Stage, Int>()
    private var currentScrollView: ScrollView? = null
    private var currentScrollStage = Stage.EXPRESS
    private val analysisSteps: List<String>
        get() = listOf(
            "Understanding your request",
            if (selectedImageUri == null) {
                "Using your stated preferences"
            } else {
                "Reading reference image: ${imageStyleSignal ?: "image attached"}"
            },
            "Checking display, performance, and reviews",
            "Ranking the strongest matches"
        )
    private val handler = Handler(Looper.getMainLooper())
    private val imageExecutor = Executors.newSingleThreadExecutor()
    private val analysisTick = object : Runnable {
        override fun run() {
            if (stage == Stage.REFINE && !analysisPaused && analysisProgress < analysisSteps.size) {
                analysisProgress += 1
                render()
            }
        }
    }

    // Talker (Realtime) + Worker recommendations
    private var realtime: RealtimeSession? = null
    private var audioCapture: AudioCapture? = null
    private var audioPlayer: AudioPlayer? = null
    // One-shot voice-to-text recording (backend ASR API), separate from the
    // realtime duplex call above.
    private var voiceRecordCapture: AudioCapture? = null
    private var voiceRecordBuffer: ByteArrayOutputStream? = null
    private var voiceRecordDialog: AlertDialog? = null
    // Action to run once RECORD_AUDIO permission is granted (mic is used by both
    // the realtime call and the one-shot voice-to-text recorder).
    private var pendingMicAction: (() -> Unit)? = null
    private var gptConnected = false
    private var gptConnecting = false
    private var gptListening = false
    private var gptSpeaking = false
    private var engineSessionId: String? = null
    private var realtimeTalkerProvider = "qwen"
    private var realtimeInputSampleRate = AudioCapture.QWEN_SAMPLE_RATE
    private var lastWorkerPlanId: String? = null
    private val chatLines = mutableListOf<String>()
    private var streamingAssistant = ""
    private var streamingUserPartial = ""
    private var previousAudioMode = AudioManager.MODE_NORMAL
    @Volatile private var lastLocalBargeInAt = 0L
    private val streamingRenderTick = Runnable {
        if (stage == Stage.EXPRESS) render()
    }
    private class SessionGoneException : Exception()

    private val recommendationPollTick = object : Runnable {
        override fun run() {
            val sid = engineSessionId
            if (sid.isNullOrBlank()) return
            catalogExecutor.execute {
                val result = runCatching { fetchWorkerRecommendations(sid) }
                handler.post {
                    if (isDestroyed) return@post
                    if (result.exceptionOrNull() is SessionGoneException) {
                        // Backend restarted / session expired: drop the dead id and
                        // re-run the analysis so a fresh session + Worker search runs.
                        engineSessionId = null
                        stopRecommendationPolling()
                        if (shoppingNeed.isNotBlank()) startBriefAnalysis(restart = true)
                        return@post
                    }
                    if (engineSessionId.isNullOrBlank()) return@post
                    result.getOrNull()?.let { applyWorkerBundle(it) }
                    handler.postDelayed(this, 2500L)
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        restoreState(savedInstanceState)
        window.statusBarColor = Color.WHITE
        window.navigationBarColor = Color.WHITE
        Log.i("VoiceShop", "Catalog init: $catalogSource")
        render()
        loadCatalogAsync(buildSearchQuery().ifBlank { "laptop" })
        // Local image style analysis disabled — image goes to the LLM with text.
        // selectedImageUri?.takeIf { imageStyleSignal == null }?.let { analyzeImageStyleAsync(Uri.parse(it)) }
        window.insetsController?.setSystemBarsAppearance(
            WindowInsetsController.APPEARANCE_LIGHT_STATUS_BARS or WindowInsetsController.APPEARANCE_LIGHT_NAVIGATION_BARS,
            WindowInsetsController.APPEARANCE_LIGHT_STATUS_BARS or WindowInsetsController.APPEARANCE_LIGHT_NAVIGATION_BARS
        )
    }

    override fun onSaveInstanceState(outState: Bundle) {
        outState.putString("stage", stage.name)
        outState.putString("need", shoppingNeed)
        outState.putString("imageUri", selectedImageUri)
        outState.putString("imageName", selectedImageName)
        outState.putString("imageStyleSignal", imageStyleSignal)
        outState.putString("uploadedImageId", uploadedImageId)
        outState.putStringArrayList("must", ArrayList(mustHaves))
        outState.putStringArrayList("prefs", ArrayList(preferences))
        outState.putString("osPreference", osPreference)
        outState.putString("touchPreference", touchPreference)
        outState.putString("pendingConstraint", pendingConstraint)
        outState.putStringArrayList("selected", ArrayList(selectedIds))
        outState.putString("sort", sortMode.name)
        outState.putBoolean("budget", filterUnderBudget)
        outState.putBoolean("oled", filterOled)
        outState.putBoolean("portable", filterPortable)
        outState.putBoolean("topRated", filterTopRated)
        outState.putString("productCategory", productCategory)
        outState.putString("detail", detailProductId)
        outState.putString("detailOrigin", detailOrigin.name)
        outState.putString("voiceTarget", voiceTarget.name)
        outState.putString("chosen", chosenProductId)
        outState.putBoolean("confirmed", finalConfirmed)
        outState.putInt("progress", analysisProgress)
        outState.putBoolean("paused", analysisPaused)
        super.onSaveInstanceState(outState)
    }

    private fun restoreState(state: Bundle?) {
        if (state == null) return
        stage = runCatching { Stage.valueOf(state.getString("stage") ?: "EXPRESS") }.getOrDefault(Stage.EXPRESS)
        shoppingNeed = state.getString("need", "")
        selectedImageUri = state.getString("imageUri")
        selectedImageName = state.getString("imageName")
        imageStyleSignal = state.getString("imageStyleSignal")
        uploadedImageId = state.getString("uploadedImageId")
        state.getStringArrayList("must")?.let { mustHaves.apply { clear(); addAll(it) } }
        state.getStringArrayList("prefs")?.let { preferences.apply { clear(); addAll(it) } }
        osPreference = state.getString("osPreference", "No preference")
        touchPreference = state.getString("touchPreference", "Not required")
        pendingConstraint = state.getString("pendingConstraint", "")
        state.getStringArrayList("selected")?.let { selectedIds.apply { clear(); addAll(it) } }
        sortMode = runCatching { SortMode.valueOf(state.getString("sort") ?: "MATCH") }.getOrDefault(SortMode.MATCH)
        filterUnderBudget = state.getBoolean("budget", true)
        filterOled = state.getBoolean("oled", false)
        filterPortable = state.getBoolean("portable", false)
        filterTopRated = state.getBoolean("topRated", false)
        productCategory = state.getString("productCategory", "")
        detailProductId = state.getString("detail", "nova")
        detailOrigin = runCatching { Stage.valueOf(state.getString("detailOrigin") ?: "RECOMMEND") }.getOrDefault(Stage.RECOMMEND)
        voiceTarget = runCatching { VoiceTarget.valueOf(state.getString("voiceTarget") ?: "NEED") }.getOrDefault(VoiceTarget.NEED)
        chosenProductId = state.getString("chosen")
        finalConfirmed = state.getBoolean("confirmed", false)
        analysisProgress = state.getInt("progress", 0)
        analysisPaused = state.getBoolean("paused", false)
    }

    private fun render() {
        handler.removeCallbacks(analysisTick)
        currentScrollView?.let { scrollPositions[currentScrollStage] = it.scrollY }
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.WHITE)
            isFocusableInTouchMode = true
        }
        root.addView(buildHeader())
        val screen = buildScreen()
        root.addView(FrameLayout(this).apply { addView(screen) }, LinearLayout.LayoutParams(-1, 0, 1f))
        root.addView(buildBottomNav())
        setContentView(root)
        root.requestFocus()
        currentScrollView = screen as? ScrollView
        currentScrollStage = stage
        currentScrollView?.post { currentScrollView?.scrollTo(0, scrollPositions[stage] ?: 0) }
        if (stage == Stage.REFINE && !analysisPaused && analysisProgress < analysisSteps.size) {
            handler.postDelayed(analysisTick, 900L)
        }
    }

    override fun onDestroy() {
        handler.removeCallbacksAndMessages(null)
        briefAnalysisRun++
        briefAnalysisConn?.let { conn -> runCatching { conn.disconnect() } }
        briefAnalysisConn = null
        leaveDuplexAudioMode()
        stopGptVoiceCapture()
        voiceRecordCapture?.stop()
        voiceRecordCapture = null
        voiceRecordDialog?.dismiss()
        voiceRecordDialog = null
        audioPlayer?.release()
        audioPlayer = null
        realtime?.disconnect()
        realtime = null
        gptConnected = false
        gptConnecting = false
        gptSpeaking = false
        imageExecutor.shutdownNow()
        catalogExecutor.shutdownNow()
        super.onDestroy()
    }

    @Suppress("DEPRECATION", "OVERRIDE_DEPRECATION")
    override fun onBackPressed() {
        if (stage == Stage.EXPRESS) super.onBackPressed() else goBack()
    }

    private fun buildHeader(): View {
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(8), dp(16), dp(8))
            background = rect(Color.WHITE, border, 0, 0)
            val top = LinearLayout(this@MainActivity).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                if (stage == Stage.EXPRESS) {
                    addView(Space(this@MainActivity), LinearLayout.LayoutParams(dp(76), dp(38)))
                } else {
                    addView(button("‹ Back") { goBack() }, LinearLayout.LayoutParams(dp(76), dp(38)))
                }
                addView(text("VS", 13f, ink, Typeface.BOLD).apply {
                    gravity = Gravity.CENTER
                    background = rect(accent, accentDark, 1, 8)
                }, LinearLayout.LayoutParams(dp(38), dp(34)).withMargins(left = dp(8), right = dp(8)))
                addView(text("VoiceShop++", 16f, ink, Typeface.BOLD), LinearLayout.LayoutParams(0, -2, 1f))
                addView(button("New search") { confirmStartOver() }, LinearLayout.LayoutParams(dp(92), dp(38)))
            }
            addView(top)
        }
    }

    private fun buildScreen(): View {
        val content = when (stage) {
            Stage.EXPRESS -> expressScreen()
            Stage.CONTEXT -> contextScreen()
            Stage.REFINE -> refineScreen()
            Stage.RECOMMEND -> recommendationScreen()
            Stage.DETAIL -> detailScreen()
            Stage.COMPARE -> compareScreen()
            Stage.RESULT -> resultScreen()
        }
        return (content.tag as? ScrollView) ?: content
    }

    private fun expressScreen(): View = scrollColumn().apply {
        addView(card(paleYellow, Color.rgb(234, 194, 41)).apply {
            addView(text("Find what you need", 22f, ink, Typeface.BOLD), fullWidth(bottom = dp(8)))
            addView(text("Recommendations shaped around your budget, work, and preferences.", 13f, muted), fullWidth(bottom = dp(18)))
            addView(button("VOICE  Start with voice", primary = true) {
                voiceTarget = VoiceTarget.NEED
                launchVoiceInput()
            }, fullWidth())
        }, fullWidth(bottom = dp(14)))

        addView(gptRealtimeCard(), fullWidth(bottom = dp(14)))

        val actions = LinearLayout(this@MainActivity).apply {
            orientation = LinearLayout.HORIZONTAL
            addView(entryCard(
                if (selectedImageUri == null) "+ Add Reference" else "Change Reference",
                "Add product photo or screenshot"
            ) { launchImagePicker() }, LinearLayout.LayoutParams(0, dp(94), 1f).withMargins(right = dp(8)))
            addView(entryCard("Try an example", "Tap to prefill a sample request") {
                shoppingNeed = SAMPLE_NEED
                render()
            }, LinearLayout.LayoutParams(0, dp(94), 1f))
        }
        addView(actions, fullWidth(bottom = dp(14)))
        if (selectedImageUri != null) addView(imageAttachmentCard(), fullWidth(bottom = dp(14)))

        addView(card().apply {
            addView(text("What are you looking for?", 15f, ink, Typeface.BOLD), fullWidth(bottom = dp(8)))
            val input = EditText(this@MainActivity).apply {
                setText(shoppingNeed)
                setSelection(text.length)
                hint = "Budget, brand, style, key features…"
                setTextColor(ink)
                setHintTextColor(muted)
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 14f)
                inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE or InputType.TYPE_TEXT_FLAG_CAP_SENTENCES
                minLines = 3
                maxLines = 5
                background = rect(surface, border, 1, 10)
                setPadding(dp(12), dp(10), dp(12), dp(10))
                addTextChangedListener(watcher { shoppingNeed = it })
                imeOptions = EditorInfo.IME_ACTION_DONE
            }
            addView(input, fullWidth(bottom = dp(12)))
            addView(LinearLayout(this@MainActivity).apply {
                orientation = LinearLayout.HORIZONTAL
                if (gptConnected) {
                    addView(button("Ask GPT", primary = true) { sendNeedToGpt() }, LinearLayout.LayoutParams(0, dp(46), 1f).withMargins(right = dp(8)))
                    addView(button("Review my needs") { submitNeed() }, LinearLayout.LayoutParams(0, dp(46), 1f))
                } else {
                    addView(button("Review my needs", primary = true) { submitNeed() }, fullWidth())
                }
            })
        })
    }

    private fun gptRealtimeCard(): View = card(paleBlue, Color.rgb(171, 210, 229)).apply {
        val inCall = gptConnected || gptConnecting || gptListening
        addView(text("Voice assistant", 15f, ink, Typeface.BOLD), fullWidth(bottom = dp(6)))
        addView(text(
            when {
                gptConnecting -> "Starting call…"
                gptConnected && gptSpeaking -> "In call — speak anytime to interrupt"
                gptConnected -> "In call — listening"
                else -> "Talk with the shopping assistant about what you want to buy."
            },
            12f,
            muted
        ), fullWidth(bottom = dp(12)))
        addView(
            button(
                when {
                    gptConnecting -> "Starting…"
                    inCall -> "End call"
                    else -> "Start calling"
                },
                primary = true,
                enabled = !gptConnecting
            ) {
                if (inCall && !gptConnecting) {
                    disconnectGpt()
                } else if (!gptConnecting) {
                    startCalling()
                }
            },
            fullWidth(bottom = dp(10))
        )

        val logText = buildString {
            if (chatLines.isEmpty() && streamingAssistant.isEmpty() && streamingUserPartial.isEmpty()) {
                append("Tap Start calling, then speak. Transcript appears here.")
            } else {
                chatLines.takeLast(12).forEach { append(it).append('\n') }
                if (streamingUserPartial.isNotBlank()) append("You: ").append(streamingUserPartial).append('\n')
                if (streamingAssistant.isNotBlank()) append("GPT: ").append(streamingAssistant)
            }
        }.trimEnd()
        addView(text(logText, 12f, ink).apply {
            background = rect(Color.WHITE, border, 1, 8)
            setPadding(dp(10), dp(10), dp(10), dp(10))
            minHeight = dp(96)
        }, fullWidth())
        if (chatLines.isNotEmpty() || streamingAssistant.isNotBlank()) {
            addView(button("Clear chat") {
                chatLines.clear()
                streamingAssistant = ""
                streamingUserPartial = ""
                render()
            }, fullWidth(top = dp(8)))
        }
    }

    /** One-tap: connect Realtime + open the mic for a live call. */
    private fun startCalling() {
        if (gptConnecting || gptConnected) return
        ensureMicPermissionThen {
            connectGpt()
        }
    }


    private fun imageAttachmentCard(): View = card(paleBlue, Color.rgb(171, 210, 229)).apply {
        orientation = LinearLayout.HORIZONTAL
        gravity = Gravity.CENTER_VERTICAL
        val preview = ImageView(this@MainActivity).apply {
            scaleType = ImageView.ScaleType.CENTER_CROP
            contentDescription = "Selected design reference"
            runCatching { setImageURI(Uri.parse(selectedImageUri)) }
            background = rect(Color.WHITE, border, 1, 8)
        }
        addView(preview, LinearLayout.LayoutParams(dp(72), dp(72)).withMargins(right = dp(12)))
        addView(LinearLayout(this@MainActivity).apply {
            orientation = LinearLayout.VERTICAL
            addView(text("Reference added", 14f, ink, Typeface.BOLD), fullWidth(bottom = dp(4)))
            addView(text(selectedImageName ?: "Design reference image", 12f, muted), fullWidth(bottom = dp(8)))
            addView(text(
                when {
                    imageUploadInFlight -> "正在上传图片…"
                    imageLlmProvider == "error" -> "图片上传失败，可重试"
                    uploadedImageId != null -> "已添加，将与文字一起交给大模型分析"
                    else -> "准备中…"
                },
                12f,
                blue,
                Typeface.BOLD
            ), fullWidth(bottom = dp(8)))
            addView(button("Remove image", danger = true) {
                selectedImageUri = null
                selectedImageName = null
                imageStyleSignal = null
                uploadedImageId = null
                imageSearchKeywords = emptyList()
                imageCategory = ""
                imageLlmProvider = null
                imageUploadInFlight = false
                contextSignalChanged()
                render()
            })
        }, LinearLayout.LayoutParams(0, -2, 1f))
    }

    private fun contextScreen(): View = scrollColumn().apply {
        addView(text("Your shopping brief", 22f, ink, Typeface.BOLD), fullWidth(bottom = dp(6)))
        addView(text("These preferences shape your matches.", 13f, muted), fullWidth(bottom = dp(14)))
        addView(card(paleBlue, Color.rgb(171, 210, 229)).apply {
            addView(text("Your request", 12f, blue, Typeface.BOLD), fullWidth(bottom = dp(6)))
            addView(text(shoppingNeed.ifBlank { "Describe what you want to buy — budget, brand, style, key features…" }, 14f, ink), fullWidth(bottom = if (selectedImageName != null) dp(8) else 0))
            selectedImageName?.let {
                addView(text("Image: $it", 12f, muted), fullWidth(bottom = dp(4)))
                addView(text(
                    "将与文字一起交给大模型分析",
                    12f,
                    blue,
                    Typeface.BOLD
                ))
            }
            addView(button("Edit request") { setStage(Stage.EXPRESS) }, fullWidth(top = dp(10)))
        }, fullWidth(bottom = dp(14)))
        if (briefAnalyzing || briefAnalysisPaused || briefAnalysisDone || briefAnalysisText.isNotBlank()) {
            addView(analysisProcessCard(), fullWidth(bottom = dp(14)))
        }
        addView(requirementSection("Must-haves", mustHaves, paleYellow, "Add a must-have"), fullWidth(bottom = dp(14)))
        addView(requirementSection("Nice-to-haves", preferences, paleBlue, "Add a preference"), fullWidth(bottom = dp(14)))
        // "Mac or Windows / touch" only makes sense for computers.
        if (isComputerCategory()) {
            addView(clarificationSection(), fullWidth(bottom = dp(14)))
        }
        addView(LinearLayout(this@MainActivity).apply {
            orientation = LinearLayout.HORIZONTAL
            addView(button("Add by voice") {
                voiceTarget = VoiceTarget.REQUIREMENT
                launchVoiceInput()
            }, LinearLayout.LayoutParams(0, dp(48), 1f).withMargins(right = dp(8)))
            addView(button(if (selectedImageUri == null) "Add reference" else "Change reference") { launchImagePicker() }, LinearLayout.LayoutParams(0, dp(48), 1f))
        }, fullWidth(bottom = dp(14)))
        // Ready once the image has been uploaded to the backend (no on-device
        // analysis anymore); the image is analyzed later together with the text.
        val imageReady = selectedImageUri == null || (!imageUploadInFlight && uploadedImageId != null)
        addView(button(
            if (imageReady) "Find matching products" else "Uploading reference...",
            primary = true,
            enabled = imageReady
        ) {
            if (mustHaves.isEmpty()) toast("Add at least one must-have before searching.") else {
                analysisProgress = 0
                analysisPaused = false
                setStage(Stage.REFINE)
            }
        })
    }

    private fun analysisProcessCard(): View = card(paleBlue, Color.rgb(171, 210, 229)).apply {
        addView(LinearLayout(this@MainActivity).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            addView(text("对话总结", 15f, ink, Typeface.BOLD), LinearLayout.LayoutParams(0, -2, 1f))
            val status = when {
                briefAnalyzing -> "分析中"
                briefAnalysisPaused -> "已暂停"
                briefAnalysisDone -> "完成"
                else -> ""
            }
            val statusColor = when {
                briefAnalysisPaused -> accent
                briefAnalysisDone -> green
                else -> blue
            }
            if (status.isNotEmpty()) {
                addView(text(status, 11f, statusColor, Typeface.BOLD).apply {
                    gravity = Gravity.CENTER
                    background = rect(surface, border, 1, 14)
                    setPadding(dp(10), dp(5), dp(10), dp(5))
                })
            }
        }, fullWidth(bottom = dp(10)))

        val body = when {
            briefAnalysisText.isNotBlank() -> briefAnalysisText
            briefAnalyzing -> "正在请求 API 生成这段对话的总结…"
            briefAnalysisPaused -> "已暂停。可在下面补充一句描述后继续。"
            else -> "等待 API 返回这段对话的总结。"
        }
        addView(text(body, 13f, ink), fullWidth(bottom = dp(10)))

        when {
            briefAnalyzing -> {
                addView(button("暂停并补充") { pauseBriefAnalysis() }, fullWidth())
            }
            briefAnalysisPaused -> {
                val extra = EditText(this@MainActivity).apply {
                    setText(briefExtraInput)
                    setSelection(text.length)
                    hint = "补充一句需求（可选），如：适合油皮、无香精"
                    setTextColor(ink)
                    setHintTextColor(muted)
                    setSingleLine(true)
                    setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f)
                    background = rect(Color.WHITE, border, 1, 9)
                    setPadding(dp(12), 0, dp(12), 0)
                    addTextChangedListener(watcher { briefExtraInput = it })
                }
                addView(extra, fullWidth(bottom = dp(8)).apply { height = dp(46) })
                addView(LinearLayout(this@MainActivity).apply {
                    orientation = LinearLayout.HORIZONTAL
                    addView(button("继续分析", primary = true) {
                        val add = extra.text.toString().trim()
                        if (add.isNotEmpty()) {
                            shoppingNeed = (shoppingNeed.trim() + "。" + add).trim()
                            briefExtraInput = ""
                            loadCatalogAsync(buildSearchQuery())
                        }
                        startBriefAnalysis(restart = true)
                    }, LinearLayout.LayoutParams(0, dp(46), 1f).withMargins(right = dp(8)))
                    addView(button("跳过分析") {
                        briefAnalysisPaused = false
                        briefAnalysisDone = true
                        render()
                    }, LinearLayout.LayoutParams(0, dp(46), 1f))
                })
            }
            briefAnalysisDone -> {
                addView(button("重新分析") { startBriefAnalysis(restart = true) }, fullWidth())
            }
        }
    }

    private fun requirementSection(title: String, values: MutableList<String>, chipColor: Int, addHint: String): View {
        return card().apply {
            addView(text(title, 16f, ink, Typeface.BOLD), fullWidth(bottom = dp(8)))
            if (values.isEmpty()) addView(text("None yet", 13f, muted), fullWidth(bottom = dp(8)))
            values.forEachIndexed { index, value ->
                val row = LinearLayout(this@MainActivity).apply {
                    orientation = LinearLayout.HORIZONTAL
                    gravity = Gravity.CENTER_VERTICAL
                    val editor = EditText(this@MainActivity).apply {
                        setText(value)
                        setTextColor(ink)
                        setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f)
                        setSingleLine(true)
                        background = rect(chipColor, border, 1, 18)
                        setPadding(dp(12), 0, dp(12), 0)
                        addTextChangedListener(watcher {
                            values[index] = it
                            invalidateResults()
                        })
                        setOnFocusChangeListener { _, focused -> if (!focused && text.isBlank()) removeRequirement(values, index) }
                    }
                    addView(editor, LinearLayout.LayoutParams(0, dp(42), 1f).withMargins(right = dp(6)))
                    addView(button("×", danger = true) { removeRequirement(values, index) }.apply {
                        contentDescription = "Remove $value"
                    }, LinearLayout.LayoutParams(dp(44), dp(42)))
                }
                addView(row, fullWidth(bottom = dp(7)))
            }
            val addInput = EditText(this@MainActivity).apply {
                hint = addHint
                setTextColor(ink)
                setHintTextColor(muted)
                setSingleLine(true)
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f)
                background = rect(surface, border, 1, 9)
                setPadding(dp(12), 0, dp(12), 0)
            }
            addView(LinearLayout(this@MainActivity).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                addView(addInput, LinearLayout.LayoutParams(0, dp(44), 1f).withMargins(right = dp(6)))
                addView(button("+ Add", primary = true) { addRequirement(values, addInput.text.toString()) }, LinearLayout.LayoutParams(dp(72), dp(44)))
            })
        }
    }

    private fun clarificationSection(): View = card(paleRed, Color.rgb(236, 188, 188)).apply {
        addView(text("A couple of choices", 16f, ink, Typeface.BOLD), fullWidth(bottom = dp(10)))
        addView(choiceRow("Mac or Windows?", listOf("No preference", "Windows", "macOS"), osPreference) {
            osPreference = it
            contextSignalChanged()
            render()
        }, fullWidth(bottom = dp(10)))
        addView(choiceRow("Prefer touch screen?", listOf("Not required", "Prefer touch"), touchPreference) {
            touchPreference = it
            contextSignalChanged()
            render()
        })
    }

    private fun choiceRow(label: String, options: List<String>, selected: String, onSelect: (String) -> Unit): View {
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(text(label, 13f, ink, Typeface.BOLD), fullWidth(bottom = dp(6)))
            val scroll = HorizontalScrollView(this@MainActivity).apply { isHorizontalScrollBarEnabled = false }
            val row = LinearLayout(this@MainActivity).apply { orientation = LinearLayout.HORIZONTAL }
            options.forEach { option ->
                row.addView(button(option, primary = option == selected) { onSelect(option) }, LinearLayout.LayoutParams(-2, dp(40)).withMargins(right = dp(7)))
            }
            scroll.addView(row)
            addView(scroll)
        }
    }

    private fun refineScreen(): View = scrollColumn().apply {
        addView(text("Finding matches", 22f, ink, Typeface.BOLD), fullWidth(bottom = dp(14)))
        addView(card().apply {
            addView(LinearLayout(this@MainActivity).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                addView(text(if (analysisPaused) "Search paused" else if (analysisProgress == analysisSteps.size) "${filteredLaptops().size} matches ready" else "Searching products", 15f, ink, Typeface.BOLD), LinearLayout.LayoutParams(0, -2, 1f))
                addView(text(if (analysisProgress == analysisSteps.size) "Ready" else if (analysisPaused) "Paused" else "Searching", 11f, if (analysisProgress == analysisSteps.size) green else blue, Typeface.BOLD).apply {
                    gravity = Gravity.CENTER
                    background = rect(if (analysisProgress == analysisSteps.size) paleGreen else paleBlue, border, 1, 14)
                    setPadding(dp(10), dp(5), dp(10), dp(5))
                })
            }, fullWidth(bottom = dp(12)))
            analysisSteps.forEachIndexed { index, step -> addView(progressRow(index, step), fullWidth(bottom = dp(8))) }
            addView(button(
                if (analysisPaused) "Resume search" else "Stop and edit",
                primary = analysisPaused,
                enabled = analysisProgress < analysisSteps.size
            ) {
                analysisPaused = !analysisPaused
                render()
            }, fullWidth(top = dp(6)))
        }, fullWidth(bottom = dp(14)))

        addView(card(paleRed, Color.rgb(236, 188, 188)).apply {
            addView(text("Applied to this search", 15f, ink, Typeface.BOLD), fullWidth(bottom = dp(8)))
            addView(text(confirmedSignals().joinToString("  •  "), 12f, ink), fullWidth(bottom = dp(12)))
            val canEdit = analysisPaused || analysisProgress == analysisSteps.size
            val newConstraint = EditText(this@MainActivity).apply {
                setText(pendingConstraint)
                setSelection(text.length)
                hint = if (canEdit) "Add another must-have" else "Pause search to edit"
                setTextColor(ink)
                setHintTextColor(muted)
                setSingleLine(true)
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f)
                background = rect(Color.WHITE, border, 1, 9)
                setPadding(dp(12), 0, dp(12), 0)
                addTextChangedListener(watcher { pendingConstraint = it })
                isEnabled = canEdit
                alpha = if (canEdit) 1f else 0.65f
            }
            addView(newConstraint, fullWidth(bottom = dp(8)).apply { height = dp(46) })
            addView(LinearLayout(this@MainActivity).apply {
                orientation = LinearLayout.HORIZONTAL
                addView(button("Add need", primary = true, enabled = canEdit) {
                    val value = newConstraint.text.toString()
                    if (value.isNotBlank()) pendingConstraint = ""
                    addRequirement(mustHaves, value)
                }, LinearLayout.LayoutParams(0, dp(46), 1f).withMargins(right = dp(8)))
                addView(button("Add by voice", enabled = canEdit) {
                    voiceTarget = VoiceTarget.REQUIREMENT
                    launchVoiceInput()
                }, LinearLayout.LayoutParams(0, dp(46), 1f))
            })
        }, fullWidth(bottom = dp(14)))
        addView(button("View ${filteredLaptops().size} matches", primary = true, enabled = analysisProgress == analysisSteps.size) { setStage(Stage.RECOMMEND) })
    }

    private fun progressRow(index: Int, label: String): View {
        val done = index < analysisProgress
        val current = index == analysisProgress && analysisProgress < analysisSteps.size && !analysisPaused
        val paused = index == analysisProgress && analysisProgress < analysisSteps.size && analysisPaused
        return LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            addView(text(if (done) "✓" else if (current) "●" else if (paused) "Ⅱ" else "○", 12f, if (done) Color.WHITE else ink, Typeface.BOLD).apply {
                gravity = Gravity.CENTER
                background = rect(if (done) green else if (current) accent else surface, border, 1, 20)
            }, LinearLayout.LayoutParams(dp(28), dp(28)).withMargins(right = dp(10)))
            addView(text(label, 12f, if (done || current) ink else muted), LinearLayout.LayoutParams(0, -2, 1f))
            addView(text(if (done) "Complete" else if (current) "In progress" else if (paused) "Paused" else "Waiting", 11f, if (done) green else muted, if (done) Typeface.BOLD else Typeface.NORMAL))
        }
    }

    private fun recommendationScreen(): View = scrollColumn(left = dp(14), right = dp(14)).apply {
        val cat = categoryLabel()
        addView(text(
            when {
                shoppingNeed.isBlank() -> "Explore products"
                cat.isNotBlank() -> "Matches for $cat"
                else -> "Matches for you"
            },
            22f, ink, Typeface.BOLD
        ), fullWidth(bottom = dp(4)))
        addView(text(
            if (shoppingNeed.isBlank()) "Popular options for you"
            else "${money(budgetLimit())} budget • ${mustHaves.size + preferences.size} active preferences",
            13f,
            muted
        ), fullWidth(bottom = dp(12)))
        addView(text("Sort by", 13f, ink, Typeface.BOLD), fullWidth(bottom = dp(7)))
        addView(sortControls(), fullWidth(bottom = dp(12)))
        addView(filterControls(), fullWidth(bottom = dp(12)))

        val visible = filteredLaptops()
        val hiddenSelections = selectedIds.count { id -> visible.none { it.id == id } }
        addView(LinearLayout(this@MainActivity).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            addView(text("${visible.size} ${if (visible.size == 1) "result" else "results"}", 14f, ink, Typeface.BOLD), LinearLayout.LayoutParams(0, -2, 1f))
            addView(text(
                "${selectedIds.size} of $MAX_COMPARE in compare${if (hiddenSelections > 0) " • $hiddenSelections hidden" else ""}",
                13f,
                if (canCompare()) green else blue,
                Typeface.BOLD
            ))
        }, fullWidth(bottom = dp(10)))
        if (visible.isEmpty()) {
            addView(card(surface).apply {
                addView(text("No products match all active filters.", 14f, ink, Typeface.BOLD), fullWidth(bottom = dp(10)))
                addView(button("Clear filters", primary = true) {
                    filterUnderBudget = false; filterOled = false; filterPortable = false; filterTopRated = false; render()
                })
            }, fullWidth(bottom = dp(12)))
        } else visible.forEach { addView(productCard(it), fullWidth(bottom = dp(10))) }
        addView(button(
            if (canCompare()) "Compare ${selectedIds.size} items" else "Choose 2+ to compare",
            primary = canCompare(),
            enabled = canCompare()
        ) { setStage(Stage.COMPARE) }, fullWidth(top = dp(4), bottom = dp(8)))
        addView(button("Edit needs") { beginRefinement() })
    }

    private fun sortControls(): View {
        val scroll = HorizontalScrollView(this).apply { isHorizontalScrollBarEnabled = false }
        val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        SortMode.entries.forEach { mode ->
            row.addView(button(mode.label, primary = sortMode == mode) { sortMode = mode; render() }, LinearLayout.LayoutParams(-2, dp(40)).withMargins(right = dp(7)))
        }
        scroll.addView(row)
        return scroll
    }

    private fun filterControls(): View = card(surface).apply {
        addView(text("Filters", 13f, ink, Typeface.BOLD), fullWidth(bottom = dp(4)))
        addView(CheckBox(this@MainActivity).apply {
            text = "Under ${money(budgetLimit())}"
            setTextColor(ink)
            isChecked = filterUnderBudget
            setOnCheckedChangeListener { _, value -> filterUnderBudget = value; render() }
        })
        // OLED / weight are laptop-specific — only offer them for computers.
        if (isComputerCategory()) {
            addView(CheckBox(this@MainActivity).apply {
                text = "OLED display"
                setTextColor(ink)
                isChecked = filterOled
                setOnCheckedChangeListener { _, value -> filterOled = value; render() }
            })
            addView(CheckBox(this@MainActivity).apply {
                text = "Under 1.5 kg"
                setTextColor(ink)
                isChecked = filterPortable
                setOnCheckedChangeListener { _, value -> filterPortable = value; render() }
            })
        }
        addView(CheckBox(this@MainActivity).apply {
            text = "Top rated (4.5+)"
            setTextColor(ink)
            isChecked = filterTopRated
            setOnCheckedChangeListener { _, value -> filterTopRated = value; render() }
        })
    }

    // True when the current need is about computers, so laptop-specific controls
    // (OS choice, OLED/weight filters, Display/Performance specs) make sense.
    private fun isComputerCategory(): Boolean {
        val hay = (productCategory + " " + imageCategory + " " + shoppingNeed)
            .lowercase(Locale.getDefault())
        if (hay.isBlank()) return false
        return listOf(
            "laptop", "notebook", "macbook", "ultrabook", "chromebook", "computer",
            "desktop", " pc", "tablet", "ipad", "笔记本", "电脑", "平板", "超极本"
        ).any(hay::contains)
    }

    // A catalog string is worth showing only if the backend actually filled it.
    private fun isSpecMeaningful(value: String?): Boolean =
        !value.isNullOrBlank() && !value.equals("Not specified", ignoreCase = true)

    // Short label for the current category, used in headers / empty states.
    private fun categoryLabel(): String =
        productCategory.ifBlank { imageCategory }.trim()

    private fun filteredLaptops(): List<Laptop> {
        val computer = isComputerCategory()
        val filtered = laptops.filter { laptop ->
            (!filterUnderBudget || laptop.price <= budgetLimit()) &&
                (!filterTopRated || laptop.rating >= 4.5) &&
                (!(computer && filterOled) || laptop.display.contains("OLED")) &&
                (!(computer && filterPortable) || laptop.weightKg < 1.5) &&
                (!computer || osPreference == "No preference" || laptop.platform == osPreference)
        }
        return when (sortMode) {
            SortMode.MATCH -> filtered.sortedByDescending { matchScore(it) }
            SortMode.PRICE -> filtered.sortedBy { it.price }
            SortMode.RATING -> filtered.sortedByDescending { it.rating }
        }
    }

    private fun budgetLimit(): Int {
        // 1) A number written in the must-haves is authoritative (user-editable).
        budgetsIn(mustHaves.joinToString(" ")).minOrNull()?.let { return it }
        // 2) Otherwise use the budget the backend LLM extracted from the request.
        analyzedBudget?.let { return it }
        // 3) Otherwise try the free-text request, else a permissive default.
        return budgetsIn(shoppingNeed).minOrNull() ?: 7_000
    }

    private fun budgetsIn(text: String): List<Int> = Regex("(?<!\\d)(\\d[\\d,]{1,6})(?!\\d)")
        .findAll(text)
        .mapNotNull { it.groupValues[1].replace(",", "").toIntOrNull() }
        // Generic catalog: budgets can be small (¥100 shoes) or large. Only drop
        // clearly-not-a-price values. Do NOT cut off amounts under 3000 anymore.
        .filter { it in 50..1_000_000 }
        .toList()

    private fun confirmedSignals(): List<String> = buildList {
        addAll(mustHaves)
        addAll(preferences)
        add("OS: $osPreference")
        add("Touch: $touchPreference")
        // Image style text removed — the image is analyzed multimodally by the LLM.
        // imageStyleSignal?.let { add("Image style: $it") }
    }

    private fun deriveAssumptionsFromNeed() {
        budgetsIn(shoppingNeed).minOrNull()?.let { budget ->
            val index = mustHaves.indexOfFirst { it.contains("budget", ignoreCase = true) }
            val label = "Budget up to ${money(budget)}"
            if (index >= 0) mustHaves[index] = label else mustHaves.add(0, label)
        }
        val lower = shoppingNeed.lowercase(Locale.getDefault())
        osPreference = when {
            lower.contains("macos") || lower.contains("macbook") || lower.contains(" mac ") -> "macOS"
            lower.contains("windows") -> "Windows"
            else -> osPreference
        }
        if (lower.contains("touch screen") || lower.contains("touchscreen") || lower.contains("2-in-1")) {
            touchPreference = "Prefer touch"
        }
        Regex("(?i)(8|16|32)\\s*gb").find(shoppingNeed)?.groupValues?.getOrNull(1)?.let { memory ->
            val index = mustHaves.indexOfFirst { it.contains("RAM", ignoreCase = true) || it.contains("memory", ignoreCase = true) }
            val label = "${memory}GB RAM preferred"
            if (index >= 0) mustHaves[index] = label else mustHaves.add(label)
        }
        if (listOf("portable", "lightweight", "campus", "commute").any(lower::contains) &&
            preferences.none { it.contains("portable", ignoreCase = true) || it.contains("light", ignoreCase = true) }
        ) {
            preferences.add("Lightweight / portable")
        }
    }

    private fun matchScore(product: Laptop): Int {
        val context = (listOf(shoppingNeed, osPreference, touchPreference) + mustHaves + preferences)
            .joinToString(" ")
            .lowercase(Locale.getDefault())
        var score = product.match
        val budget = budgetLimit()
        score += if (product.price <= budget) 1 else -((product.price - budget) / 100).coerceIn(2, 15)
        if (listOf("display", "design", "color", "oled").any(context::contains)) {
            score += if (product.display.contains("OLED")) 2 else 0
        }
        if (listOf("portable", "lightweight", "campus", "commut").any(context::contains)) {
            score += when {
                product.weightKg <= 1.25 -> 2
                product.weightKg <= 1.4 -> 1
                product.weightKg > 1.6 -> -4
                else -> 0
            }
        }
        if (context.contains("battery")) {
            score += when (product.id) {
                "zen" -> 2
                "pixel" -> 1
                "canvas" -> -2
                else -> 0
            }
        }
        if (context.contains("review") || context.contains("sentiment")) {
            score += when {
                product.rating >= 4.8 -> 3
                product.rating >= 4.7 -> 2
                product.rating >= 4.6 -> 1
                else -> 0
            }
        }
        if (context.contains("quiet") || context.contains("fan noise") || context.contains("fan behavior")) {
            score += when (product.id) {
                "nova" -> 3
                "zen", "pixel" -> 2
                "canvas" -> -3
                else -> 0
            }
        }
        if (touchPreference == "Prefer touch" || context.contains("touch screen") || context.contains("touchscreen")) {
            score += if (product.id == "flex") 10 else -3
        }
        if (context.contains("15-inch") || context.contains("15 inch") || context.contains("large screen")) {
            score += if (product.id == "canvas") 9 else -2
        }
        if (context.contains("3d") || context.contains("32gb") || context.contains("heavy render")) {
            score += if (product.id == "canvas") 8 else -2
        }
        imageStyleSignal?.lowercase(Locale.getDefault())?.let { signal ->
            score += when {
                signal.contains("dark") && product.id in setOf("canvas", "flex") -> 2
                signal.contains("warm") && product.id in setOf("nova", "flex") -> 2
                signal.contains("cool") && product.id in setOf("zen", "pixel") -> 2
                (signal.contains("light") || signal.contains("neutral")) && product.id in setOf("nova", "zen") -> 2
                else -> 0
            }
        }
        return score.coerceIn(60, 99)
    }

    private fun matchReasons(product: Laptop): List<String> {
        val budgetReason = if (product.price <= budgetLimit()) {
            "Within your confirmed ${money(budgetLimit())} budget"
        } else {
            "${money(product.price - budgetLimit())} above the confirmed budget"
        }
        return listOf(budgetReason) + product.reasons.filterNot {
            it.contains("budget", ignoreCase = true) || it.contains("below", ignoreCase = true) || it.contains("price", ignoreCase = true)
        }
    }

    private fun recommendationSubtitle(product: Laptop): String {
        val context = (listOf(shoppingNeed, osPreference, touchPreference) + mustHaves + preferences)
            .joinToString(" ")
            .lowercase(Locale.getDefault())
        return when {
            product.id == "flex" && (touchPreference == "Prefer touch" || context.contains("touch")) ->
                "Recommended for touch-first design work"
            product.platform == "macOS" && osPreference == "macOS" ->
                "Recommended for macOS, portability, and battery life"
            product.id == "canvas" && listOf("15-inch", "15 inch", "large screen", "3d", "32gb").any(context::contains) ->
                "Recommended for performance and a larger design canvas"
            product.id == "zen" && listOf("portable", "lightweight", "battery", "campus").any(context::contains) ->
                "Recommended for portability and value"
            product.id == "nova" && listOf("display", "design", "color", "review").any(context::contains) ->
                "Recommended for display-first design work"
            else -> product.summary
        }
    }

    // Per-card spec line: laptop specs only for computers, and only when the
    // catalog actually filled them. Non-computer categories (shoes, coffee…)
    // have no structured specs here, so the line is omitted rather than showing
    // "Display: Not specified / Performance: Not specified".
    private fun cardSpecLine(product: Laptop): String? {
        if (!isComputerCategory()) return null
        val parts = buildList {
            if (isSpecMeaningful(product.display)) add("Display: ${product.display}")
            if (isSpecMeaningful(product.performance)) add("Performance: ${product.performance}")
        }
        return parts.joinToString("\n").ifBlank { null }
    }

    private fun productCard(product: Laptop): View = card(if (selectedIds.contains(product.id)) paleYellow else Color.WHITE, if (selectedIds.contains(product.id)) accent else border).apply {
        addView(LinearLayout(this@MainActivity).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.TOP
            addView(ProductThumbView(this@MainActivity, product.color, isComputerCategory(), product.imageUrl), LinearLayout.LayoutParams(dp(76), dp(70)).withMargins(right = dp(10)))
            addView(LinearLayout(this@MainActivity).apply {
                orientation = LinearLayout.VERTICAL
                addView(text(product.name, 16f, ink, Typeface.BOLD), fullWidth(bottom = dp(3)))
                addView(text("${money(product.price)}  •  ${product.rating}/5 (${product.reviewCount})", 12f, ink), fullWidth(bottom = dp(4)))
                addView(text("${matchScore(product)}% match  •  ${product.summary}", 12f, green, Typeface.BOLD))
            }, LinearLayout.LayoutParams(0, -2, 1f))
        }, fullWidth(bottom = dp(8)))
        cardSpecLine(product)?.let { addView(text(it, 12f, muted), fullWidth(bottom = dp(9))) }
        addView(LinearLayout(this@MainActivity).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            addView(CheckBox(this@MainActivity).apply {
                text = if (selectedIds.contains(product.id)) "In compare" else "Add to compare"
                setTextColor(ink)
                isChecked = selectedIds.contains(product.id)
                setOnCheckedChangeListener { _, checked -> toggleFinalist(product, checked) }
            }, LinearLayout.LayoutParams(0, -2, 1f))
            addView(button("View details") { openDetails(product.id, Stage.RECOMMEND) }, LinearLayout.LayoutParams(dp(104), dp(42)))
        })
    }

    private fun detailScreen(): View {
        val product = laptop(detailProductId)
        return scrollColumn().apply {
            addView(LinearLayout(this@MainActivity).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                addView(ProductThumbView(this@MainActivity, product.color, isComputerCategory(), product.imageUrl), LinearLayout.LayoutParams(dp(104), dp(92)).withMargins(right = dp(14)))
                addView(LinearLayout(this@MainActivity).apply {
                    orientation = LinearLayout.VERTICAL
                    addView(text(product.name, 20f, ink, Typeface.BOLD), fullWidth(bottom = dp(5)))
                    addView(text(money(product.price), 17f, ink, Typeface.BOLD), fullWidth(bottom = dp(5)))
                    addView(text("${matchScore(product)}% match", 13f, green, Typeface.BOLD), fullWidth(bottom = dp(3)))
                    addView(text("${product.rating}/5 • ${product.reviewCount}", 12f, muted))
                }, LinearLayout.LayoutParams(0, -2, 1f))
            }, fullWidth(bottom = dp(14)))
            addView(card(paleGreen, Color.rgb(174, 220, 194)).apply {
                addView(text("Why it matches your needs", 16f, ink, Typeface.BOLD), fullWidth(bottom = dp(8)))
                addView(bullets(matchReasons(product), green))
            }, fullWidth(bottom = dp(12)))
            addView(card().apply {
                addView(text("Product details", 16f, ink, Typeface.BOLD), fullWidth(bottom = dp(8)))
                if (categoryLabel().isNotBlank()) addView(detailRow("Category", categoryLabel()))
                if (isComputerCategory()) {
                    addView(detailRow("Platform", product.platform))
                    if (isSpecMeaningful(product.display)) addView(detailRow("Display", product.display))
                    if (isSpecMeaningful(product.performance)) addView(detailRow("Performance", product.performance))
                    if (isSpecMeaningful(product.battery)) addView(detailRow("Battery", product.battery))
                    if (product.weightKg > 0.0) addView(detailRow("Weight", "${product.weightKg} kg"))
                }
                addView(detailRow("Reviews", "${product.rating}/5 from ${product.reviewCount}"))
                if (isSpecMeaningful(product.reviewSentiment) &&
                    !product.reviewSentiment.startsWith("No aggregated", ignoreCase = true)) {
                    addView(detailRow("Review insight", product.reviewSentiment))
                }
            }, fullWidth(bottom = dp(12)))
            addView(card(paleRed, Color.rgb(236, 188, 188)).apply {
                addView(text("Trade-offs to consider", 16f, ink, Typeface.BOLD), fullWidth(bottom = dp(8)))
                addView(bullets(product.tradeOffs, red))
            }, fullWidth(bottom = dp(14)))
            addView(button(if (selectedIds.contains(product.id)) "Remove from compare" else "Add to compare", primary = !selectedIds.contains(product.id)) {
                toggleFinalist(product, !selectedIds.contains(product.id))
            }, fullWidth(bottom = dp(8)))
            addView(button("Choose ${product.name}", primary = true) { chooseProduct(product.id) }, fullWidth(bottom = dp(8)))
            addView(button(if (canCompare()) "Compare ${selectedIds.size} items" else "Back to matches") {
                setStage(if (canCompare()) Stage.COMPARE else Stage.RECOMMEND)
            })
        }
    }

    private fun compareScreen(): View {
        val finalists = selectedIds.mapNotNull { id -> laptops.find { it.id == id } }
            .sortedByDescending { matchScore(it) }
        return scrollColumn(left = dp(14), right = dp(14)).apply {
            addView(text("Compare products", 22f, ink, Typeface.BOLD), fullWidth(bottom = dp(4)))
            addView(text(
                "Attributes side by side. Scroll sideways to see every column.",
                13f, muted
            ), fullWidth(bottom = dp(14)))
            if (finalists.size < 2) {
                addView(card(surface).apply {
                    addView(text(
                        if (finalists.isEmpty()) "Your compare list is empty" else "Add one more item",
                        16f, ink, Typeface.BOLD
                    ), fullWidth(bottom = dp(6)))
                    addView(text(
                        if (finalists.isEmpty())
                            "Add at least two items from Matches to see them side by side."
                        else "${finalists.first().name} is in compare. Add at least one more to build the table.",
                        13f, muted
                    ), fullWidth(bottom = dp(12)))
                    addView(button("Browse matches", primary = true) { setStage(Stage.RECOMMEND) })
                })
            } else {
                addView(compareTable(finalists), fullWidth(bottom = dp(12)))
                addView(button("Edit comparison") { setStage(Stage.RECOMMEND) })
            }
        }
    }

    // Side-by-side attribute table: one column per product, one row per
    // attribute. Rows are picked dynamically so laptop-only specs show only for
    // computers, and any attribute with no data across all products is skipped.
    // Wrapped in a HorizontalScrollView so 2..MAX_COMPARE columns fit.
    private fun compareTable(finalists: List<Laptop>): View {
        val labelW = dp(94)
        val colW = dp(150)
        val budget = budgetLimit()
        val computer = isComputerCategory()

        fun meaningfulWeakness(p: Laptop): Boolean =
            isSpecMeaningful(p.weakness) && !p.weakness.startsWith("No notable", ignoreCase = true)

        val attrs = mutableListOf<Pair<String, (Laptop) -> String>>()
        attrs += "Match" to { p: Laptop -> "${matchScore(p)}%" }
        attrs += "Price" to { p: Laptop ->
            money(p.price) + (if (p.price <= budget) "\nwithin budget" else "\nover budget")
        }
        attrs += "Rating" to { p: Laptop -> "${p.rating}/5\n(${p.reviewCount})" }
        if (computer) {
            attrs += "Platform" to { p: Laptop -> p.platform }
            if (finalists.any { isSpecMeaningful(it.display) })
                attrs += "Display" to { p: Laptop -> if (isSpecMeaningful(p.display)) p.display else "—" }
            if (finalists.any { isSpecMeaningful(it.performance) })
                attrs += "Performance" to { p: Laptop -> if (isSpecMeaningful(p.performance)) p.performance else "—" }
        }
        attrs += "Highlights" to { p: Laptop -> p.summary }
        if (finalists.any { meaningfulWeakness(it) })
            attrs += "Weakness" to { p: Laptop -> if (meaningfulWeakness(p)) p.weakness else "—" }

        val table = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }

        // Header row: empty corner + product names (recommended first, tinted).
        val header = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        header.addView(compareCell("", labelW, bold = true, textColor = muted, bg = surface))
        finalists.forEachIndexed { i, p ->
            val best = i == 0
            header.addView(compareCell(
                (if (best) "★ " else "") + p.name,
                colW, bold = true,
                textColor = if (best) accentDark else ink,
                bg = if (best) paleYellow else surface
            ))
        }
        table.addView(header)

        // Image row: real product thumbnail per column (falls back to placeholder).
        val imgRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        imgRow.addView(compareCell("", labelW, bold = false, textColor = muted, bg = surface))
        finalists.forEachIndexed { i, p ->
            imgRow.addView(FrameLayout(this).apply {
                layoutParams = LinearLayout.LayoutParams(colW, dp(96))
                setPadding(dp(10), dp(8), dp(10), dp(8))
                background = rect(if (i == 0) paleYellow else Color.WHITE, border, 1, 0)
                addView(
                    ProductThumbView(this@MainActivity, p.color, isComputerCategory(), p.imageUrl),
                    FrameLayout.LayoutParams(-1, -1)
                )
            })
        }
        table.addView(imgRow)

        // Attribute rows with zebra striping; recommended column stays tinted.
        attrs.forEachIndexed { rowIdx, (label, value) ->
            val rowBg = if (rowIdx % 2 == 0) Color.WHITE else surface
            val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
            row.addView(compareCell(label, labelW, bold = true, textColor = muted, bg = surface))
            finalists.forEachIndexed { i, p ->
                row.addView(compareCell(
                    value(p), colW, bold = false, textColor = ink,
                    bg = if (i == 0) paleYellow else rowBg
                ))
            }
            table.addView(row)
        }

        // Choose-button row.
        val chooseRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        chooseRow.addView(compareCell("", labelW, bold = false, textColor = muted, bg = surface))
        finalists.forEach { p ->
            chooseRow.addView(LinearLayout(this).apply {
                layoutParams = LinearLayout.LayoutParams(colW, -2)
                setPadding(dp(6), dp(8), dp(6), dp(8))
                background = rect(Color.WHITE, border, 1, 0)
                addView(button("Choose", primary = true) { chooseProduct(p.id) },
                    LinearLayout.LayoutParams(-1, -2))
            })
        }
        table.addView(chooseRow)

        return HorizontalScrollView(this).apply {
            isHorizontalScrollBarEnabled = true
            addView(table)
        }
    }

    private fun compareCell(value: String, width: Int, bold: Boolean, textColor: Int, bg: Int): TextView =
        text(value, 12f, textColor, if (bold) Typeface.BOLD else Typeface.NORMAL).apply {
            gravity = Gravity.CENTER_VERTICAL
            minHeight = dp(44)
            setPadding(dp(8), dp(9), dp(8), dp(9))
            background = rect(bg, border, 1, 0)
            layoutParams = LinearLayout.LayoutParams(width, -2)
        }

    private fun resultScreen(): View {
        if (chosenProductId == null) {
            return scrollColumn().apply {
                addView(text("Your choice", 22f, ink, Typeface.BOLD), fullWidth(bottom = dp(14)))
                addView(card(surface).apply {
                    addView(text("No item chosen yet", 16f, ink, Typeface.BOLD), fullWidth(bottom = dp(6)))
                    addView(text(
                        if (canCompare()) "Your comparison is ready."
                        else "Open a match or comparison when you are ready to choose.",
                        13f,
                        muted
                    ), fullWidth(bottom = dp(12)))
                    addView(button(
                        if (canCompare()) "Compare products" else "Browse matches",
                        primary = true
                    ) { setStage(if (canCompare()) Stage.COMPARE else Stage.RECOMMEND) })
                })
            }
        }
        val product = laptop(chosenProductId!!)
        return scrollColumn().apply {
            addView(text(if (finalConfirmed) "Choice confirmed" else "Your choice", 22f, ink, Typeface.BOLD), fullWidth(bottom = if (finalConfirmed) dp(5) else dp(14)))
            if (finalConfirmed) addView(text("Saved for this session.", 13f, muted), fullWidth(bottom = dp(14)))
            addView(card(if (finalConfirmed) paleGreen else paleYellow, if (finalConfirmed) green else accent).apply {
                addView(LinearLayout(this@MainActivity).apply {
                    orientation = LinearLayout.HORIZONTAL
                    gravity = Gravity.CENTER_VERTICAL
                    addView(ProductThumbView(this@MainActivity, product.color, isComputerCategory(), product.imageUrl), LinearLayout.LayoutParams(dp(90), dp(80)).withMargins(right = dp(12)))
                    addView(LinearLayout(this@MainActivity).apply {
                        orientation = LinearLayout.VERTICAL
                        addView(text(product.name, 19f, ink, Typeface.BOLD), fullWidth(bottom = dp(5)))
                        addView(text("${money(product.price)} • ${matchScore(product)}% match", 14f, ink, Typeface.BOLD), fullWidth(bottom = dp(4)))
                        addView(text("${product.rating}/5 • ${product.reviewCount}", 12f, muted))
                    }, LinearLayout.LayoutParams(0, -2, 1f))
                }, fullWidth(bottom = dp(12)))
                addView(text("Why this is the strongest fit", 15f, ink, Typeface.BOLD), fullWidth(bottom = dp(7)))
                addView(bullets(matchReasons(product).take(3), green))
            }, fullWidth(bottom = dp(12)))
            addView(card().apply {
                addView(text("Matched to", 15f, ink, Typeface.BOLD), fullWidth(bottom = dp(7)))
                addView(text(confirmedSignals().joinToString("  •  "), 12f, muted))
            }, fullWidth(bottom = dp(14)))
            if (!finalConfirmed) {
                addView(button("Confirm choice", primary = true) {
                    finalConfirmed = true
                    chosenProductId = product.id
                    toast("Choice confirmed")
                    render()
                }, fullWidth(bottom = dp(8)))
            } else {
                addView(text("✓ Confirmed - ${product.name}", 15f, green, Typeface.BOLD).apply {
                    gravity = Gravity.CENTER
                    background = rect(paleGreen, green, 1, 10)
                    setPadding(dp(12), dp(13), dp(12), dp(13))
                }, fullWidth(bottom = dp(8)))
            }
            addView(button("Edit needs") {
                beginRefinement()
            }, fullWidth(bottom = dp(8)))
            if (canCompare()) addView(button("Review comparison") { setStage(Stage.COMPARE) })
        }
    }

    private fun detailRow(label: String, value: String): View = LinearLayout(this).apply {
        orientation = LinearLayout.HORIZONTAL
        setPadding(0, dp(8), 0, dp(8))
        addView(text(label, 12f, muted, Typeface.BOLD), LinearLayout.LayoutParams(dp(96), -2))
        addView(text(value, 12f, ink), LinearLayout.LayoutParams(0, -2, 1f))
    }

    private fun bullets(values: List<String>, bulletColor: Int): View = LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        values.forEach { value ->
            addView(LinearLayout(this@MainActivity).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.TOP
                addView(text("•", 16f, bulletColor, Typeface.BOLD), LinearLayout.LayoutParams(dp(18), -2))
                addView(text(value, 13f, ink), LinearLayout.LayoutParams(0, -2, 1f))
            }, fullWidth(bottom = dp(5)))
        }
    }

    private fun buildBottomNav(): View {
        val items = listOf("Search", "Matches", "Compare", "Choice")
        return LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            background = rect(Color.WHITE, border, 1, 0)
            items.forEach { label ->
                val active = when (label) {
                    "Search" -> stage in setOf(Stage.EXPRESS, Stage.CONTEXT, Stage.REFINE)
                    "Matches" -> stage == Stage.RECOMMEND || (stage == Stage.DETAIL && detailOrigin == Stage.RECOMMEND)
                    "Compare" -> stage == Stage.COMPARE || (stage == Stage.DETAIL && detailOrigin == Stage.COMPARE)
                    else -> stage == Stage.RESULT
                }
                addView(text(label, 11f, ink, if (active) Typeface.BOLD else Typeface.NORMAL).apply {
                    gravity = Gravity.CENTER
                    background = rect(if (active) paleYellow else Color.WHITE, border, 0, 0)
                    setOnClickListener { navigateFromTab(label) }
                }, LinearLayout.LayoutParams(0, dp(56), 1f))
            }
        }
    }

    private fun navigateFromTab(label: String) {
        when (label) {
            "Search" -> when {
                stage == Stage.REFINE || analysisPaused || analysisProgress in 1 until analysisSteps.size -> setStage(Stage.REFINE)
                shoppingNeed.isBlank() -> setStage(Stage.EXPRESS)
                else -> setStage(Stage.CONTEXT)
            }
            "Matches" -> setStage(Stage.RECOMMEND)
            "Compare" -> setStage(Stage.COMPARE)
            "Choice" -> setStage(Stage.RESULT)
        }
    }

    private fun submitNeed() {
        if (shoppingNeed.trim().length < 2) {
            toast("请描述你的购物需求，或使用语音/示例任务。")
            return
        }
        shoppingNeed = shoppingNeed.trim()
        invalidateResults()
        // Start from a clean brief so the AI's analysis (not stale laptop
        // defaults) drives the Must-haves / Nice-to-haves for this request.
        mustHaves.clear()
        preferences.clear()
        productCategory = ""
        deriveAssumptionsFromNeed()
        loadCatalogAsync(buildSearchQuery())
        startBriefAnalysis(restart = true)
        setStage(Stage.CONTEXT)
    }

    private fun sendNeedToGpt() {
        val text = shoppingNeed.trim()
        if (text.length < 2) {
            toast("Type a message first")
            return
        }
        if (!gptConnected) {
            toast("Start calling first")
            return
        }
        val played = audioPlayer?.playedMs() ?: 0
        audioPlayer?.interrupt()
        gptSpeaking = false
        realtime?.interruptAssistant(played)
        appendChat("You: $text")
        streamingAssistant = ""
        realtime?.sendText(text)
        render()
    }

    private fun connectGpt() {
        if (gptConnecting || gptConnected) return
        gptConnecting = true
        streamingAssistant = ""
        streamingUserPartial = ""
        gptSpeaking = false
        lastWorkerPlanId = null
        if (audioPlayer == null) audioPlayer = AudioPlayer()
        render()
        val session = RealtimeSession(
            BACKEND_BASE_URL,
            initialSessionId = engineSessionId,
            listener = object : RealtimeSession.Listener {
            override fun onConnectionChanged(connected: Boolean) {
                handler.post {
                    gptConnected = connected
                    gptConnecting = false
                    if (!connected) {
                        stopRecommendationPolling()
                        stopGptVoiceCapture()
                        audioPlayer?.interrupt()
                        leaveDuplexAudioMode()
                        gptSpeaking = false
                        engineSessionId = null
                    } else {
                        ensureMicPermissionThen { startGptVoiceCapture() }
                        startRecommendationPolling()
                    }
                    if (stage == Stage.EXPRESS) render() else toast(if (connected) "Call connected" else "Call ended")
                }
            }

            override fun onSessionReady(sessionId: String) {
                handler.post {
                    engineSessionId = sessionId
                    Log.i("VoiceShop", "Engine session $sessionId")
                }
            }

            override fun onTalkerReady(provider: String, inputAudioSampleRate: Int) {
                handler.post {
                    realtimeTalkerProvider = provider
                    realtimeInputSampleRate = inputAudioSampleRate
                    Log.i("VoiceShop", "Talker provider=$provider inputRate=$inputAudioSampleRate")
                }
            }

            override fun onUserSpeechStarted() {
                handler.post {
                    // Barge-in: stop local playback and truncate unplayed assistant audio.
                    val played = audioPlayer?.playedMs() ?: 0
                    audioPlayer?.interrupt()
                    gptSpeaking = false
                    streamingAssistant = ""
                    realtime?.interruptAssistant(played)
                    scheduleExpressRender(immediate = true)
                }
            }

            override fun onUserSpeechStopped() {
                handler.post { scheduleExpressRender() }
            }

            override fun onUserTranscript(text: String, isFinal: Boolean) {
                handler.post {
                    if (isFinal) {
                        streamingUserPartial = ""
                        if (text.isNotBlank()) {
                            appendChat("You: $text")
                            if (shoppingNeed.isBlank() || gptListening) {
                                shoppingNeed = text
                            }
                        }
                        scheduleExpressRender(immediate = true)
                    } else {
                        streamingUserPartial = text
                        scheduleExpressRender()
                    }
                }
            }

            override fun onAssistantTextDelta(delta: String) {
                handler.post {
                    streamingAssistant += delta
                    scheduleExpressRender()
                }
            }

            override fun onAssistantTextDone(full: String) {
                handler.post {
                    val message = full.trim()
                    streamingAssistant = ""
                    if (message.isNotBlank()) appendChat("GPT: $message")
                    scheduleExpressRender(immediate = true)
                }
            }

            override fun onAssistantAudio(pcm16: ByteArray) {
                gptSpeaking = true
                audioPlayer?.playPcm16(pcm16)
            }

            override fun onAssistantItemStarted(itemId: String) {
                audioPlayer?.startItem(itemId)
                gptSpeaking = true
            }

            override fun onStatus(message: String) {
                handler.post { Log.i("VoiceShop", "Realtime: $message") }
            }

            override fun onError(message: String) {
                handler.post {
                    gptConnecting = false
                    toast(message)
                    if (stage == Stage.EXPRESS) render()
                }
            }
            }
        )
        realtime = session
        session.connect()
    }

    private fun disconnectGpt() {
        stopRecommendationPolling()
        stopGptVoiceCapture()
        audioPlayer?.interrupt()
        leaveDuplexAudioMode()
        realtime?.disconnect()
        realtime = null
        gptConnected = false
        gptConnecting = false
        gptSpeaking = false
        engineSessionId = null
        streamingAssistant = ""
        streamingUserPartial = ""
        render()
    }

    private fun startRecommendationPolling() {
        handler.removeCallbacks(recommendationPollTick)
        handler.postDelayed(recommendationPollTick, 1500L)
    }

    private fun stopRecommendationPolling() {
        handler.removeCallbacks(recommendationPollTick)
    }

    private fun ensureBackendSession(): String {
        engineSessionId?.takeIf { it.isNotBlank() }?.let { return it }
        val url = URL("$BACKEND_BASE_URL/api/v1/session")
        val conn = (url.openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            doOutput = true
            connectTimeout = 5000
            readTimeout = 8000
            setRequestProperty("Content-Type", "application/json; charset=utf-8")
        }
        try {
            conn.outputStream.use { it.write("{}".toByteArray(Charsets.UTF_8)) }
            val body = (if (conn.responseCode in 200..299) conn.inputStream else conn.errorStream)
                ?.bufferedReader()
                ?.use { it.readText() }
                .orEmpty()
            if (conn.responseCode !in 200..299) {
                throw IllegalStateException("Session create failed: HTTP ${conn.responseCode} $body")
            }
            val sid = JSONObject(body).optString("session_id")
            if (sid.isBlank()) throw IllegalStateException("Backend did not return session_id")
            engineSessionId = sid
            return sid
        } finally {
            conn.disconnect()
        }
    }

    private fun uploadSelectedImageAsync(uri: Uri) {
        val uriString = uri.toString()
        imageUploadInFlight = true
        uploadedImageId = null
        render()
        imageExecutor.execute {
            val result = runCatching {
                val sid = ensureBackendSession()
                val jpeg = encodeImageForUpload(uri)
                val payload = JSONObject().apply {
                    put("session_id", sid)
                    put("filename", selectedImageName ?: displayName(uri))
                    put("mime_type", "image/jpeg")
                    put("user_text", shoppingNeed)
                    put("image_base64", Base64.encodeToString(jpeg, Base64.NO_WRAP))
                }
                postImagePayload(payload)
            }
            handler.post {
                if (isDestroyed || selectedImageUri != uriString) return@post
                imageUploadInFlight = false
                result
                    .onSuccess { response ->
                        uploadedImageId = response.optString("image_id").takeIf { it.isNotBlank() }
                        val analysisObj = response.optJSONObject("analysis")
                        val provider = analysisObj?.optString("provider")
                        imageLlmProvider = provider?.takeIf { it.isNotBlank() }
                        imageSearchKeywords = jsonArrayToList(analysisObj?.optJSONArray("search_keywords"))
                            .map { it.trim() }
                            .filter { it.isNotBlank() }
                        imageCategory = analysisObj?.optString("product_category").orEmpty().trim()
                        // Vision analysis is disabled server-side; visual_context
                        // is empty. Do NOT derive any on-device style text — the
                        // image itself will be sent to the LLM with the text.
                        toast("图片已添加，将与文字一起分析")
                        if (imageSearchKeywords.isNotEmpty() || imageCategory.isNotBlank()) {
                            loadCatalogAsync(buildSearchQuery().ifBlank { "laptop" })
                        }
                        render()
                    }
                    .onFailure { error ->
                        Log.e("VoiceShop", "Image upload failed", error)
                        // No local fallback analysis — just report the failure.
                        imageLlmProvider = "error"
                        toast("图片上传失败: ${error.message}")
                        render()
                    }
            }
        }
    }

    // Kick off (or restart) the live streaming analysis shown on the brief page.
    private fun startBriefAnalysis(restart: Boolean) {
        // A non-restart call yields to an in-flight run. An explicit restart
        // (user re-submitting) must always proceed, even if a previous run got
        // stuck (e.g. its stream broke when the backend was restarted); bumping
        // the run id below cancels any in-flight reader.
        if (briefAnalyzing && !restart) return
        if (restart) {
            briefAnalysisText = ""
            val stale = briefAnalysisConn
            briefAnalysisConn = null
            if (stale != null) catalogExecutor.execute { runCatching { stale.disconnect() } }
        }
        briefAnalyzing = true
        briefAnalysisPaused = false
        briefAnalysisDone = false
        val runId = ++briefAnalysisRun
        if (stage == Stage.CONTEXT) render()
        // Send the request text plus the confirmed must-haves / preferences (so
        // the backend LLM can read constraints like "价格在1000元以内" and extract
        // the budget). The uploaded image is added server-side as multimodal
        // input; we still do NOT inject any on-device image description.
        val extras = (mustHaves + preferences).map { it.trim() }.filter { it.isNotEmpty() }
        val text = buildString {
            append(shoppingNeed.trim())
            if (extras.isNotEmpty()) {
                if (isNotEmpty()) append("\n\n")
                append("Preferences/constraints: ")
                append(extras.joinToString("; "))
            }
        }.trim()
        catalogExecutor.execute {
            val outcome = runCatching {
                val sid = ensureBackendSession()
                streamBriefAnalysis(runId, sid, text)
            }
            if (outcome.isFailure && runId == briefAnalysisRun) {
                Log.e("VoiceShop", "brief analysis stream failed", outcome.exceptionOrNull())
                fallbackBlockingAnalysis(runId, text)
            }
        }
    }

    // Pause the stream at any time so the user can add more description. Bumping
    // the run id makes any in-flight reader's updates no-ops.
    private fun pauseBriefAnalysis() {
        if (!briefAnalyzing) return
        briefAnalysisRun++
        briefAnalyzing = false
        briefAnalysisPaused = true
        val conn = briefAnalysisConn
        briefAnalysisConn = null
        catalogExecutor.execute { runCatching { conn?.disconnect() } }
        render()
    }

    // Read the newline-delimited JSON stream and surface reasoning deltas live.
    private fun streamBriefAnalysis(runId: Int, sid: String, text: String) {
        val url = URL("$BACKEND_BASE_URL/api/v1/session/$sid/analyze_stream")
        val conn = (url.openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            doOutput = true
            connectTimeout = 6000
            readTimeout = 60_000
            setRequestProperty("Content-Type", "application/json; charset=utf-8")
        }
        briefAnalysisConn = conn
        try {
            conn.outputStream.use { it.write(JSONObject().put("text", text).toString().toByteArray(Charsets.UTF_8)) }
            val code = conn.responseCode
            if (code !in 200..299) {
                val err = conn.errorStream?.bufferedReader()?.use { it.readText() }.orEmpty()
                throw IllegalStateException("analyze_stream HTTP $code: $err")
            }
            conn.inputStream.bufferedReader().use { reader ->
                while (runId == briefAnalysisRun) {
                    val line = reader.readLine() ?: break
                    val ln = line.trim()
                    if (ln.isEmpty()) continue
                    val obj = runCatching { JSONObject(ln) }.getOrNull() ?: continue
                    when (obj.optString("type")) {
                        "delta" -> Unit
                        "done" -> {
                            val analysis = obj.optJSONObject("analysis")
                            handler.post {
                                if (runId != briefAnalysisRun) return@post
                                briefAnalyzing = false
                                briefAnalysisPaused = false
                                briefAnalysisDone = true
                                applyNeedAnalysis(analysis)
                                startRecommendationPolling()
                                if (stage == Stage.CONTEXT) render()
                            }
                        }
                        "error" -> handler.post {
                            if (runId == briefAnalysisRun) toast("分析出错: ${obj.optString("detail")}")
                        }
                    }
                }
            }
        } catch (e: Exception) {
            if (runId == briefAnalysisRun) throw e
        } finally {
            runCatching { conn.disconnect() }
            if (briefAnalysisConn === conn) briefAnalysisConn = null
            handler.post {
                if (runId == briefAnalysisRun && briefAnalyzing && !briefAnalysisPaused) {
                    briefAnalyzing = false
                    if (stage == Stage.CONTEXT) render()
                }
            }
        }
    }

    // If streaming is unavailable, fall back to the blocking /text analysis.
    private fun fallbackBlockingAnalysis(runId: Int, text: String) {
        catalogExecutor.execute {
            val result = runCatching {
                val sid = ensureBackendSession()
                postSessionText(sid, JSONObject().apply { put("text", text) })
            }
            handler.post {
                if (runId != briefAnalysisRun) return@post
                result
                    .onSuccess { response ->
                        response.optString("visual_context")
                            .takeIf { it.isNotBlank() && selectedImageUri != null }
                            ?.let { imageStyleSignal = it }
                        briefAnalyzing = false
                        briefAnalysisPaused = false
                        briefAnalysisDone = true
                        applyNeedAnalysis(response.optJSONObject("analysis"))
                        startRecommendationPolling()
                        if (stage == Stage.CONTEXT) render()
                    }
                    .onFailure { error ->
                        Log.e("VoiceShop", "Need submit failed", error)
                        briefAnalyzing = false
                        briefAnalysisDone = true
                        toast("需求分析失败: ${error.message}")
                        if (stage == Stage.CONTEXT) render()
                    }
            }
        }
    }

    // Apply the LLM-derived shopping brief returned by /session/{id}/text so
    // the Must-haves / Nice-to-haves reflect the model's understanding.
    private fun applyNeedAnalysis(analysis: JSONObject?) {
        if (analysis == null) return
        val provider = analysis.optString("provider")
        needLlmProvider = provider.takeIf { it.isNotBlank() }
        val apiSummary = analysis.optString("summary").trim()
        briefAnalysisText = if (apiSummary.isNotBlank()) {
            "API 对这段对话的对话总结：\n$apiSummary"
        } else {
            "API 未返回对话总结，已使用基础分析。"
        }
        analysis.optString("category").trim().takeIf { it.isNotEmpty() }?.let { productCategory = it }
        if (provider == "qwen") {
            val must = jsonArrayToList(analysis.optJSONArray("must_haves"))
            val nice = jsonArrayToList(analysis.optJSONArray("nice_to_haves"))
            if (must.isNotEmpty()) { mustHaves.clear(); mustHaves.addAll(must) }
            if (nice.isNotEmpty()) { preferences.clear(); preferences.addAll(nice) }
            // Capture the LLM's budget so the filter/label reflect it even when
            // the must-have wording carries no digits (e.g. "价格实惠").
            analysis.optInt("budget", 0).takeIf { it > 0 }?.let { analyzedBudget = it }
            when (analysis.optString("platform")) {
                "Windows" -> osPreference = "Windows"
                "macOS" -> osPreference = "macOS"
            }
            toast("AI 已分析你的需求")
        } else {
            toast("未连上大模型，使用基础分析")
        }
        render()
    }

    private fun postSessionText(sessionId: String, payload: JSONObject): JSONObject {
        val url = URL("$BACKEND_BASE_URL/api/v1/session/$sessionId/text")
        val conn = (url.openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            doOutput = true
            connectTimeout = 6000
            readTimeout = 30_000
            setRequestProperty("Content-Type", "application/json; charset=utf-8")
        }
        try {
            conn.outputStream.use { it.write(payload.toString().toByteArray(Charsets.UTF_8)) }
            val body = (if (conn.responseCode in 200..299) conn.inputStream else conn.errorStream)
                ?.bufferedReader()
                ?.use { it.readText() }
                .orEmpty()
            if (conn.responseCode !in 200..299) {
                throw IllegalStateException("Need submit HTTP ${conn.responseCode}: $body")
            }
            return JSONObject(body)
        } finally {
            conn.disconnect()
        }
    }

    private fun postImagePayload(payload: JSONObject): JSONObject {
        val url = URL("$BACKEND_BASE_URL/api/v1/image")
        val conn = (url.openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            doOutput = true
            connectTimeout = 8000
            readTimeout = 60_000
            setRequestProperty("Content-Type", "application/json; charset=utf-8")
        }
        try {
            conn.outputStream.use { it.write(payload.toString().toByteArray(Charsets.UTF_8)) }
            val body = (if (conn.responseCode in 200..299) conn.inputStream else conn.errorStream)
                ?.bufferedReader()
                ?.use { it.readText() }
                .orEmpty()
            if (conn.responseCode !in 200..299) {
                throw IllegalStateException("Image upload HTTP ${conn.responseCode}: $body")
            }
            return JSONObject(body)
        } finally {
            conn.disconnect()
        }
    }

    private fun encodeImageForUpload(uri: Uri): ByteArray {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
            throw IllegalArgumentException("Could not read image size")
        }
        var sampleSize = 1
        while (maxOf(bounds.outWidth, bounds.outHeight) / sampleSize > IMAGE_UPLOAD_MAX_SIDE) {
            sampleSize *= 2
        }
        val options = BitmapFactory.Options().apply { inSampleSize = sampleSize }
        val bitmap = contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, options) }
            ?: throw IllegalArgumentException("Could not decode image")
        return try {
            ByteArrayOutputStream().use { out ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, IMAGE_UPLOAD_JPEG_QUALITY, out)
                out.toByteArray()
            }
        } finally {
            bitmap.recycle()
        }
    }

    private fun fetchWorkerRecommendations(sessionId: String): JSONObject? {
        val url = URL("$BACKEND_BASE_URL/api/v1/session/$sessionId/recommendations")
        val conn = (url.openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = 4000
            readTimeout = 6000
        }
        try {
            // 404 means the backend no longer knows this session (e.g. it was
            // restarted). Signal that distinctly so the poller can drop the stale
            // id and re-establish, instead of polling a dead session forever.
            if (conn.responseCode == HttpURLConnection.HTTP_NOT_FOUND) throw SessionGoneException()
            if (conn.responseCode != HttpURLConnection.HTTP_OK) return null
            val body = conn.inputStream.bufferedReader().use { it.readText() }
            val obj = JSONObject(body)
            val bundle = obj.optJSONObject("bundle") ?: return null
            val rankedLen = bundle.optJSONArray("ranked")?.length() ?: 0
            if (rankedLen <= 0) return null
            return bundle
        } finally {
            conn.disconnect()
        }
    }

    private fun applyWorkerBundle(bundle: JSONObject) {
        val planId = bundle.optString("plan_id")
        if (planId.isNotBlank() && planId == lastWorkerPlanId) return
        val ranked = bundle.optJSONArray("ranked") ?: return
        if (ranked.length() == 0) return
        val mapped = ArrayList<Laptop>(ranked.length())
        for (i in 0 until ranked.length()) {
            val item = ranked.optJSONObject(i) ?: continue
            mapped.add(
                Laptop(
                    id = item.optString("id").ifBlank { "w$i" },
                    name = item.optString("name").ifBlank { "Product" },
                    price = item.optInt("price", 0),
                    match = item.optInt("score", 80).coerceIn(60, 99),
                    rating = item.optDouble("rating", 0.0),
                    reviewCount = "Worker match",
                    display = item.optString("display").ifBlank { "Not specified" },
                    performance = item.optString("performance").ifBlank { "Not specified" },
                    battery = "Not specified",
                    weightKg = item.optDouble("weight_kg", 0.0),
                    summary = item.optString("summary").ifBlank {
                        item.optJSONArray("reasons")?.optString(0) ?: "Worker recommendation"
                    },
                    reviewSentiment = "Ranked by Worker runtime",
                    weakness = "See trade-offs in details",
                    reasons = jsonArrayToList(item.optJSONArray("reasons")).ifEmpty { listOf("Worker match") },
                    tradeOffs = listOf("Confirm specs before purchase"),
                    color = catalogColors[i % catalogColors.size],
                    platform = item.optString("platform").ifBlank { "Windows" },
                    imageUrl = item.optString("image_url").trim()
                )
            )
        }
        if (mapped.isEmpty()) return
        lastWorkerPlanId = planId
        laptops = mapped
        catalogSource = "Worker: ${mapped.size} ranked (plan=$planId)"
        val summary = bundle.optString("summary")
        if (summary.isNotBlank()) toast(summary)
        Log.i("VoiceShop", catalogSource)
        if (stage == Stage.RECOMMEND || stage == Stage.EXPRESS) render()
    }

    private fun ensureMicPermissionThen(action: () -> Unit) {
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            action()
        } else {
            pendingMicAction = action
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), REQUEST_MIC)
        }
    }

    private fun startGptVoiceCapture() {
        if (audioCapture?.isRunning() == true) return
        try {
            enterDuplexAudioMode()
            if (audioPlayer == null) audioPlayer = AudioPlayer()
            val capture = AudioCapture(
                sampleRate = realtimeInputSampleRate,
                onPcmChunk = { pcm -> realtime?.appendAudioPcm16(pcm) },
                onEnergy = { rms -> maybeLocalBargeIn(rms) }
            )
            capture.start()
            audioCapture = capture
            gptListening = true
            val providerLabel = if (realtimeTalkerProvider == "openai") "GPT Realtime" else "Qwen Realtime"
            toast("$providerLabel duplex on — speak anytime to interrupt")
            render()
        } catch (error: Exception) {
            gptListening = false
            audioCapture = null
            leaveDuplexAudioMode()
            toast("Mic failed: ${error.message}")
            render()
        }
    }

    /**
     * Emulator barge-in: server VAD often misses the user while speakerphone is blasting.
     * If mic energy spikes while we are playing assistant audio, stop locally immediately.
     */
    private fun maybeLocalBargeIn(rms: Double) {
        if (!gptSpeaking && audioPlayer?.isPlaying() != true) return
        if (rms < LOCAL_BARGE_IN_RMS) return
        val now = SystemClock.elapsedRealtime()
        if (now - lastLocalBargeInAt < 700L) return
        lastLocalBargeInAt = now
        handler.post {
            if (!gptConnected) return@post
            Log.i("VoiceShop", "Local barge-in rms=${rms.toInt()}")
            val played = audioPlayer?.playedMs() ?: 0
            audioPlayer?.interrupt()
            gptSpeaking = false
            streamingAssistant = ""
            realtime?.interruptAssistant(played)
            scheduleExpressRender(immediate = true)
        }
    }

    private fun stopGptVoiceCapture() {
        audioCapture?.stop()
        audioCapture = null
        gptListening = false
    }

    private fun enterDuplexAudioMode() {
        val am = getSystemService(AUDIO_SERVICE) as AudioManager
        previousAudioMode = am.mode
        // MODE_NORMAL + MEDIA playback avoids strong AEC that mutes the mic during TTS.
        am.mode = AudioManager.MODE_NORMAL
        @Suppress("DEPRECATION")
        am.isSpeakerphoneOn = true
    }

    private fun leaveDuplexAudioMode() {
        val am = getSystemService(AUDIO_SERVICE) as AudioManager
        @Suppress("DEPRECATION")
        am.isSpeakerphoneOn = false
        am.mode = previousAudioMode
    }

    private fun appendChat(line: String) {
        chatLines.add(line)
        while (chatLines.size > 40) chatLines.removeAt(0)
    }

    private fun scheduleExpressRender(immediate: Boolean = false) {
        handler.removeCallbacks(streamingRenderTick)
        if (stage != Stage.EXPRESS) return
        if (immediate) render() else handler.postDelayed(streamingRenderTick, 120L)
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_MIC) {
            val action = pendingMicAction
            pendingMicAction = null
            if (grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
                if (action != null) action()
                else if (gptConnected) startGptVoiceCapture() else connectGpt()
            } else {
                toast("Microphone permission required for voice input")
            }
        }
    }

    private fun addRequirement(values: MutableList<String>, raw: String) {
        val value = raw.trim()
        if (value.isBlank()) {
            toast("Enter a requirement first")
            return
        }
        if (values.none { it.equals(value, ignoreCase = true) }) values.add(value)
        requirementsChanged()
    }

    private fun removeRequirement(values: MutableList<String>, index: Int) {
        if (index in values.indices) values.removeAt(index)
        requirementsChanged()
    }

    private fun requirementsChanged() {
        invalidateResults()
        if (stage == Stage.REFINE) {
            analysisPaused = true
            analysisProgress = minOf(analysisProgress, 2)
            toast("Needs updated")
        }
        render()
    }

    private fun contextSignalChanged() {
        invalidateResults()
    }

    private fun beginRefinement() {
        pendingConstraint = ""
        stage = if (shoppingNeed.isBlank()) Stage.EXPRESS else Stage.CONTEXT
        render()
    }

    private fun invalidateResults() {
        selectedIds.clear()
        chosenProductId = null
        finalConfirmed = false
        analysisProgress = 0
        analysisPaused = false
        scrollPositions.keys.removeAll { it in setOf(Stage.REFINE, Stage.RECOMMEND, Stage.DETAIL, Stage.COMPARE, Stage.RESULT) }
    }

    // At least 2 products are needed to show the comparison table.
    private fun canCompare(): Boolean = selectedIds.size >= 2

    private fun toggleFinalist(product: Laptop, checked: Boolean) {
        if (checked && !selectedIds.contains(product.id) && selectedIds.size >= MAX_COMPARE) {
            val current = selectedIds.map(::laptop)
            AlertDialog.Builder(this)
                .setTitle("Compare up to $MAX_COMPARE — replace one?")
                .setItems(current.map { it.name }.toTypedArray()) { _, index ->
                    selectedIds.remove(current[index].id)
                    selectedIds.add(product.id)
                    render()
                }
                .setNegativeButton("Cancel") { _, _ -> render() }
                .show()
            return
        }
        if (checked) selectedIds.add(product.id) else selectedIds.remove(product.id)
        render()
    }

    private fun openDetails(id: String, origin: Stage) {
        detailProductId = id
        detailOrigin = origin
        setStage(Stage.DETAIL)
    }

    private fun chooseProduct(id: String) {
        chosenProductId = id
        detailProductId = id
        finalConfirmed = false
        setStage(Stage.RESULT)
    }

    private fun laptop(id: String): Laptop = laptops.find { it.id == id } ?: laptops.first()

    private fun setStage(next: Stage) {
        stage = next
        render()
    }

    private fun goBack() {
        when (stage) {
            Stage.EXPRESS -> Unit
            Stage.CONTEXT -> setStage(Stage.EXPRESS)
            Stage.REFINE -> setStage(Stage.CONTEXT)
            Stage.RECOMMEND -> setStage(if (shoppingNeed.isBlank()) Stage.EXPRESS else Stage.CONTEXT)
            Stage.DETAIL -> setStage(detailOrigin)
            Stage.COMPARE -> setStage(Stage.RECOMMEND)
            Stage.RESULT -> setStage(
                when {
                    chosenProductId == null -> Stage.RECOMMEND
                    canCompare() -> Stage.COMPARE
                    else -> Stage.DETAIL
                }
            )
        }
    }

    private fun confirmStartOver() {
        AlertDialog.Builder(this)
            .setTitle("Start a new search?")
            .setMessage("This clears the current request, compare list, and choice.")
            .setNegativeButton("Cancel", null)
            .setPositiveButton("New search") { _, _ ->
                shoppingNeed = ""
                selectedImageUri = null
                selectedImageName = null
                imageStyleSignal = null
                uploadedImageId = null
                imageSearchKeywords = emptyList()
                imageCategory = ""
                imageLlmProvider = null
                needLlmProvider = null
                imageUploadInFlight = false
                mustHaves.apply { clear(); addAll(listOf("Budget up to ¥7,000", "Design-studio use", "Strong display", "16GB RAM preferred")) }
                preferences.apply { clear(); addAll(listOf("Portable for campus", "Long battery life", "Silver or neutral style")) }
                osPreference = "No preference"
                touchPreference = "Not required"
                pendingConstraint = ""
                selectedIds.clear()
                sortMode = SortMode.MATCH
                filterUnderBudget = true
                filterOled = false
                filterPortable = false
                detailProductId = "nova"
                detailOrigin = Stage.RECOMMEND
                voiceTarget = VoiceTarget.NEED
                chosenProductId = null
                finalConfirmed = false
                analysisProgress = 0
                analysisPaused = false
                scrollPositions.clear()
                currentScrollView = null
                chatLines.clear()
                streamingAssistant = ""
                streamingUserPartial = ""
                stage = Stage.EXPRESS
                render()
            }
            .show()
    }

    @Suppress("DEPRECATION")
    // Voice input via the backend ASR API (Qwen-Omni). Works even where the
    // on-device recognizer is missing (e.g. emulators). Records mic PCM, uploads
    // it for transcription, then merges the transcript with any typed text.
    // Image signals are already merged through the session + buildSearchQuery.
    private fun launchVoiceInput() {
        ensureMicPermissionThen { startVoiceRecording() }
    }

    private fun startVoiceRecording() {
        if (voiceRecordCapture != null) return
        val buffer = ByteArrayOutputStream()
        val capture = try {
            AudioCapture(onPcmChunk = { pcm -> synchronized(buffer) { buffer.write(pcm) } })
                .also { it.start() }
        } catch (e: Exception) {
            Log.w("VoiceShop", "mic capture failed, falling back to device recognizer", e)
            launchDeviceVoiceInput()
            return
        }
        voiceRecordBuffer = buffer
        voiceRecordCapture = capture
        voiceRecordDialog = AlertDialog.Builder(this)
            .setTitle(if (voiceTarget == VoiceTarget.NEED) "Speak your need" else "Speak to add")
            .setMessage("Listening… tap \"Stop\" when you're done.")
            .setCancelable(false)
            .setPositiveButton("Stop & transcribe") { _, _ -> finishVoiceRecording(cancelled = false) }
            .setNegativeButton("Cancel") { _, _ -> finishVoiceRecording(cancelled = true) }
            .show()
    }

    private fun finishVoiceRecording(cancelled: Boolean) {
        val capture = voiceRecordCapture
        val buffer = voiceRecordBuffer
        voiceRecordCapture = null
        voiceRecordBuffer = null
        voiceRecordDialog?.dismiss()
        voiceRecordDialog = null
        capture?.stop()
        if (cancelled || buffer == null) return
        val pcm = synchronized(buffer) { buffer.toByteArray() }
        if (pcm.size < 3200) { toast("Too short — try again"); return }
        val target = voiceTarget
        toast("Transcribing…")
        catalogExecutor.execute {
            val result = runCatching { postTranscribe(pcm16ToWav(pcm, AudioCapture.SAMPLE_RATE)) }
            handler.post {
                result
                    .onSuccess { text ->
                        if (text.isBlank()) toast("Didn't catch that")
                        else applyTranscript(text, target)
                    }
                    .onFailure { e ->
                        Log.e("VoiceShop", "transcribe failed", e)
                        toast("Voice API failed: ${e.message}")
                    }
            }
        }
    }

    // Merge the transcript with whatever is already typed. For NEED this keeps
    // any existing text (and the session already holds image context), so voice
    // + image + text are combined rather than overwriting each other.
    private fun applyTranscript(text: String, target: VoiceTarget) {
        val spoken = text.trim()
        if (spoken.isEmpty()) return
        if (target == VoiceTarget.NEED) {
            shoppingNeed = listOf(shoppingNeed.trim(), spoken)
                .filter { it.isNotBlank() }
                .joinToString(". ")
            render()
            submitNeed()
        } else {
            mustHaves.add(spoken)
            invalidateResults()
            if (stage == Stage.REFINE) {
                analysisPaused = true
                analysisProgress = minOf(analysisProgress, 2)
            }
            render()
        }
    }

    private fun postTranscribe(wav: ByteArray): String {
        val url = URL("$BACKEND_BASE_URL/api/v1/transcribe")
        val conn = (url.openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            doOutput = true
            connectTimeout = 8000
            readTimeout = 60_000
            setRequestProperty("Content-Type", "application/json; charset=utf-8")
        }
        try {
            val payload = JSONObject().apply {
                put("audio_base64", Base64.encodeToString(wav, Base64.NO_WRAP))
                put("format", "wav")
                put("mime_type", "audio/wav")
            }
            conn.outputStream.use { it.write(payload.toString().toByteArray(Charsets.UTF_8)) }
            val body = (if (conn.responseCode in 200..299) conn.inputStream else conn.errorStream)
                ?.bufferedReader()?.use { it.readText() }.orEmpty()
            if (conn.responseCode !in 200..299) {
                throw IllegalStateException("Transcribe HTTP ${conn.responseCode}: $body")
            }
            return JSONObject(body).optString("text").trim()
        } finally {
            conn.disconnect()
        }
    }

    private fun pcm16ToWav(pcm: ByteArray, sampleRate: Int): ByteArray {
        val channels = 1
        val bits = 16
        val byteRate = sampleRate * channels * bits / 8
        val out = ByteArrayOutputStream(44 + pcm.size)
        fun le32(v: Int) { out.write(v and 0xff); out.write((v shr 8) and 0xff); out.write((v shr 16) and 0xff); out.write((v shr 24) and 0xff) }
        fun le16(v: Int) { out.write(v and 0xff); out.write((v shr 8) and 0xff) }
        out.write("RIFF".toByteArray(Charsets.US_ASCII)); le32(36 + pcm.size); out.write("WAVE".toByteArray(Charsets.US_ASCII))
        out.write("fmt ".toByteArray(Charsets.US_ASCII)); le32(16); le16(1); le16(channels)
        le32(sampleRate); le32(byteRate); le16(channels * bits / 8); le16(bits)
        out.write("data".toByteArray(Charsets.US_ASCII)); le32(pcm.size); out.write(pcm)
        return out.toByteArray()
    }

    // On-device recognizer, kept as a fallback when the mic/API path is unusable.
    private fun launchDeviceVoiceInput() {
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
            putExtra(RecognizerIntent.EXTRA_PROMPT, if (voiceTarget == VoiceTarget.NEED) "What are you looking to buy?" else "What should I add?")
        }
        try {
            startActivityForResult(intent, REQUEST_VOICE)
        } catch (_: Exception) {
            toast("Voice input unavailable")
        }
    }

    @Suppress("DEPRECATION")
    private fun launchImagePicker() {
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "image/*"
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION)
        }
        try {
            startActivityForResult(intent, REQUEST_IMAGE)
        } catch (_: Exception) {
            toast("Image picker unavailable")
        }
    }

    @Suppress("DEPRECATION")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode != RESULT_OK) return
        when (requestCode) {
            REQUEST_VOICE -> {
                val spoken = data?.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS)?.firstOrNull()?.trim().orEmpty()
                if (spoken.isNotBlank()) applyTranscript(spoken, voiceTarget)
            }
            REQUEST_IMAGE -> data?.data?.let { uri ->
                runCatching {
                    contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
                }
                selectedImageUri = uri.toString()
                selectedImageName = displayName(uri)
                imageStyleSignal = null
                uploadedImageId = null
                imageSearchKeywords = emptyList()
                imageCategory = ""
                imageLlmProvider = null
                invalidateResults()
                contextSignalChanged()
                render()
                // Local pixel "style" analysis disabled on purpose: the image is
                // not processed on-device; it is uploaded and later fed to the
                // LLM together with the text (true multimodal).
                // analyzeImageStyleAsync(uri)
                uploadSelectedImageAsync(uri)
            }
        }
    }

    private fun displayName(uri: Uri): String {
        return runCatching {
            contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
                if (cursor.moveToFirst()) cursor.getString(0) else null
            }
        }.getOrNull() ?: "Design reference image"
    }

    private fun analyzeImageStyleAsync(uri: Uri) {
        val uriString = uri.toString()
        imageExecutor.execute {
            val signal = analyzeImageStyle(uri) ?: "Visual reference"
            handler.post {
                if (!isDestroyed && selectedImageUri == uriString && uploadedImageId == null) {
                    imageStyleSignal = signal
                    render()
                }
            }
        }
    }

    private fun analyzeImageStyle(uri: Uri): String? = runCatching {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return@runCatching null
        var sampleSize = 1
        while (maxOf(bounds.outWidth, bounds.outHeight) / sampleSize > 128) sampleSize *= 2
        val options = BitmapFactory.Options().apply { inSampleSize = sampleSize }
        val bitmap = contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, options) }
            ?: return@runCatching null
        var redTotal = 0L
        var greenTotal = 0L
        var blueTotal = 0L
        var count = 0L
        val step = maxOf(1, minOf(bitmap.width, bitmap.height) / 32)
        for (y in 0 until bitmap.height step step) {
            for (x in 0 until bitmap.width step step) {
                val pixel = bitmap.getPixel(x, y)
                if (Color.alpha(pixel) < 64) continue
                redTotal += Color.red(pixel)
                greenTotal += Color.green(pixel)
                blueTotal += Color.blue(pixel)
                count += 1
            }
        }
        bitmap.recycle()
        if (count == 0L) return@runCatching null
        val redAverage = (redTotal / count).toInt()
        val greenAverage = (greenTotal / count).toInt()
        val blueAverage = (blueTotal / count).toInt()
        val brightness = (redAverage + greenAverage + blueAverage) / 3
        val spread = maxOf(redAverage, greenAverage, blueAverage) - minOf(redAverage, greenAverage, blueAverage)
        when {
            brightness >= 205 && spread <= 28 -> "Light neutral"
            brightness <= 78 -> "Dark minimalist"
            redAverage - blueAverage >= 18 -> "Warm-toned"
            blueAverage - redAverage >= 18 -> "Cool-toned"
            else -> "Balanced neutral"
        }
    }.getOrNull()

    private fun scrollColumn(left: Int = dp(18), right: Int = dp(18)): LinearLayout {
        val column = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(left, dp(16), right, dp(24))
        }
        return column.also { wrapper ->
            wrapper.layoutParams = ViewGroup.LayoutParams(-1, -2)
            val scroll = ScrollView(this).apply {
                isFillViewport = true
                setBackgroundColor(Color.WHITE)
            }
            scroll.addView(wrapper)
            wrapper.tag = scroll
        }
    }

    private fun card(color: Int = Color.WHITE, stroke: Int = border): LinearLayout = LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(dp(14), dp(14), dp(14), dp(14))
        background = rect(color, stroke, 1, 12)
    }

    private fun entryCard(title: String, helper: String, action: () -> Unit): LinearLayout = card().apply {
        gravity = Gravity.CENTER_VERTICAL
        isClickable = true
        isFocusable = true
        contentDescription = "$title. $helper"
        addView(text(title, 13f, blue, Typeface.BOLD), fullWidth(bottom = dp(5)))
        addView(text(helper, 10f, muted))
        setOnClickListener { action() }
    }

    private fun button(
        label: String,
        primary: Boolean = false,
        danger: Boolean = false,
        enabled: Boolean = true,
        action: () -> Unit
    ): TextView = text(label, 12f, when { !enabled -> muted; danger -> red; primary -> ink; else -> blue }, Typeface.BOLD).apply {
        gravity = Gravity.CENTER
        minHeight = dp(40)
        setPadding(dp(10), dp(7), dp(10), dp(7))
        background = rect(when { !enabled -> surface; danger -> paleRed; primary -> accent; else -> Color.WHITE }, when { danger -> red; primary -> accentDark; else -> border }, 1, 9)
        isEnabled = enabled
        alpha = if (enabled) 1f else 0.65f
        if (enabled) setOnClickListener { action() }
    }

    private fun text(value: String, size: Float, color: Int, style: Int = Typeface.NORMAL): TextView = TextView(this).apply {
        text = value
        setTextColor(color)
        setTextSize(TypedValue.COMPLEX_UNIT_SP, size)
        typeface = Typeface.create(Typeface.DEFAULT, style)
        includeFontPadding = true
        setLineSpacing(dp(1).toFloat(), 1f)
    }

    private fun watcher(onChange: (String) -> Unit): TextWatcher = object : TextWatcher {
        override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) = Unit
        override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) { onChange(s?.toString().orEmpty()) }
        override fun afterTextChanged(s: Editable?) = Unit
    }

    private fun rect(color: Int, strokeColor: Int, strokeWidth: Int, radius: Int): GradientDrawable = GradientDrawable().apply {
        setColor(color)
        cornerRadius = dp(radius).toFloat()
        if (strokeWidth > 0) setStroke(dp(strokeWidth), strokeColor)
    }

    private fun fullWidth(left: Int = 0, top: Int = 0, right: Int = 0, bottom: Int = 0): LinearLayout.LayoutParams =
        LinearLayout.LayoutParams(-1, -2).withMargins(left, top, right, bottom)

    private fun LinearLayout.LayoutParams.withMargins(left: Int = 0, top: Int = 0, right: Int = 0, bottom: Int = 0): LinearLayout.LayoutParams {
        setMargins(left, top, right, bottom)
        return this
    }

    private fun money(value: Int): String = "¥%,d".format(Locale.US, value)
    private fun toast(message: String) = Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    private fun dp(value: Int): Int = (value * resources.displayMetrics.density + 0.5f).toInt()

    // Generic product thumbnail. Draws a laptop for computer categories and a
    // neutral shopping bag for everything else, so the icon reflects the search.
    private class LaptopPreviewView(
        context: Context,
        private val laptopColor: Int,
        private val isLaptop: Boolean = true
    ) : View(context) {
        private val fill = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = laptopColor; style = Paint.Style.FILL }
        private val screen = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(242, 246, 249); style = Paint.Style.FILL }
        private val stroke = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(63, 72, 82); style = Paint.Style.STROKE; strokeWidth = 2.2f }

        override fun onDraw(canvas: Canvas) {
            super.onDraw(canvas)
            val w = width.toFloat()
            val h = height.toFloat()
            if (isLaptop) drawLaptop(canvas, w, h) else drawBag(canvas, w, h)
        }

        private fun drawLaptop(canvas: Canvas, w: Float, h: Float) {
            val display = RectF(w * .16f, h * .10f, w * .84f, h * .72f)
            canvas.drawRoundRect(display, 8f, 8f, fill)
            canvas.drawRoundRect(RectF(w * .21f, h * .16f, w * .79f, h * .65f), 4f, 4f, screen)
            canvas.drawRoundRect(display, 8f, 8f, stroke)
            val base = Path().apply {
                moveTo(w * .08f, h * .76f)
                lineTo(w * .92f, h * .76f)
                lineTo(w * .80f, h * .90f)
                lineTo(w * .20f, h * .90f)
                close()
            }
            canvas.drawPath(base, fill)
            canvas.drawPath(base, stroke)
        }

        private fun drawBag(canvas: Canvas, w: Float, h: Float) {
            // Bag body.
            val body = RectF(w * .20f, h * .32f, w * .80f, h * .90f)
            canvas.drawRoundRect(body, 10f, 10f, fill)
            canvas.drawRoundRect(body, 10f, 10f, stroke)
            // Handle arc.
            val handle = RectF(w * .34f, h * .12f, w * .66f, h * .52f)
            canvas.drawArc(handle, 180f, 180f, false, stroke)
        }
    }

    /**
     * Product thumbnail: shows the real catalog image (loaded from image_url) with
     * rounded corners, falling back to the drawn placeholder icon while loading or
     * if the network image fails. Zero external dependencies.
     */
    private class ProductThumbView(
        context: Context,
        color: Int,
        isLaptop: Boolean,
        imageUrl: String,
        cornerRadiusDp: Int = 10
    ) : FrameLayout(context) {
        init {
            val radiusPx = cornerRadiusDp * context.resources.displayMetrics.density
            outlineProvider = object : ViewOutlineProvider() {
                override fun getOutline(view: View, outline: Outline) {
                    outline.setRoundRect(0, 0, view.width, view.height, radiusPx)
                }
            }
            clipToOutline = true
            val placeholder = LaptopPreviewView(context, color, isLaptop)
            addView(placeholder, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT))
            val url = imageUrl.trim()
            if (url.startsWith("http")) {
                val img = ImageView(context).apply {
                    scaleType = ImageView.ScaleType.CENTER_CROP
                    visibility = View.GONE
                }
                addView(img, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT))
                ImageLoader.load(url, 512) { bmp ->
                    if (bmp != null) {
                        img.setImageBitmap(bmp)
                        img.visibility = View.VISIBLE
                        placeholder.visibility = View.GONE
                    }
                }
            }
        }
    }

    /** Minimal async network image loader with an in-memory LRU cache (no libraries). */
    private object ImageLoader {
        private val executor = Executors.newFixedThreadPool(3)
        private val main = Handler(Looper.getMainLooper())
        private val cache = object : LinkedHashMap<String, Bitmap>(64, 0.75f, true) {
            override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, Bitmap>?): Boolean = size > 80
        }

        @Synchronized private fun getCached(key: String): Bitmap? = cache[key]
        @Synchronized private fun putCached(key: String, bmp: Bitmap) { cache[key] = bmp }

        fun load(url: String, maxPx: Int, callback: (Bitmap?) -> Unit) {
            getCached(url)?.let { callback(it); return }
            executor.execute {
                val bmp = try { fetch(url, maxPx) } catch (e: Exception) { null }
                if (bmp != null) putCached(url, bmp)
                main.post { callback(bmp) }
            }
        }

        private fun fetch(urlStr: String, maxPx: Int): Bitmap? {
            val conn = (URL(urlStr).openConnection() as HttpURLConnection).apply {
                connectTimeout = 8000
                readTimeout = 8000
                instanceFollowRedirects = true
                setRequestProperty("User-Agent", "VoiceShop/1.0 (Android)")
            }
            try {
                if (conn.responseCode !in 200..299) return null
                val bytes = conn.inputStream.use { it.readBytes() }
                if (bytes.isEmpty()) return null
                val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
                var sample = 1
                val longest = maxOf(bounds.outWidth, bounds.outHeight)
                while (longest > 0 && longest / sample > maxPx) sample *= 2
                val opts = BitmapFactory.Options().apply { inSampleSize = sample }
                return BitmapFactory.decodeByteArray(bytes, 0, bytes.size, opts)
            } finally {
                conn.disconnect()
            }
        }
    }

    companion object {
        private const val REQUEST_VOICE = 1001
        private const val REQUEST_IMAGE = 1002
        private const val REQUEST_MIC = 1003
        // Android emulator → host machine loopback (verified working).
        // Physical device on same Wi-Fi: use http://192.168.31.199:8000 instead.
        // private const val BACKEND_BASE_URL = "http://127.0.0.1:8000"
        private const val BACKEND_BASE_URL = "http://10.0.2.2:8000"
        /** Mic RMS above this while assistant is talking → local barge-in (emulator-friendly). */
        private const val LOCAL_BARGE_IN_RMS = 1800.0
        private const val IMAGE_UPLOAD_MAX_SIDE = 1024
        private const val IMAGE_UPLOAD_JPEG_QUALITY = 82
        /** How many products can be added to the side-by-side comparison table. */
        private const val MAX_COMPARE = 4
        private const val SAMPLE_NEED = "I want lightweight running shoes for daily jogging, budget around ¥500, breathable and from a well-reviewed brand."
    }
}

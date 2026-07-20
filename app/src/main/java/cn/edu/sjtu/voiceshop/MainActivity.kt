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
        val platform: String = "Windows"
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
                    catalogSource = "Backend: ${loaded.size} laptops (q=\"$query\")"
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
            name = obj.optString("name").ifBlank { "Unnamed laptop" },
            price = obj.optInt("price", 0),
            match = (rating / 5.0 * 100).toInt().coerceIn(60, 99),
            rating = rating,
            reviewCount = reviewCount,
            display = obj.optString("display").orEmptyText(),
            performance = obj.optString("performance").orEmptyText(),
            battery = obj.optString("battery").orEmptyText(),
            weightKg = obj.optDouble("weight_kg", 0.0),
            summary = obj.optString("summary").ifNullOrBlank { "Laptop option${if (store.isNotBlank()) " from $store" else ""}" },
            reviewSentiment = obj.optString("review_sentiment").ifNullOrBlank { "No aggregated review summary available." },
            weakness = obj.optString("weakness").ifNullOrBlank { "No notable weaknesses in the catalog data." },
            reasons = jsonArrayToList(obj.optJSONArray("reasons")).ifEmpty { listOf("Matches your laptop search") },
            tradeOffs = jsonArrayToList(obj.optJSONArray("trade_offs")).ifEmpty { listOf("Limited spec detail available for this item.") },
            color = catalogColors[index % catalogColors.size],
            platform = obj.optString("platform").ifNullOrBlank { "Windows" }
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
    private var imageUploadInFlight = false
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
    private var detailProductId = "nova"
    private var detailOrigin = Stage.RECOMMEND
    private var chosenProductId: String? = null
    private var finalConfirmed = false
    private var analysisProgress = 0
    private var analysisPaused = false
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
    private var gptConnected = false
    private var gptConnecting = false
    private var gptListening = false
    private var gptSpeaking = false
    private var engineSessionId: String? = null
    private var lastWorkerPlanId: String? = null
    private val chatLines = mutableListOf<String>()
    private var streamingAssistant = ""
    private var streamingUserPartial = ""
    private var previousAudioMode = AudioManager.MODE_NORMAL
    @Volatile private var lastLocalBargeInAt = 0L
    private val streamingRenderTick = Runnable {
        if (stage == Stage.EXPRESS) render()
    }
    private val recommendationPollTick = object : Runnable {
        override fun run() {
            val sid = engineSessionId
            if (sid.isNullOrBlank()) return
            catalogExecutor.execute {
                val bundle = runCatching { fetchWorkerRecommendations(sid) }.getOrNull()
                handler.post {
                    if (isDestroyed || engineSessionId.isNullOrBlank()) return@post
                    if (bundle != null) applyWorkerBundle(bundle)
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
        loadCatalogAsync(if (shoppingNeed.isBlank()) "laptop" else shoppingNeed)
        selectedImageUri?.takeIf { imageStyleSignal == null }?.let { analyzeImageStyleAsync(Uri.parse(it)) }
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
        leaveDuplexAudioMode()
        stopGptVoiceCapture()
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
            addView(text("Find your next laptop", 22f, ink, Typeface.BOLD), fullWidth(bottom = dp(8)))
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
            addView(entryCard("Design student preset", "¥7,000 • color-accurate display • portable") {
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
                hint = "Budget, apps, screen, portability…"
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
                else -> "Talk with the shopping assistant about your laptop needs."
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
                    imageUploadInFlight -> "Uploading to vision worker..."
                    uploadedImageId != null && imageStyleSignal != null -> "LLM visual read: $imageStyleSignal"
                    imageStyleSignal != null -> "Local visual read: $imageStyleSignal"
                    else -> "Reading image..."
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
            addView(text(shoppingNeed.ifBlank { SAMPLE_NEED }, 14f, ink), fullWidth(bottom = if (selectedImageName != null) dp(8) else 0))
            selectedImageName?.let {
                addView(text("Image: $it", 12f, muted), fullWidth(bottom = dp(4)))
                addView(text(
                    imageStyleSignal?.let { signal -> "Visual preference - $signal" } ?: "Reading image...",
                    12f,
                    blue,
                    Typeface.BOLD
                ))
            }
            addView(button("Edit request") { setStage(Stage.EXPRESS) }, fullWidth(top = dp(10)))
        }, fullWidth(bottom = dp(14)))
        addView(requirementSection("Must-haves", mustHaves, paleYellow, "Add a must-have"), fullWidth(bottom = dp(14)))
        addView(requirementSection("Nice-to-haves", preferences, paleBlue, "Add a preference"), fullWidth(bottom = dp(14)))
        addView(clarificationSection(), fullWidth(bottom = dp(14)))
        addView(LinearLayout(this@MainActivity).apply {
            orientation = LinearLayout.HORIZONTAL
            addView(button("Add by voice") {
                voiceTarget = VoiceTarget.REQUIREMENT
                launchVoiceInput()
            }, LinearLayout.LayoutParams(0, dp(48), 1f).withMargins(right = dp(8)))
            addView(button(if (selectedImageUri == null) "Add reference" else "Change reference") { launchImagePicker() }, LinearLayout.LayoutParams(0, dp(48), 1f))
        }, fullWidth(bottom = dp(14)))
        val imageReady = selectedImageUri == null || (!imageUploadInFlight && imageStyleSignal != null)
        addView(button(
            if (imageReady) "Find matching laptops" else "Reading reference...",
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
                addView(text(if (analysisPaused) "Search paused" else if (analysisProgress == analysisSteps.size) "${filteredLaptops().size} matches ready" else "Searching laptops", 15f, ink, Typeface.BOLD), LinearLayout.LayoutParams(0, -2, 1f))
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
        addView(text(if (shoppingNeed.isBlank()) "Explore laptops" else "Matches for you", 22f, ink, Typeface.BOLD), fullWidth(bottom = dp(4)))
        addView(text(
            if (shoppingNeed.isBlank()) "Popular options for design work"
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
                "${selectedIds.size} of 2 in compare${if (hiddenSelections > 0) " • $hiddenSelections hidden" else ""}",
                13f,
                if (selectedIds.size == 2) green else blue,
                Typeface.BOLD
            ))
        }, fullWidth(bottom = dp(10)))
        if (visible.isEmpty()) {
            addView(card(surface).apply {
                addView(text("No laptops match all active filters.", 14f, ink, Typeface.BOLD), fullWidth(bottom = dp(10)))
                addView(button("Clear filters", primary = true) {
                    filterUnderBudget = false; filterOled = false; filterPortable = false; render()
                })
            }, fullWidth(bottom = dp(12)))
        } else visible.forEach { addView(productCard(it), fullWidth(bottom = dp(10))) }
        addView(button(
            if (selectedIds.size == 2) "Compare 2 laptops" else "Choose 2 to compare",
            primary = selectedIds.size == 2,
            enabled = selectedIds.size == 2
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

    private fun filteredLaptops(): List<Laptop> {
        val filtered = laptops.filter { laptop ->
            (!filterUnderBudget || laptop.price <= budgetLimit()) &&
                (!filterOled || laptop.display.contains("OLED")) &&
                (!filterPortable || laptop.weightKg < 1.5) &&
                (osPreference == "No preference" || laptop.platform == osPreference)
        }
        return when (sortMode) {
            SortMode.MATCH -> filtered.sortedByDescending { matchScore(it) }
            SortMode.PRICE -> filtered.sortedBy { it.price }
            SortMode.RATING -> filtered.sortedByDescending { it.rating }
        }
    }

    private fun budgetLimit(): Int {
        val confirmedBudgets = budgetsIn(mustHaves.joinToString(" "))
        return (confirmedBudgets.ifEmpty { budgetsIn(shoppingNeed) }).minOrNull() ?: 7_000
    }

    private fun budgetsIn(text: String): List<Int> = Regex("(?<!\\d)(\\d[\\d,]{3,5})(?!\\d)")
        .findAll(text)
        .mapNotNull { it.groupValues[1].replace(",", "").toIntOrNull() }
        .filter { it in 3_000..50_000 }
        .toList()

    private fun confirmedSignals(): List<String> = buildList {
        addAll(mustHaves)
        addAll(preferences)
        add("OS: $osPreference")
        add("Touch: $touchPreference")
        imageStyleSignal?.let { add("Image style: $it") }
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
            preferences.add("Portable for campus")
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

    private fun productCard(product: Laptop): View = card(if (selectedIds.contains(product.id)) paleYellow else Color.WHITE, if (selectedIds.contains(product.id)) accent else border).apply {
        addView(LinearLayout(this@MainActivity).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.TOP
            addView(LaptopPreviewView(this@MainActivity, product.color), LinearLayout.LayoutParams(dp(76), dp(70)).withMargins(right = dp(10)))
            addView(LinearLayout(this@MainActivity).apply {
                orientation = LinearLayout.VERTICAL
                addView(text(product.name, 16f, ink, Typeface.BOLD), fullWidth(bottom = dp(3)))
                addView(text("${money(product.price)}  •  ${product.rating}/5 (${product.reviewCount})", 12f, ink), fullWidth(bottom = dp(4)))
                addView(text("${matchScore(product)}% match  •  ${product.summary}", 12f, green, Typeface.BOLD))
            }, LinearLayout.LayoutParams(0, -2, 1f))
        }, fullWidth(bottom = dp(8)))
        addView(text("Display: ${product.display}\nPerformance: ${product.performance}", 12f, muted), fullWidth(bottom = dp(9)))
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
                addView(LaptopPreviewView(this@MainActivity, product.color), LinearLayout.LayoutParams(dp(104), dp(92)).withMargins(right = dp(14)))
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
                addView(detailRow("Platform", product.platform))
                addView(detailRow("Display", product.display))
                addView(detailRow("Performance", product.performance))
                addView(detailRow("Battery", product.battery))
                addView(detailRow("Weight", "${product.weightKg} kg"))
                addView(detailRow("Reviews", "${product.rating}/5 from ${product.reviewCount}"))
                addView(detailRow("Review insight", product.reviewSentiment))
            }, fullWidth(bottom = dp(12)))
            addView(card(paleRed, Color.rgb(236, 188, 188)).apply {
                addView(text("Trade-offs to consider", 16f, ink, Typeface.BOLD), fullWidth(bottom = dp(8)))
                addView(bullets(product.tradeOffs, red))
            }, fullWidth(bottom = dp(14)))
            addView(button(if (selectedIds.contains(product.id)) "Remove from compare" else "Add to compare", primary = !selectedIds.contains(product.id)) {
                toggleFinalist(product, !selectedIds.contains(product.id))
            }, fullWidth(bottom = dp(8)))
            addView(button("Choose ${product.name}", primary = true) { chooseProduct(product.id) }, fullWidth(bottom = dp(8)))
            addView(button(if (selectedIds.size == 2) "Compare 2 laptops" else "Back to matches") {
                setStage(if (selectedIds.size == 2) Stage.COMPARE else Stage.RECOMMEND)
            })
        }
    }

    private fun compareScreen(): View {
        val finalists = selectedIds.mapNotNull { id -> laptops.find { it.id == id } }.sortedByDescending { matchScore(it) }
        return scrollColumn(left = dp(14), right = dp(14)).apply {
            addView(text("Compare laptops", 22f, ink, Typeface.BOLD), fullWidth(bottom = dp(14)))
            if (finalists.size != 2) {
                addView(card(surface).apply {
                    addView(text(
                        if (finalists.isEmpty()) "Your compare list is empty" else "One laptop ready",
                        16f,
                        ink,
                        Typeface.BOLD
                    ), fullWidth(bottom = dp(6)))
                    addView(text(
                        if (finalists.isEmpty()) "Add two laptops from Matches to see them side by side."
                        else "${finalists.first().name} is in compare. Add one more laptop.",
                        13f,
                        muted
                    ), fullWidth(bottom = dp(12)))
                    addView(button("Browse matches", primary = true) { setStage(Stage.RECOMMEND) })
                })
            } else {
                finalists.forEachIndexed { index, product ->
                    addView(comparisonCard(product, index == 0), fullWidth(bottom = dp(12)))
                }
                addView(button("Edit comparison") { setStage(Stage.RECOMMEND) })
            }
        }
    }

    private fun comparisonCard(product: Laptop, recommended: Boolean): View = card(if (recommended) paleYellow else Color.WHITE, if (recommended) accent else border).apply {
        addView(LinearLayout(this@MainActivity).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            addView(LinearLayout(this@MainActivity).apply {
                orientation = LinearLayout.VERTICAL
                addView(text(product.name, 17f, ink, Typeface.BOLD), fullWidth(bottom = dp(3)))
                addView(text(if (recommended) recommendationSubtitle(product) else product.summary, 12f, if (recommended) accentDark else muted))
            }, LinearLayout.LayoutParams(0, -2, 1f))
            addView(LaptopPreviewView(this@MainActivity, product.color), LinearLayout.LayoutParams(dp(72), dp(62)))
        }, fullWidth(bottom = dp(10)))
        addView(detailRow("Price", "${money(product.price)}${if (product.price <= budgetLimit()) " - within budget" else " - over budget"}"))
        addView(detailRow("Platform", product.platform))
        addView(detailRow("Display", product.display))
        addView(detailRow("Performance", product.performance))
        addView(detailRow("Reviews", "${product.rating}/5 from ${product.reviewCount}. ${product.reviewSentiment}"))
        addView(detailRow("Weakness", product.weakness))
        addView(LinearLayout(this@MainActivity).apply {
            orientation = LinearLayout.HORIZONTAL
            addView(button("Choose ${product.name.substringBefore(" ")}", primary = true) { chooseProduct(product.id) }, LinearLayout.LayoutParams(0, dp(46), 1f).withMargins(right = dp(8)))
            addView(button("View details") { openDetails(product.id, Stage.COMPARE) }, LinearLayout.LayoutParams(0, dp(46), 1f))
        }, fullWidth(top = dp(10)))
    }

    private fun resultScreen(): View {
        if (chosenProductId == null) {
            return scrollColumn().apply {
                addView(text("Your choice", 22f, ink, Typeface.BOLD), fullWidth(bottom = dp(14)))
                addView(card(surface).apply {
                    addView(text("No laptop chosen yet", 16f, ink, Typeface.BOLD), fullWidth(bottom = dp(6)))
                    addView(text(
                        if (selectedIds.size == 2) "Your comparison is ready."
                        else "Open a match or comparison when you are ready to choose.",
                        13f,
                        muted
                    ), fullWidth(bottom = dp(12)))
                    addView(button(
                        if (selectedIds.size == 2) "Compare laptops" else "Browse matches",
                        primary = true
                    ) { setStage(if (selectedIds.size == 2) Stage.COMPARE else Stage.RECOMMEND) })
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
                    addView(LaptopPreviewView(this@MainActivity, product.color), LinearLayout.LayoutParams(dp(90), dp(80)).withMargins(right = dp(12)))
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
            if (selectedIds.size == 2) addView(button("Review comparison") { setStage(Stage.COMPARE) })
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
        if (shoppingNeed.trim().length < 8) {
            toast("Describe your laptop need, use voice, or choose the sample task.")
            return
        }
        shoppingNeed = shoppingNeed.trim()
        invalidateResults()
        deriveAssumptionsFromNeed()
        loadCatalogAsync(shoppingNeed)
        submitNeedToBackendAsync(shoppingNeed)
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
                        val visual = response.optString("visual_context")
                        if (visual.isNotBlank()) {
                            imageStyleSignal = visual
                            if (preferences.none { it.contains("image reference", ignoreCase = true) }) {
                                preferences.add("Image reference: ${visual.take(90)}")
                            }
                        }
                        toast("Image analyzed")
                        render()
                    }
                    .onFailure { error ->
                        Log.e("VoiceShop", "Image upload failed", error)
                        if (imageStyleSignal == null) imageStyleSignal = analyzeImageStyle(uri) ?: "Visual reference"
                        toast("Image upload failed: ${error.message}")
                        render()
                    }
            }
        }
    }

    private fun submitNeedToBackendAsync(text: String) {
        catalogExecutor.execute {
            val result = runCatching {
                val sid = ensureBackendSession()
                val payload = JSONObject().apply { put("text", text) }
                postSessionText(sid, payload)
            }
            handler.post {
                result
                    .onSuccess { response ->
                        response.optString("visual_context")
                            .takeIf { it.isNotBlank() && selectedImageUri != null }
                            ?.let { imageStyleSignal = it }
                        startRecommendationPolling()
                        Log.i("VoiceShop", "Need submitted to engine session ${engineSessionId.orEmpty()}")
                    }
                    .onFailure { error ->
                        Log.e("VoiceShop", "Need submit failed", error)
                        toast("Backend need update failed: ${error.message}")
                    }
            }
        }
    }

    private fun postSessionText(sessionId: String, payload: JSONObject): JSONObject {
        val url = URL("$BACKEND_BASE_URL/api/v1/session/$sessionId/text")
        val conn = (url.openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            doOutput = true
            connectTimeout = 6000
            readTimeout = 15_000
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
                    name = item.optString("name").ifBlank { "Laptop" },
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
                    platform = item.optString("platform").ifBlank { "Windows" }
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
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), REQUEST_MIC)
        }
    }

    private fun startGptVoiceCapture() {
        if (audioCapture?.isRunning() == true) return
        try {
            enterDuplexAudioMode()
            if (audioPlayer == null) audioPlayer = AudioPlayer()
            val capture = AudioCapture(
                onPcmChunk = { pcm -> realtime?.appendAudioPcm16(pcm) },
                onEnergy = { rms -> maybeLocalBargeIn(rms) }
            )
            capture.start()
            audioCapture = capture
            gptListening = true
            toast("Live duplex on — speak anytime to interrupt")
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
            if (grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
                if (gptConnected) startGptVoiceCapture() else connectGpt()
            } else {
                toast("Microphone permission required to start calling")
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

    private fun toggleFinalist(product: Laptop, checked: Boolean) {
        if (checked && !selectedIds.contains(product.id) && selectedIds.size >= 2) {
            val current = selectedIds.map(::laptop)
            AlertDialog.Builder(this)
                .setTitle("Replace a laptop?")
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
                    selectedIds.size == 2 -> Stage.COMPARE
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
    private fun launchVoiceInput() {
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
            putExtra(RecognizerIntent.EXTRA_PROMPT, if (voiceTarget == VoiceTarget.NEED) "What matters in your next laptop?" else "What should I add?")
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
                if (spoken.isNotBlank()) {
                    if (voiceTarget == VoiceTarget.NEED) {
                        shoppingNeed = spoken
                        invalidateResults()
                    } else {
                        mustHaves.add(spoken)
                        invalidateResults()
                        if (stage == Stage.REFINE) {
                            analysisPaused = true
                            analysisProgress = minOf(analysisProgress, 2)
                        }
                    }
                    render()
                }
            }
            REQUEST_IMAGE -> data?.data?.let { uri ->
                runCatching {
                    contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
                }
                selectedImageUri = uri.toString()
                selectedImageName = displayName(uri)
                imageStyleSignal = null
                uploadedImageId = null
                invalidateResults()
                contextSignalChanged()
                render()
                analyzeImageStyleAsync(uri)
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

    private class LaptopPreviewView(context: Context, private val laptopColor: Int) : View(context) {
        private val fill = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = laptopColor; style = Paint.Style.FILL }
        private val screen = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(242, 246, 249); style = Paint.Style.FILL }
        private val stroke = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(63, 72, 82); style = Paint.Style.STROKE; strokeWidth = 2.2f }

        override fun onDraw(canvas: Canvas) {
            super.onDraw(canvas)
            val w = width.toFloat()
            val h = height.toFloat()
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
        private const val SAMPLE_NEED = "I need a laptop for design studies with a budget of ¥7,000 RMB. I care about display quality, 16GB memory, and portability."
    }
}

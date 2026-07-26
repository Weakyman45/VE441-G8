package cn.edu.sjtu.voiceshop

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import kotlin.math.sqrt

/**
 * Streams microphone PCM16 mono for Realtime input_audio_buffer.append.
 *
 * [onEnergy] reports PCM RMS per chunk so the UI can do local barge-in when server VAD
 * is deaf (common on emulators with speaker echo / weak host mic).
 */
class AudioCapture(
    private val sampleRate: Int = DEFAULT_SAMPLE_RATE,
    private val onPcmChunk: (ByteArray) -> Unit,
    private val onEnergy: ((Double) -> Unit)? = null
) {
    private var record: AudioRecord? = null
    @Volatile private var running = false
    private var captureThread: Thread? = null
    @Volatile private var chunksSent = 0

    fun start() {
        if (running) return
        val rate = sampleRate
        val minBuf = AudioRecord.getMinBufferSize(
            rate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )
        if (minBuf <= 0) {
            throw IllegalStateException("AudioRecord buffer unavailable ($minBuf)")
        }
        val recorder = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            rate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            minBuf * 2
        )
        if (recorder.state != AudioRecord.STATE_INITIALIZED) {
            recorder.release()
            throw IllegalStateException("AudioRecord failed to initialize")
        }
        record = recorder
        running = true
        chunksSent = 0
        recorder.startRecording()
        captureThread = Thread({
            val chunk = ByteArray(rate * 2 / 10)
            while (running) {
                val n = record?.read(chunk, 0, chunk.size) ?: break
                if (n > 0) {
                    val copy = if (n == chunk.size) chunk.copyOf() else chunk.copyOf(n)
                    val energy = rmsPcm16(copy)
                    onEnergy?.invoke(energy)
                    onPcmChunk(copy)
                    chunksSent++
                    if (chunksSent == 1 || chunksSent % 50 == 0) {
                        Log.i(TAG, "Sent $chunksSent audio chunks (rms=${energy.toInt()})")
                    }
                } else if (n < 0) {
                    Log.w(TAG, "AudioRecord read error: $n")
                    break
                }
            }
        }, "audio-capture").also { it.start() }
        Log.i(TAG, "AudioCapture started @ ${rate}Hz")
    }

    fun stop() {
        running = false
        captureThread?.join(500)
        captureThread = null
        try {
            record?.stop()
        } catch (_: Exception) {
        }
        record?.release()
        record = null
        Log.i(TAG, "AudioCapture stopped after $chunksSent chunks")
    }

    fun isRunning(): Boolean = running

    companion object {
        private const val TAG = "AudioCapture"
        const val DEFAULT_SAMPLE_RATE = 16_000
        const val QWEN_SAMPLE_RATE = 16_000
        const val OPENAI_SAMPLE_RATE = 24_000
        const val SAMPLE_RATE = DEFAULT_SAMPLE_RATE

        fun rmsPcm16(pcm: ByteArray): Double {
            if (pcm.size < 2) return 0.0
            var sum = 0.0
            var count = 0
            var i = 0
            while (i + 1 < pcm.size) {
                val sample = (pcm[i].toInt() and 0xff) or (pcm[i + 1].toInt() shl 8)
                val signed = sample.toShort().toInt()
                sum += (signed * signed).toDouble()
                count++
                i += 2
            }
            if (count == 0) return 0.0
            return sqrt(sum / count)
        }
    }
}

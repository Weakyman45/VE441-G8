package cn.edu.sjtu.voiceshop

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.util.Log
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/**
 * Streams PCM16 mono @ 24 kHz from Realtime `response.output_audio.delta` events.
 * Tracks how many milliseconds have been written so barge-in can truncate accurately.
 */
class AudioPlayer {
    private var track: AudioTrack? = null
    private val playing = AtomicBoolean(false)
    private val writtenFrames = AtomicLong(0)
    @Volatile private var activeItemId: String? = null

    fun startItem(itemId: String?) {
        activeItemId = itemId
        writtenFrames.set(0)
        ensureTrack()
    }

    fun playPcm16(pcm: ByteArray) {
        if (pcm.isEmpty()) return
        val audio = ensureTrack() ?: return
        if (!playing.get()) {
            playing.set(true)
            runCatching { audio.play() }
        }
        var offset = 0
        while (offset < pcm.size) {
            val written = audio.write(pcm, offset, pcm.size - offset)
            if (written <= 0) break
            offset += written
            writtenFrames.addAndGet(written / 2L) // 16-bit samples
        }
    }

    /** Milliseconds of audio already handed to AudioTrack for the active item. */
    fun playedMs(): Int {
        val frames = writtenFrames.get()
        return ((frames * 1000L) / SAMPLE_RATE).toInt().coerceAtLeast(0)
    }

    fun activeItemId(): String? = activeItemId

    fun isPlaying(): Boolean = playing.get()

    /** Stop playback immediately (barge-in / disconnect). */
    fun interrupt() {
        playing.set(false)
        writtenFrames.set(0)
        activeItemId = null
        val audio = track ?: return
        runCatching {
            audio.pause()
            audio.flush()
            audio.stop()
        }
    }

    fun release() {
        interrupt()
        runCatching { track?.release() }
        track = null
    }

    private fun ensureTrack(): AudioTrack? {
        track?.let { return it }
        val minBuf = AudioTrack.getMinBufferSize(
            SAMPLE_RATE,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )
        if (minBuf <= 0) {
            Log.e(TAG, "AudioTrack buffer unavailable: $minBuf")
            return null
        }
        val audio = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    // MEDIA (not VOICE_COMMUNICATION) so hardware AEC is less likely to
                    // wipe the user's mic while the assistant is talking — critical for barge-in.
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setSampleRate(SAMPLE_RATE)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build()
            )
            .setBufferSizeInBytes(minBuf * 2)
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()
        track = audio
        return audio
    }

    companion object {
        private const val TAG = "AudioPlayer"
        const val SAMPLE_RATE = 24_000
    }
}

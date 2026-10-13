import { useEffect, useMemo, useRef, useState } from "react"
import {
  Activity,
  ArrowUp,
  BrainCircuit,
  CircleStop,
  Cpu,
  Feather,
  KeyRound,
  Link2,
  LoaderCircle,
  MessageSquareText,
  Mic,
  MicOff,
  RefreshCw,
  SlidersHorizontal,
  Square,
  Trash2,
  Volume2,
} from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import {
  getHealth,
  listModels,
  listVoices,
  streamChat,
  synthesizeSpeech,
  transcribeAudio,
  type ChatMessage,
  type HealthResponse,
  type StreamChatResult,
  type VoiceInfo,
} from "@/lib/api"
import { activeRequests, supportsCacheSlots } from "@/lib/runtime"
import { persistPublicSettings, stored } from "@/lib/storage"
import { cn } from "@/lib/utils"

const message = (role: ChatMessage["role"], content: string): ChatMessage => ({
  id: crypto.randomUUID(),
  role,
  content,
})

export default function App() {
  const [baseUrl, setBaseUrl] = useState(() => {
    const saved = stored(localStorage, "colibri.baseUrl", "http://127.0.0.1:8001/v1")
    return saved.includes(":8000") ? "http://127.0.0.1:8001/v1" : saved
  })
  const [apiKey, setApiKey] = useState("")
  const [models, setModels] = useState<string[]>([])
  const [model, setModel] = useState(() => stored(localStorage, "colibri.model", "olmoe"))
  const [temperature, setTemperature] = useState(0.7)
  const [maxTokens, setMaxTokens] = useState(256)
  const [thinking, setThinking] = useState(false)
  const [cacheSlot, setCacheSlot] = useState(0)
  const [conversations, setConversations] = useState<Record<number, ChatMessage[]>>({ 0: [] })
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [healthError, setHealthError] = useState("")
  const [lastRun, setLastRun] = useState<StreamChatResult | null>(null)
  const [draft, setDraft] = useState("")
  const [loading, setLoading] = useState(false)
  const [connecting, setConnecting] = useState(false)
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState("")

  // Voice & Speech State
  const [voices, setVoices] = useState<VoiceInfo[]>([])
  const [voice, setVoice] = useState(() => stored(localStorage, "colibri.voice", "local"))
  const [autoPlayAudio, setAutoPlayAudio] = useState(
    () => stored(localStorage, "colibri.autoPlay", "false") === "true",
  )
  const [recording, setRecording] = useState(false)
  const [transcribing, setTranscribing] = useState(false)
  const [playingId, setPlayingId] = useState<string | null>(null)

  const abortRef = useRef<AbortController | null>(null)
  const probeRef = useRef<AbortController | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const currentAudioRef = useRef<HTMLAudioElement | null>(null)

  const messages = conversations[cacheSlot] || []
  const kvSlots = Math.max(1, health?.kv_slots || 1)
  const active = activeRequests(health)
  const capacity = health?.scheduler?.capacity || kvSlots
  const failures = health?.scheduler
    ? health.scheduler.rejected + health.scheduler.timed_out + health.scheduler.cancelled
    : 0

  const updateMessages = (next: ChatMessage[] | ((current: ChatMessage[]) => ChatMessage[])) =>
    setConversations((current) => ({
      ...current,
      [cacheSlot]: typeof next === "function" ? next(current[cacheSlot] || []) : next,
    }))

  useEffect(() => {
    persistPublicSettings(localStorage, baseUrl, model)
    localStorage.setItem("colibri.voice", voice)
    localStorage.setItem("colibri.autoPlay", String(autoPlayAudio))
  }, [baseUrl, model, voice, autoPlayAudio])

  useEffect(() => {
    setConnected(false)
    setHealth(null)
    setHealthError("")
  }, [baseUrl, apiKey])

  useEffect(() => {
    void connect()
    return () => {
      probeRef.current?.abort()
      abortRef.current?.abort()
      stopAudio()
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
        mediaRecorderRef.current.stop()
      }
    }
  }, [])

  useEffect(() => {
    if (!connected) return
    let disposed = false
    const poll = async () => {
      if (document.visibilityState === "hidden") return
      try {
        const result = await getHealth(baseUrl, apiKey)
        if (!disposed) {
          setHealth(result)
          setHealthError("")
        }
      } catch (cause) {
        if (!disposed)
          setHealthError(cause instanceof Error ? cause.message : "Runtime metrics unavailable")
      }
    }
    const timer = window.setInterval(() => void poll(), 5000)
    return () => {
      disposed = true
      window.clearInterval(timer)
    }
  }, [apiKey, baseUrl, connected])

  useEffect(() => {
    if (cacheSlot >= kvSlots) setCacheSlot(0)
  }, [cacheSlot, kvSlots])

  useEffect(() => {
    setLastRun(null)
  }, [cacheSlot])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const stopAudio = () => {
    if (currentAudioRef.current) {
      currentAudioRef.current.pause()
      currentAudioRef.current = null
    }
    setPlayingId(null)
  }

  const playSpeech = async (id: string, text: string) => {
    if (playingId === id) {
      stopAudio()
      return
    }
    stopAudio()
    if (!text.trim()) return

    setPlayingId(id)
    try {
      const blob = await synthesizeSpeech(baseUrl, apiKey, text, voice)
      const url = URL.createObjectURL(blob)
      const audio = new Audio(url)
      currentAudioRef.current = audio
      audio.onended = () => {
        setPlayingId(null)
        URL.revokeObjectURL(url)
        currentAudioRef.current = null
      }
      audio.onerror = () => {
        setPlayingId(null)
        URL.revokeObjectURL(url)
        currentAudioRef.current = null
      }
      await audio.play()
    } catch (cause) {
      setPlayingId(null)
      setError(cause instanceof Error ? cause.message : "Speech playback failed.")
    }
  }

  const toggleRecording = async () => {
    if (recording) {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
        mediaRecorderRef.current.stop()
      }
      return
    }

    try {
      setError("")
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      audioChunksRef.current = []

      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "audio/webm"
      const recorder = new MediaRecorder(stream, { mimeType })
      mediaRecorderRef.current = recorder

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) audioChunksRef.current.push(event.data)
      }

      recorder.onstop = async () => {
        setRecording(false)
        stream.getTracks().forEach((track) => track.stop())
        const audioBlob = new Blob(audioChunksRef.current, { type: recorder.mimeType })
        if (audioBlob.size > 0) {
          setTranscribing(true)
          try {
            const transcribed = await transcribeAudio(baseUrl, apiKey, audioBlob, "recording.webm")
            if (transcribed.trim()) {
              setDraft((current) => (current ? `${current} ${transcribed}` : transcribed))
            }
          } catch (cause) {
            setError(cause instanceof Error ? cause.message : "Audio transcription failed.")
          } finally {
            setTranscribing(false)
          }
        }
      }

      recorder.start(250)
      setRecording(true)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Microphone access denied or unavailable.")
      setRecording(false)
    }
  }

  const connect = async () => {
    probeRef.current?.abort()
    const controller = new AbortController()
    probeRef.current = controller
    setConnecting(true)
    setError("")
    try {
      const [found, voiceList] = await Promise.all([
        listModels(baseUrl, apiKey, controller.signal),
        listVoices(baseUrl, apiKey, controller.signal),
      ])
      setModels(found)
      if (found.length && !found.includes(model)) setModel(found[0])
      if (voiceList.length) setVoices(voiceList)
      setConnected(true)
      try {
        setHealth(await getHealth(baseUrl, apiKey, controller.signal))
        setHealthError("")
      } catch (cause) {
        if (!controller.signal.aborted) {
          setHealth(null)
          setHealthError(cause instanceof Error ? cause.message : "Runtime metrics unavailable")
        }
      }
    } catch (cause) {
      if (controller.signal.aborted) return
      setConnected(false)
      setError(cause instanceof Error ? cause.message : "Could not reach the server.")
    } finally {
      if (probeRef.current === controller) {
        probeRef.current = null
        setConnecting(false)
      }
    }
  }

  const canSend = useMemo(() => draft.trim() && model && !loading, [draft, loading, model])

  const send = async () => {
    const content = draft.trim()
    if (!content || loading) return
    const user = message("user", content)
    const assistant = message("assistant", "")
    const history = [...messages, user]
    setDraft("")
    setError("")
    updateMessages([...history, assistant])
    setLoading(true)
    const controller = new AbortController()
    abortRef.current = controller

    let accumulatedText = ""
    try {
      const result = await streamChat({
        baseUrl,
        apiKey,
        model,
        messages: history,
        temperature,
        maxTokens,
        enableThinking: thinking,
        cacheSlot: supportsCacheSlots(health) ? cacheSlot : undefined,
        signal: controller.signal,
        onDelta: (delta) => {
          accumulatedText += delta
          updateMessages((current) =>
            current.map((item) =>
              item.id === assistant.id ? { ...item, content: item.content + delta } : item,
            ),
          )
        },
      })
      setLastRun(result)
      setConnected(true)
      if (autoPlayAudio && accumulatedText.trim()) {
        void playSpeech(assistant.id, accumulatedText)
      }
    } catch (cause) {
      if (controller.signal.aborted) {
        updateMessages((current) =>
          current.filter((item) => item.id !== assistant.id || item.content),
        )
      } else {
        setError(cause instanceof Error ? cause.message : "Generation failed.")
        updateMessages((current) =>
          current.filter((item) => item.id !== assistant.id || item.content),
        )
      }
    } finally {
      abortRef.current = null
      setLoading(false)
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-row">
          <div className="brand-mark">
            <Feather className="size-5" />
          </div>
          <div>
            <h1>colibrì</h1>
            <p>local giant, tiny footprint</p>
          </div>
        </div>

        <section className="side-section">
          <div className="section-title">
            <Link2 className="size-3.5" /> Connection
          </div>
          <label>
            API endpoint
            <Input value={baseUrl} onChange={(event) => setBaseUrl(event.target.value)} />
          </label>
          <label>
            API key
            <div className="relative">
              <KeyRound className="field-icon" />
              <Input
                className="pl-9"
                type="password"
                value={apiKey}
                placeholder="optional"
                onChange={(event) => setApiKey(event.target.value)}
              />
            </div>
            <span className="field-help">Kept in memory only · sent to this endpoint</span>
          </label>
          <Button type="button" variant="secondary" onClick={connect} disabled={connecting}>
            {connecting ? (
              <LoaderCircle className="size-4 animate-spin" />
            ) : (
              <RefreshCw className="size-4" />
            )}
            Probe server
          </Button>
          <div className={cn("connection-state", connected && "connected")} aria-live="polite">
            <span />
            {connected ? "Engine reachable" : "Not connected"}
          </div>
        </section>

        <section className="side-section runtime-section" aria-live="polite">
          <div className="section-title">
            <Activity className="size-3.5" /> Runtime
          </div>
          {health?.scheduler ? (
            <>
              <div className="runtime-grid">
                <div>
                  <span>Active</span>
                  <strong>
                    {active}
                    <small> / {capacity}</small>
                  </strong>
                </div>
                <div>
                  <span>Queued</span>
                  <strong>
                    {health.scheduler.queued}
                    <small> / {health.scheduler.max_queue}</small>
                  </strong>
                </div>
                <div>
                  <span>Completed</span>
                  <strong>{health.scheduler.completed}</strong>
                </div>
                <div>
                  <span>Failures</span>
                  <strong>{failures}</strong>
                </div>
              </div>
              <div className="runtime-foot">
                <span className="runtime-dot" /> Scheduler online <code>{kvSlots} KV</code>
              </div>
            </>
          ) : (
            <p className="runtime-unavailable">
              {connected
                ? healthError || "Runtime metrics unavailable"
                : "Probe the server to inspect runtime state."}
            </p>
          )}
        </section>

        <section className="side-section">
          <div className="section-title">
            <SlidersHorizontal className="size-3.5" /> Inference
          </div>
          <label>
            Model
            <select value={model} onChange={(event) => setModel(event.target.value)}>
              {models.length ? (
                models.map((id) => <option key={id}>{id}</option>)
              ) : (
                <option>{model}</option>
              )}
            </select>
          </label>
          {health?.kv_slots && health.kv_slots > 1 ? (
            <label>
              KV session
              <select
                value={cacheSlot}
                onChange={(event) => setCacheSlot(Number(event.target.value))}
                disabled={loading}
              >
                {Array.from({ length: kvSlots }, (_, slot) => (
                  <option key={slot} value={slot}>
                    Session {slot + 1}
                  </option>
                ))}
              </select>
              <span className="field-help">Isolated context · conversation follows selected slot</span>
            </label>
          ) : null}
          <label>
            <span className="label-line">
              <span>Temperature</span>
              <code>{temperature.toFixed(1)}</code>
            </span>
            <input
              className="range"
              type="range"
              min="0"
              max="2"
              step="0.1"
              value={temperature}
              onChange={(event) => setTemperature(Number(event.target.value))}
            />
          </label>
          <label>
            Max output tokens
            <Input
              type="number"
              min={1}
              max={4096}
              value={maxTokens}
              onChange={(event) => {
                const value = Number(event.target.value)
                if (Number.isFinite(value))
                  setMaxTokens(Math.min(4096, Math.max(1, Math.round(value))))
              }}
            />
          </label>
          <button
            type="button"
            className={cn("toggle-row", thinking && "active")}
            aria-pressed={thinking}
            onClick={() => setThinking((value) => !value)}
          >
            <span>
              <BrainCircuit className="size-4" /> Reasoning
            </span>
            <i>
              <b />
            </i>
          </button>
        </section>

        <section className="side-section">
          <div className="section-title">
            <Volume2 className="size-3.5" /> Speech & Audio
          </div>
          <label>
            Voice model
            <select value={voice} onChange={(event) => setVoice(event.target.value)}>
              {voices.length ? (
                voices.map((v) => (
                  <option key={v.id} value={v.id}>
                    {v.name}
                  </option>
                ))
              ) : (
                <option value="local">Piper Lessac (100% Offline Local)</option>
              )}
            </select>
          </label>
          <button
            type="button"
            className={cn("toggle-row", autoPlayAudio && "active")}
            aria-pressed={autoPlayAudio}
            onClick={() => setAutoPlayAudio((val) => !val)}
          >
            <span>
              <Volume2 className="size-4" /> Auto-speak replies
            </span>
            <i>
              <b />
            </i>
          </button>
        </section>

        <div className="sidebar-foot">
          <Cpu className="size-3.5" />
          <span>OpenAI-compatible transport</span>
        </div>
      </aside>

      <main className="chat-panel">
        <header className="topbar">
          <div>
            <span className="eyebrow">ACTIVE MODEL</span>
            <strong>{model}</strong>
          </div>
          <div className="top-actions">
            {lastRun?.queueWaitMs != null ? (
              <Badge>queue {Math.round(lastRun.queueWaitMs)}ms</Badge>
            ) : null}
            <Badge>
              <Activity className="size-3" /> slot {cacheSlot + 1}
            </Badge>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => updateMessages([])}
              disabled={!messages.length || loading}
            >
              <Trash2 className="size-3.5" /> Clear
            </Button>
          </div>
        </header>

        <div className="conversation">
          {!messages.length ? (
            <div className="empty-state">
              <div className="orb">
                <Feather />
              </div>
              <span className="eyebrow">COLIBRÌ ENGINE</span>
              <h2>
                Ask the giant.
                <br />
                <em>Keep the machine yours.</em>
              </h2>
              <p>
                Connect to a local colibrì server and stream responses directly from your hardware.
                Now with CUDA voice input & speech synthesis.
              </p>
              <div className="suggestions">
                {[
                  "Explain how expert routing works",
                  "Write a small C benchmark",
                  "Compare RAM and VRAM caching",
                ].map((item) => (
                  <button key={item} onClick={() => setDraft(item)}>
                    {item}
                    <ArrowUp className="size-3.5 rotate-45" />
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="message-list">
              {messages.map((item) => (
                <article key={item.id} className={cn("message", item.role)}>
                  <div className="avatar">
                    {item.role === "user" ? "Y" : <Feather className="size-4" />}
                  </div>
                  <div>
                    <div className="message-meta">
                      <span>{item.role === "user" ? "You" : "colibrì"}</span>
                      {item.role === "assistant" && item.content ? (
                        <button
                          type="button"
                          className={cn("audio-btn", playingId === item.id && "playing")}
                          onClick={() => void playSpeech(item.id, item.content)}
                          title={playingId === item.id ? "Stop playback" : "Read aloud (TTS)"}
                          aria-label={playingId === item.id ? "Stop playback" : "Read aloud"}
                        >
                          {playingId === item.id ? (
                            <Square className="size-3" />
                          ) : (
                            <Volume2 className="size-3" />
                          )}
                        </button>
                      ) : null}
                    </div>
                    <div className="message-body">
                      {item.content || (
                        <span className="typing" aria-label="Generating">
                          <i />
                          <i />
                          <i />
                        </span>
                      )}
                    </div>
                  </div>
                </article>
              ))}
              <div ref={bottomRef} />
            </div>
          )}
        </div>

        <div className="composer-wrap">
          {error && (
            <div className="error-banner" role="alert">
              {error}
            </div>
          )}
          <div className="composer">
            <Textarea
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              placeholder="Message colibrì, or click mic to dictate…"
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                  event.preventDefault()
                  void send()
                }
              }}
            />
            <div className="composer-foot">
              <span>
                <MessageSquareText className="size-3.5" /> Enter to send · Shift+Enter for newline
              </span>
              <div className="composer-controls">
                <Button
                  type="button"
                  variant={recording ? "destructive" : "ghost"}
                  size="icon"
                  className={cn("rec-btn", recording && "recording")}
                  disabled={loading || transcribing}
                  onClick={() => void toggleRecording()}
                  title={recording ? "Stop recording (Transcribe)" : "Voice input (STT)"}
                  aria-label="Voice input"
                >
                  {transcribing ? (
                    <LoaderCircle className="size-4 animate-spin text-primary" />
                  ) : recording ? (
                    <MicOff className="size-4 animate-pulse" />
                  ) : (
                    <Mic className="size-4" />
                  )}
                </Button>
                {loading ? (
                  <Button
                    variant="destructive"
                    size="icon"
                    aria-label="Stop generation"
                    onClick={() => abortRef.current?.abort()}
                  >
                    <CircleStop className="size-4" />
                  </Button>
                ) : (
                  <Button
                    size="icon"
                    aria-label="Send message"
                    disabled={!canSend || recording}
                    onClick={() => void send()}
                  >
                    <ArrowUp className="size-4" />
                  </Button>
                )}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}

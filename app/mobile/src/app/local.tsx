import { useCallback, useEffect, useRef, useState } from 'react'
import { ScrollView, View } from 'react-native'
import { router } from 'expo-router'
import { SafeAreaView } from 'react-native-safe-area-context'
import { useKeepAwake } from 'expo-keep-awake'
import { Body, Brand, Button, Field, Heading, Notice, layout, useColors } from '../components/ui'
import { useSession } from '../lib/session'
import {
  benchmarkLocalModel,
  cancelLocalDownload,
  downloadLocalModel,
  localCapabilities,
  localChat,
  localRuntimeAvailable,
  localStatus,
  removeLocalModel,
  startLocalModel,
  stopLocalModel,
  type LocalCapabilities,
  type LocalBenchmarkResult,
  type LocalChatMessage,
  type LocalRuntimeStatus,
} from '../lib/local-runtime'

function bytes(value: number) {
  if (!Number.isFinite(value) || value <= 0) return '0 GB'
  return `${(value / 1024 ** 3).toFixed(1)} GB`
}

function percent(done: number, total: number) {
  return total > 0 ? `${Math.min(100, done * 100 / total).toFixed(1)}%` : '0%'
}

export default function LocalTensorFold() {
  useKeepAwake('tensorfold-local-runtime')
  const c = useColors()
  const session = useSession()
  const [capabilities, setCapabilities] = useState<LocalCapabilities | null>(null)
  const [status, setStatus] = useState<LocalRuntimeStatus | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [prompt, setPrompt] = useState('')
  const [messages, setMessages] = useState<LocalChatMessage[]>([])
  const [tokens, setTokens] = useState('256')
  const temperature = '0'
  const [lastRate, setLastRate] = useState<number | null>(null)
  const [benchmark, setBenchmark] = useState<LocalBenchmarkResult | null>(null)
  const cancellationRequested = useRef(false)

  const refresh = useCallback(async () => {
    if (!localRuntimeAvailable) return
    try {
      const [nextCapabilities, nextStatus] = await Promise.all([localCapabilities(), localStatus()])
      setCapabilities(nextCapabilities); setStatus(nextStatus)
      if (nextStatus.error) setError(nextStatus.error)
    } catch (failure) { setError((failure as Error).message) }
  }, [])

  useEffect(() => {
    void refresh()
    const timer = setInterval(() => void refresh(), status?.state === 'starting' || status?.transfer.state === 'downloading' || status?.transfer.state === 'verifying' ? 1000 : 3000)
    return () => clearInterval(timer)
  }, [refresh, status?.state, status?.transfer.state])

  const run = async (name: string, operation: () => Promise<unknown>) => {
    if (busy) return
    setBusy(name); setError(null)
    try { await operation(); await refresh() }
    catch (failure) { setError((failure as Error).message || 'The local operation failed.') }
    finally { setBusy(null) }
  }

  const send = async () => {
    const text = prompt.trim()
    if (!text || status?.state !== 'ready' || busy) return
    const maxTokens = Number(tokens); const sampling = Number(temperature)
    if (!Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 1024 || sampling !== 0) {
      setError('Use 1–1,024 output tokens. This qualified native engine currently decodes greedily at temperature 0.'); return
    }
    const next = [...messages, { role: 'user' as const, content: text }]
    cancellationRequested.current = false
    setMessages(next); setPrompt(''); setBusy('chat'); setError(null)
    try {
      const result = await localChat([{ role: 'system', content: 'You are a concise helpful assistant running privately on this Android phone.' }, ...next], maxTokens, sampling)
      setMessages([...next, { role: 'assistant', content: result.content }])
      const rate = result.runtime?.tokens_per_second
      setLastRate(typeof rate === 'number' ? rate : null)
    } catch (failure) {
      if (!cancellationRequested.current) setError((failure as Error).message || 'Local generation failed.')
    } finally {
      cancellationRequested.current = false
      setBusy(null)
      void refresh()
    }
  }

  const runBenchmark = async () => {
    if (busy || status?.state !== 'ready') return
    cancellationRequested.current = false
    setBusy('benchmark'); setError(null); setBenchmark(null)
    try { setBenchmark(await benchmarkLocalModel()) }
    catch (failure) {
      if (!cancellationRequested.current) setError((failure as Error).message || 'Local benchmark failed.')
    } finally {
      cancellationRequested.current = false
      setBusy(null)
      void refresh()
    }
  }

  const stop = async () => {
    const isCancelling = busy === 'chat' || busy === 'benchmark'
    if (isCancelling) cancellationRequested.current = true
    setBusy('stop'); setError(null)
    try { await stopLocalModel(); await refresh() }
    catch (failure) { setError((failure as Error).message || 'Could not stop the local engine.') }
    finally {
      if (!isCancelling) cancellationRequested.current = false
      setBusy(null)
    }
  }

  if (!localRuntimeAvailable) return <SafeAreaView style={[layout.page, { backgroundColor: c.bg }]}><View style={layout.content}><Brand /><Heading>Run on this phone</Heading><Notice error>Install the Android runtime build of TensorFold. The iOS and web companion builds do not contain the local engine.</Notice><Button secondary onPress={() => session.session ? router.replace('/') : router.push('/pair')}>{session.session ? 'Workspace' : 'Connect to Mac'}</Button></View></SafeAreaView>

  const transfer = status?.transfer
  return <SafeAreaView style={[layout.page, { backgroundColor: c.bg }]}><ScrollView keyboardShouldPersistTaps="handled" contentContainerStyle={[layout.content, { paddingTop: 24, paddingBottom: 48 }]}>
    <View style={layout.row}><Brand /><Button secondary onPress={() => session.session ? router.replace('/') : router.push('/pair')}>{session.session ? 'Workspace' : 'Connect to Mac'}</Button></View>
    <View style={{ gap: 8 }}><Heading>Run on this phone</Heading><Body muted>Models, prompts, replies, benchmarks, and caches remain inside this app. No Mac, Python, Termux, root, or command line.</Body></View>
    {error && <Notice error>{error}</Notice>}
    <View style={[layout.card, { backgroundColor: c.card, borderColor: c.line }]}>
      <View style={layout.row}><Heading small>Device</Heading><Body style={{ color: capabilities?.qualification.startsWith('qualified-') ? c.positive : c.ink2 }}>{capabilities?.qualification || 'Checking'}</Body></View>
      <Body>{capabilities ? `${capabilities.manufacturer} ${capabilities.model}` : 'Inspecting hardware'}</Body>
      {capabilities && <><Body muted>{capabilities.socManufacturer} {capabilities.socModel} · selected {capabilities.backend} · Vulkan {capabilities.vulkanVersion}</Body><Body muted>{bytes(capabilities.totalRamBytes)} RAM · {bytes(capabilities.availableStorageBytes)} free · {capabilities.batteryTemperatureC?.toFixed(1) ?? '—'} C</Body></>}
      {capabilities && !capabilities.runtimeInstalled && <Notice error>This APK is missing: {capabilities.missingRuntime.join(', ')}.</Notice>}
      {capabilities?.vulkanAvailable && !capabilities.vulkanQualified && <Notice>Vulkan is installed but disabled: {capabilities.vulkanReason}</Notice>}
      {capabilities?.platform === 'pixel-tensor' && <Body muted>Pixel Tensor detected. The exact portable CPU path is qualified on Pixel 10 Pro Fold; GPU and TPU acceleration remain separately gated.</Body>}
      {capabilities?.platform === 'qualcomm' && <Body muted>{`${capabilities.htpAvailableToApp ? `QNN HTP ${capabilities.qnnHtp.architecture || ''} fixed synthetic projection passed (${capabilities.qnnHtp.elapsedMs.toFixed(2)} ms). Checkpoint fixture ${capabilities.qnnHtp.checkpointProjection.available ? 'passed' : `failed: ${capabilities.qnnHtp.checkpointProjection.reason}`}. Fixed 128x128 EXL3 block ${capabilities.qnnHtp.fullBlockProjection.available ? `passed (${capabilities.qnnHtp.fullBlockProjection.elapsedMs.toFixed(2)} ms, max error ${capabilities.qnnHtp.fullBlockProjection.maxAbsoluteError.toFixed(4)})` : `failed: ${capabilities.qnnHtp.fullBlockProjection.reason}`}.` : capabilities.htpReason} Live app-owned checkpoint projection ${capabilities.qnnHtp.liveCheckpointProjection.available ? `passed in process (${capabilities.qnnHtp.liveCheckpointProjection.elapsedMs.toFixed(2)} ms, max error ${capabilities.qnnHtp.liveCheckpointProjection.maxAbsoluteError.toFixed(4)})` : `unavailable: ${capabilities.qnnHtp.liveCheckpointProjection.reason}`}. TensorFold model inference remains on CPU pending retained-token parity.`}</Body>}
    </View>
    <View style={{ gap: 12 }}><View style={layout.row}><Heading small>Models</Heading><Body muted>{status?.state || 'checking'}</Body></View>
      {transfer && ['downloading', 'verifying'].includes(transfer.state) && <Notice>{transfer.state === 'verifying' ? 'Verifying SHA-256' : `Downloading ${percent(transfer.bytesDone, transfer.bytesTotal)}`} · {bytes(transfer.bytesDone)} / {bytes(transfer.bytesTotal)}</Notice>}
      {status?.models.map((model) => <View key={model.id} style={[layout.card, { backgroundColor: c.card, borderColor: c.line }]}>
        <View style={layout.row}><Body style={{ fontWeight: '600', flex: 1 }}>{model.id}</Body><Body style={{ color: model.installed ? c.positive : c.ink2 }}>{model.installed ? 'Verified' : bytes(model.downloadBytes)}</Body></View>
        <Body muted>{model.repo} · downloads {bytes(model.downloadBytes)} · uses about {bytes(model.installedBytes)} · needs {bytes(model.minimumRamBytes)} RAM</Body>
        {model.reason && <Notice error>{model.reason === 'insufficient-memory' ? 'This model exceeds the device memory qualification.' : model.reason === 'requires-snapdragon' ? 'This paged MoE model is currently qualified only on Snapdragon.' : 'Free more app storage for the verified model and its cache reserve.'}</Notice>}
        {!model.installed
          ? <Button disabled={!model.fits || !!busy || transfer?.state === 'downloading'} busy={busy === `download-${model.id}`} onPress={() => void run(`download-${model.id}`, () => downloadLocalModel(model.id))}>Download and verify</Button>
          : <View style={{ gap: 8 }}><Button disabled={!!busy || !model.fits || !capabilities?.runtimeInstalled || status?.state === 'starting'} busy={busy === `start-${model.id}` || (status?.state === 'starting' && status.modelId === model.id)} onPress={() => void run(`start-${model.id}`, () => startLocalModel(model.id))}>{status?.state === 'ready' && status.modelId === model.id ? 'Restart model' : 'Load model'}</Button><Button secondary danger disabled={!!busy || (status?.modelId === model.id && status.state !== 'stopped')} onPress={() => void run(`remove-${model.id}`, () => removeLocalModel(model.id))}>Remove model files</Button></View>}
      </View>)}
      {transfer && ['downloading', 'verifying'].includes(transfer.state) && <Button secondary onPress={() => void run('cancel-download', cancelLocalDownload)}>Pause download</Button>}
      {status?.state !== 'stopped' && <Button secondary danger busy={busy === 'stop'} onPress={() => void stop()}>{busy === 'chat' || busy === 'benchmark' ? 'Cancel current work' : 'Stop local engine'}</Button>}
    </View>
    <View style={{ gap: 12 }}>
      <Heading small>Quick benchmark</Heading>
      <Body muted>One excluded warm-up and three measured 64-token greedy runs per workload. This is an on-phone check, not the five-repeat 256-token community-v2 comparison.</Body>
      <Button disabled={status?.state !== 'ready' || !!busy} busy={busy === 'benchmark'} onPress={() => void runBenchmark()}>{busy === 'benchmark' ? 'Benchmarking locally' : 'Run quick benchmark'}</Button>
      {benchmark && <View style={[layout.card, { backgroundColor: c.card, borderColor: c.line }]}>
        <Body style={{ fontWeight: '600' }}>{benchmark.modelId} · {benchmark.backend}</Body>
        {(['code', 'chat'] as const).map((name) => <View key={name} style={layout.row}><Body>{name === 'code' ? 'Code' : 'Chat'}</Body><Body>{benchmark.workloads[name].medianTokensPerSecond.toFixed(2)} tok/s · {benchmark.workloads[name].medianPrefillSeconds.toFixed(2)} s prefill</Body></View>)}
        <Body muted>{benchmark.temperatureBeforeC ?? '—'} C → {benchmark.temperatureAfterC ?? '—'} C</Body>
      </View>}
    </View>
    <View style={{ gap: 12 }}><View style={layout.row}><Heading small>Private chat</Heading>{lastRate != null && <Body muted>{lastRate.toFixed(1)} tok/s</Body>}</View>
      {messages.length === 0 ? <Body muted>Load a verified model, then start a conversation. Chat history stays in memory and clears when this screen closes.</Body> : messages.map((message, index) => <View key={`${index}-${message.role}`} style={[layout.card, { backgroundColor: message.role === 'user' ? c.card2 : c.card, borderColor: c.line }]}><Body style={{ fontWeight: '600' }}>{message.role === 'user' ? 'You' : 'TensorFold'}</Body><Body>{message.content}</Body></View>)}
      <Field label="Message" value={prompt} onChangeText={setPrompt} multiline placeholder="Ask the on-device model" editable={!busy && status?.state === 'ready'} />
      <View style={{ flexDirection: 'row', gap: 10 }}><View style={{ flex: 1 }}><Field label="Output tokens" value={tokens} onChangeText={setTokens} keyboardType="number-pad" /></View><View style={{ flex: 1 }}><Field label="Temperature" value={temperature} editable={false} /></View></View>
      <Button disabled={!prompt.trim() || status?.state !== 'ready' || !!busy} busy={busy === 'chat'} onPress={() => void send()}>{busy === 'chat' ? 'Generating locally' : 'Send'}</Button>
      {messages.length > 0 && <Button secondary onPress={() => { setMessages([]); setLastRate(null) }}>Clear chat</Button>}
    </View>
  </ScrollView></SafeAreaView>
}

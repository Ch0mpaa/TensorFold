import { requireOptionalNativeModule } from 'expo'
import { Platform } from 'react-native'

export type LocalBackend = 'vulkan' | 'cpu' | 'unsupported'
export type LocalEngineState = 'stopped' | 'starting' | 'ready' | 'unhealthy' | 'failed'
export type TransferState = 'idle' | 'downloading' | 'verifying' | 'ready' | 'cancelled' | 'failed'

export interface QnnProjection { available: boolean; elapsedMs: number; maxAbsoluteError: number; tolerance: number; reason: string | null }
export interface QnnPersistentProjection {
  available: boolean; createMs: number; finalizeMs: number; firstExecuteMs: number; warmMedianMs: number; warmLastMs: number
  maxAbsoluteError: number; tolerance: number; reason: string | null
}
export interface QnnServiceProjection extends QnnPersistentProjection {
  serviceProcess: string; backend: string; servicePid: number; graphCount: number; repeats: 100; status: number
}
export interface QnnCapability {
  available: boolean; architecture: string | null; probe: string; elapsedMs: number; maxAbsoluteError: number; tolerance: number; reason: string | null
  checkpointProjection: QnnProjection; fullBlockProjection: QnnProjection; liveCheckpointProjection: QnnProjection; liveFullProjection: QnnProjection
  persistentFullProjection: QnnPersistentProjection; serviceFullProjection: QnnServiceProjection
}

export interface LocalCapabilities {
  platform: 'qualcomm' | 'pixel-tensor' | 'android-arm64'
  manufacturer: string
  model: string
  socManufacturer: string
  socModel: string
  apiLevel: number
  batteryTemperatureC: number | null
  abis: string[]
  totalRamBytes: number
  availableRamBytes: number
  availableStorageBytes: number
  vulkanVersion: string
  backend: LocalBackend
  vulkanAvailable: boolean
  vulkanQualified: false
  vulkanReason: string
  qualification: 'qualified-reference' | 'compatible-unqualified' | 'pixel-device-validation-required' | 'unsupported'
  runtimeInstalled: boolean
  missingRuntime: string[]
  qnnHtp: QnnCapability
  htpAvailableToApp: boolean
  htpReason: string
}

export interface LocalModel {
  id: string
  repo: string
  profile: string
  downloadBytes: number
  downloadedBytes: number
  decodedCacheBytes: number
  installedBytes: number
  minimumRamBytes: number
  installed: boolean
  fits: boolean
  reason: 'insufficient-storage' | 'insufficient-memory' | 'requires-snapdragon' | null
}

export interface LocalTransfer {
  state: TransferState
  modelId: string | null
  bytesDone: number
  bytesTotal: number
  error: string | null
}

export interface LocalRuntimeStatus {
  state: LocalEngineState
  modelId: string | null
  endpoint: string
  error: string | null
  health: Record<string, unknown> | null
  metrics: Record<string, unknown> | null
  batteryTemperatureC: number | null
  transfer: LocalTransfer
  models: LocalModel[]
}

export interface LocalChatMessage { role: 'system' | 'user' | 'assistant'; content: string }
export interface LocalChatResult { content: string; runtime: Record<string, unknown> | null }
export interface LocalBenchmarkWorkload {
  medianTokensPerSecond: number
  medianPrefillSeconds: number
  medianRequestSeconds: number
  repetitions: 3
  outputTokens: 64
  temperature: 0
}
export interface LocalBenchmarkResult {
  suite: 'tensorfold-android-quick-v1'
  modelId: string
  backend: LocalBackend
  temperatureBeforeC: number | null
  temperatureAfterC: number | null
  workloads: { code: LocalBenchmarkWorkload; chat: LocalBenchmarkWorkload }
}

interface NativeLocalRuntime {
  capabilities(): Promise<LocalCapabilities>
  models(): Promise<LocalModel[]>
  status(): Promise<LocalRuntimeStatus>
  removeModel(id: string): Promise<LocalRuntimeStatus>
  downloadModel(id: string): Promise<LocalRuntimeStatus>
  cancelDownload(): Promise<LocalRuntimeStatus>
  start(id: string): Promise<LocalRuntimeStatus>
  stop(): Promise<LocalRuntimeStatus>
  benchmark(): Promise<LocalBenchmarkResult>
  chat(messages: LocalChatMessage[], maxTokens: number, temperature: number): Promise<LocalChatResult>
}

const native = Platform.OS === 'android'
  ? requireOptionalNativeModule<NativeLocalRuntime>('TensorFoldLocalRuntime')
  : null

function runtime(): NativeLocalRuntime {
  if (!native) throw new Error('Phone-local TensorFold is available in the Android runtime build.')
  return native
}

export const localRuntimeAvailable = native !== null
export const localCapabilities = () => runtime().capabilities()
export const localModels = () => runtime().models()
export const localStatus = () => runtime().status()
export const downloadLocalModel = (id: string) => runtime().downloadModel(id)
export const removeLocalModel = (id: string) => runtime().removeModel(id)
export const cancelLocalDownload = () => runtime().cancelDownload()
export const startLocalModel = (id: string) => runtime().start(id)
export const stopLocalModel = () => runtime().stop()
export const benchmarkLocalModel = () => runtime().benchmark()
export const localChat = (messages: LocalChatMessage[], maxTokens: number, temperature: number) => runtime().chat(messages, maxTokens, temperature)

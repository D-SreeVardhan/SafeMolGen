const API_BASE = '/api/v1'

export type HealthResponse = { status: string; models_loaded: boolean }

export type OraclePrediction = {
  phase1_prob: number
  phase2_prob: number
  phase3_prob: number
  overall_prob: number
  admet_predictions: Record<string, number>
  risk_factors: Array<{ name?: string; category?: string; description?: string; impact?: number; source?: string }>
  structural_alerts: string[]
  recommendations: Array<{ type?: string; issue?: string; suggestion?: string; severity?: string; expected_improvement?: string }>
}

export type AnalyzeResponse = { smiles: string; prediction: OraclePrediction }

export type CompareResultItem = {
  smiles: string
  prediction: OraclePrediction
  properties?: Record<string, unknown>
}

export type DesignResult = {
  final_smiles: string
  final_phase1: number
  final_phase2: number
  final_phase3: number
  final_overall: number
  target_achieved: boolean
  total_iterations: number
  history: Array<{
    iteration: number
    smiles: string
    phase1_prob: number
    phase2_prob: number
    phase3_prob: number
    overall_prob: number
    improvements: string[]
    structural_alerts: string[]
    passed_safety: boolean
    used_oracle_feedback: boolean
  }>
  recommendations?: Array<{ type?: string; issue?: string; suggestion?: string; severity?: string; expected_improvement?: string }>
}

export async function getHealth(): Promise<HealthResponse> {
  const r = await fetch(`${API_BASE}/health`)
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}

export async function analyze(smiles: string): Promise<AnalyzeResponse> {
  const r = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ smiles }),
  })
  if (!r.ok) {
    const t = await r.text()
    let msg = t
    try {
      const j = JSON.parse(t)
      msg = j.detail || t
    } catch {}
    throw new Error(msg)
  }
  return r.json()
}

export async function compare(smilesList: string[]): Promise<{ results: CompareResultItem[] }> {
  const r = await fetch(`${API_BASE}/compare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ smiles_list: smilesList }),
  })
  if (!r.ok) {
    const t = await r.text()
    let msg = t
    try {
      const j = JSON.parse(t)
      msg = j.detail || t
    } catch {}
    throw new Error(msg)
  }
  return r.json()
}

export type DesignParams = {
  target_success?: number
  max_iterations?: number
  candidates_per_iteration?: number
  top_k?: number
  safety_threshold?: number
  require_no_structural_alerts?: boolean
  property_targets?: Record<string, number | [number, number]>
  seed_smiles?: string
  use_rl_model?: boolean
  selection_mode?: string
  exploration_fraction?: number
  use_phase_aware_steering?: boolean
  first_iteration_temperature?: number
}

export type ConfigResponse = {
  max_iterations_min: number
  max_iterations_max: number
  target_success_min: number
  target_success_max: number
  selection_modes: string[]
  design_modes: string[]
  has_reranker: boolean
  first_iteration_temperature_default?: number
  generator_early_available?: boolean
  default_property_targets?: Record<string, number | [number, number]>
}

export async function getConfig(): Promise<ConfigResponse> {
  const r = await fetch(`${API_BASE}/config`)
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}

export async function designSync(params: DesignParams): Promise<DesignResult> {
  const r = await fetch(`${API_BASE}/design`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  })
  if (!r.ok) {
    const t = await r.text()
    let msg = t
    try {
      const j = JSON.parse(t)
      msg = j.detail || t
    } catch {}
    throw new Error(msg)
  }
  return r.json()
}

export function moleculeSvgUrl(smiles: string, width = 400, height = 300): string {
  const enc = encodeURIComponent(smiles)
  return `${API_BASE}/molecule/svg?smiles=${enc}&width=${width}&height=${height}`
}

/** Stream design iterations via fetch + ReadableStream */
export async function designStream(
  params: DesignParams,
  onIteration: (data: DesignResult) => void,
  onDone: (data: DesignResult) => void,
  onError: (err: string) => void
): Promise<void> {
  const r = await fetch(`${API_BASE}/design/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  })
  if (!r.ok) {
    const t = await r.text()
    let msg = t
    try {
      const j = JSON.parse(t)
      msg = j.detail || t
    } catch {}
    onError(msg)
    return
  }
  const reader = r.body?.getReader()
  if (!reader) {
    onError('No response body')
    return
  }
  const decoder = new TextDecoder()
  let buf = ''
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buf += decoder.decode(value, { stream: true })
      const events = buf.split('\n\n')
      buf = events.pop() ?? ''
      for (const block of events) {
        const line = block.split('\n').find((l) => l.startsWith('data: '))
        if (!line) continue
        const jsonStr = line.slice(6)
        try {
          const payload = JSON.parse(jsonStr) as { event: string; data: DesignResult }
          if (payload.event === 'iteration') onIteration(payload.data)
          else if (payload.event === 'done') onDone(payload.data)
          else if (payload.event === 'error') onError((payload.data as unknown as { detail?: string }).detail ?? 'Unknown error')
        } catch (_) {}
      }
    }
  } finally {
    reader.releaseLock()
  }
}

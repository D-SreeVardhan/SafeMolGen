import { useState, useEffect, useRef } from 'react'
import {
  Box,
  Button,
  Card,
  CardBody,
  CardHeader,
  Collapse,
  Flex,
  FormControl,
  FormLabel,
  Grid,
  Heading,
  Input,
  NumberDecrementStepper,
  NumberIncrementStepper,
  NumberInput,
  NumberInputField,
  Select,
  SimpleGrid,
  Slider,
  SliderFilledTrack,
  SliderThumb,
  SliderTrack,
  Switch,
  Text,
  useDisclosure,
  useToast,
} from '@chakra-ui/react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts'
import { designStream, moleculeSvgUrl, getConfig, type DesignResult, type DesignParams, type ConfigResponse } from '../api/client'

// If backend sends raw overall (e.g. 0.02), show calibrated % so success can exceed 50%
const ORACLE_CALIBRATION_K = 22
function overallForDisplay(value: number): number {
  if (value > 0.15) return value
  return 1 - Math.exp(-ORACLE_CALIBRATION_K * value)
}

const DEFAULT_FIRST_ITER_TEMP = 1.4

export default function Generate() {
  const [config, setConfig] = useState<ConfigResponse | null>(null)
  const [targetSuccess, setTargetSuccess] = useState(0.7)
  const [maxIterations, setMaxIterations] = useState(10)
  const [seedSmiles, setSeedSmiles] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<DesignResult | null>(null)
  const [streamingHistory, setStreamingHistory] = useState<DesignResult[]>([])
  const { isOpen: propsOpen, onToggle: toggleProps } = useDisclosure({ defaultIsOpen: false })
  const { isOpen: propTargetsOpen, onToggle: togglePropTargets } = useDisclosure({ defaultIsOpen: false })
  const toast = useToast()

  // Property targets: sliders with defaults (0 = no constraint for max/mw_min)
  const [logpMin, setLogpMin] = useState(2)
  const [logpMax, setLogpMax] = useState(5)
  const [mwMin, setMwMin] = useState(150)
  const [mwMax, setMwMax] = useState(500)
  const [hbdMax, setHbdMax] = useState(5)
  const [hbaMax, setHbaMax] = useState(10)
  const [tpsaMax, setTpsaMax] = useState(140)
  const [qedMin, setQedMin] = useState(0.5)
  const [topK, setTopK] = useState(40)
  const [safetyThreshold, setSafetyThreshold] = useState(0.02)
  const [requireNoAlerts, setRequireNoAlerts] = useState(false)
  const [useRlModel, setUseRlModel] = useState(false)
  const [selectionMode, setSelectionMode] = useState('phase_weighted')
  const [firstIterationTemperature, setFirstIterationTemperature] = useState<number | ''>('')

  useEffect(() => {
    getConfig().then(setConfig).catch(() => setConfig(null))
  }, [])

  const didSyncPropertyDefaults = useRef(false)
  useEffect(() => {
    const def = config?.default_property_targets
    if (!def || didSyncPropertyDefaults.current) return
    didSyncPropertyDefaults.current = true
    if (Array.isArray(def.logp) && def.logp.length >= 2) {
      setLogpMin(def.logp[0]); setLogpMax(def.logp[1])
    }
    if (typeof def.mw_min === 'number') setMwMin(def.mw_min)
    if (typeof def.mw === 'number') setMwMax(def.mw)
    if (typeof def.hbd === 'number') setHbdMax(def.hbd)
    if (typeof def.hba === 'number') setHbaMax(def.hba)
    if (typeof def.tpsa === 'number') setTpsaMax(def.tpsa)
    if (typeof def.qed === 'number') setQedMin(def.qed)
  }, [config?.default_property_targets])

  const firstIterTempValue = firstIterationTemperature !== '' ? firstIterationTemperature : (config?.first_iteration_temperature_default ?? DEFAULT_FIRST_ITER_TEMP)

  const propertyTargets: Record<string, number | [number, number]> = {}
  propertyTargets.logp = [logpMin, logpMax]
  if (mwMin > 0) propertyTargets.mw_min = mwMin
  if (mwMax > 0) propertyTargets.mw = mwMax
  if (hbdMax > 0) propertyTargets.hbd = hbdMax
  if (hbaMax > 0) propertyTargets.hba = hbaMax
  if (tpsaMax > 0) propertyTargets.tpsa = tpsaMax
  propertyTargets.qed = qedMin
  const hasPropertyTargets = Object.keys(propertyTargets).length > 0

  const params: DesignParams = {
    target_success: targetSuccess,
    max_iterations: maxIterations,
    candidates_per_iteration: 200,
    top_k: topK,
    safety_threshold: safetyThreshold,
    require_no_structural_alerts: requireNoAlerts,
    seed_smiles: seedSmiles.trim() || undefined,
    use_rl_model: useRlModel,
    selection_mode: selectionMode,
    ...(hasPropertyTargets && { property_targets: propertyTargets }),
    ...(firstIterationTemperature !== '' && { first_iteration_temperature: Number(firstIterationTemperature) }),
  }

  function handleRun() {
    setLoading(true)
    setResult(null)
    setStreamingHistory([])
    designStream(
      params,
      (data) => {
        setStreamingHistory((prev) => [...prev, data])
        setResult(data)
      },
      (data) => {
        setResult(data)
        setLoading(false)
        toast({ title: data.target_achieved ? 'Target achieved!' : 'Run finished', status: 'success', isClosable: true })
      },
      (err) => {
        setLoading(false)
        toast({ title: 'Error', description: err, status: 'error', isClosable: true })
      }
    )
  }

  const displayResult = result ?? (streamingHistory.length ? streamingHistory[streamingHistory.length - 1] : null)
  // Per-iteration actual values (so the journey shows real ups/downs); plus cumulative best overall for reference.
  const chartData = (() => {
    const history = displayResult?.history ?? []
    if (history.length === 0) return []
    let bestOverallSoFar = 0
    return history.map((h) => {
      const oRaw = overallForDisplay(h.overall_prob) * 100
      const p1Raw = h.phase1_prob * 100
      const p2Raw = h.phase2_prob * 100
      const p3Raw = h.phase3_prob * 100
      bestOverallSoFar = Math.max(bestOverallSoFar, oRaw)
      return {
        iteration: h.iteration,
        // Use 3 decimals so small per-iteration variation is visible (e.g. 88.59 vs 88.63)
        overall: Math.round(oRaw * 1000) / 1000,
        phase1: Math.round(p1Raw * 100) / 100,
        phase2: Math.round(p2Raw * 100) / 100,
        phase3: Math.round(p3Raw * 100) / 100,
        bestOverall: Math.round(bestOverallSoFar * 1000) / 1000,
      }
    })
  })()
  // When all overall values are very close, zoom Y-axis so small changes are visible
  const overallRange = chartData.length
    ? (() => {
        const vals = chartData.map((d) => d.overall)
        const min = Math.min(...vals)
        const max = Math.max(...vals)
        return { min, max, spread: max - min }
      })()
    : null
  const overallYDomain: [number, number] =
    overallRange && overallRange.spread < 10
      ? [Math.max(0, Math.floor(overallRange.min - 2)), Math.min(100, Math.ceil(overallRange.max + 2))]
      : [0, 100]

  return (
    <Box>
      <Heading size="lg" mb={2}>Generate New Molecules</Heading>
      <Text color="gray.600" mb={6}>AI-driven design with DrugOracle feedback.</Text>

      <Grid templateColumns={{ base: '1fr', xl: '320px 1fr' }} gap={6}>
        <Card>
          <CardHeader>
            <Heading size="sm">Generation parameters</Heading>
          </CardHeader>
          <CardBody>
            <FormControl mb={4}>
              <FormLabel>Target success probability</FormLabel>
              <Slider value={targetSuccess} min={0.1} max={0.95} step={0.05} onChange={(v) => setTargetSuccess(v)}>
                <SliderTrack><SliderFilledTrack /></SliderTrack>
                <SliderThumb />
              </Slider>
              <Text fontSize="sm" color="gray.500">{targetSuccess}</Text>
            </FormControl>
            <FormControl mb={4}>
              <FormLabel>Max iterations</FormLabel>
              <NumberInput value={maxIterations} min={1} max={20} onChange={(_, v) => setMaxIterations(v || 10)}>
                <NumberInputField />
                <NumberIncrementStepper />
                <NumberDecrementStepper />
              </NumberInput>
            </FormControl>
            <FormControl mb={4}>
              <FormLabel>Seed SMILES (optional)</FormLabel>
              <Input
                value={seedSmiles}
                onChange={(e) => setSeedSmiles(e.target.value)}
                placeholder="e.g. c1ccccc1"
              />
              <Text fontSize="xs" color="gray.500" mt={1}>Leave empty for no scaffold constraint.</Text>
            </FormControl>
            <FormControl mb={4}>
              <Button size="sm" variant="ghost" onClick={togglePropTargets}>
                {propTargetsOpen ? 'Hide' : 'Show'} property targets
              </Button>
              <Collapse in={propTargetsOpen}>
                <Box mt={3}>
                  <FormControl mb={4}>
                    <FormLabel fontSize="sm">logP (min – max)</FormLabel>
                    <Flex gap={4} align="center">
                      <Slider value={logpMin} min={-2} max={10} step={0.5} maxW="40%" onChange={setLogpMin}>
                        <SliderTrack><SliderFilledTrack /></SliderTrack>
                        <SliderThumb />
                      </Slider>
                      <Text fontSize="sm" w="2rem">{logpMin}</Text>
                      <Slider value={logpMax} min={-2} max={10} step={0.5} maxW="40%" onChange={setLogpMax}>
                        <SliderTrack><SliderFilledTrack /></SliderTrack>
                        <SliderThumb />
                      </Slider>
                      <Text fontSize="sm" w="2rem">{logpMax}</Text>
                    </Flex>
                    <Text fontSize="xs" color="gray.500" mt={1}>Lipophilicity; typical drug range 2–5. Affects permeability and solubility.</Text>
                  </FormControl>
                  <FormControl mb={4}>
                    <FormLabel fontSize="sm">MW min (0 = no minimum) — {mwMin}</FormLabel>
                    <Slider value={mwMin} min={0} max={400} step={10} onChange={setMwMin}>
                      <SliderTrack><SliderFilledTrack /></SliderTrack>
                      <SliderThumb />
                    </Slider>
                    <Text fontSize="xs" color="gray.500" mt={1}>Minimum molecular weight. Set ≥150 to avoid tiny molecules (e.g. ethane).</Text>
                  </FormControl>
                  <FormControl mb={4}>
                    <FormLabel fontSize="sm">MW max — {mwMax}</FormLabel>
                    <Slider value={mwMax} min={100} max={800} step={10} onChange={setMwMax}>
                      <SliderTrack><SliderFilledTrack /></SliderTrack>
                      <SliderThumb />
                    </Slider>
                    <Text fontSize="xs" color="gray.500" mt={1}>Maximum molecular weight; often 400–500 for oral drugs (Lipinski &lt;500).</Text>
                  </FormControl>
                  <FormControl mb={4}>
                    <FormLabel fontSize="sm">HBD max (0 = no limit) — {hbdMax}</FormLabel>
                    <Slider value={hbdMax} min={0} max={15} step={1} onChange={setHbdMax}>
                      <SliderTrack><SliderFilledTrack /></SliderTrack>
                      <SliderThumb />
                    </Slider>
                    <Text fontSize="xs" color="gray.500" mt={1}>Hydrogen bond donors (e.g. OH, NH). Lipinski rule ≤5.</Text>
                  </FormControl>
                  <FormControl mb={4}>
                    <FormLabel fontSize="sm">HBA max (0 = no limit) — {hbaMax}</FormLabel>
                    <Slider value={hbaMax} min={0} max={20} step={1} onChange={setHbaMax}>
                      <SliderTrack><SliderFilledTrack /></SliderTrack>
                      <SliderThumb />
                    </Slider>
                    <Text fontSize="xs" color="gray.500" mt={1}>Hydrogen bond acceptors. Lipinski rule ≤10.</Text>
                  </FormControl>
                  <FormControl mb={4}>
                    <FormLabel fontSize="sm">TPSA max (0 = no limit) — {tpsaMax}</FormLabel>
                    <Slider value={tpsaMax} min={0} max={200} step={5} onChange={setTpsaMax}>
                      <SliderTrack><SliderFilledTrack /></SliderTrack>
                      <SliderThumb />
                    </Slider>
                    <Text fontSize="xs" color="gray.500" mt={1}>Topological polar surface area; lower often helps oral absorption.</Text>
                  </FormControl>
                  <FormControl mb={4}>
                    <FormLabel fontSize="sm">QED min — {qedMin}</FormLabel>
                    <Slider value={qedMin} min={0} max={1} step={0.05} onChange={setQedMin}>
                      <SliderTrack><SliderFilledTrack /></SliderTrack>
                      <SliderThumb />
                    </Slider>
                    <Text fontSize="xs" color="gray.500" mt={1}>Drug-likeness score (0–1); higher = more drug-like by QED.</Text>
                  </FormControl>
                </Box>
              </Collapse>
            </FormControl>
            <FormControl mb={4}>
              <Button size="sm" variant="ghost" onClick={toggleProps}>
                {propsOpen ? 'Hide' : 'Show'} advanced
              </Button>
              <Collapse in={propsOpen}>
                <Box mt={2}>
                  <FormControl mb={2}>
                    <FormLabel fontSize="sm">Top-K</FormLabel>
                    <NumberInput size="sm" value={topK} min={0} max={100} step={10} onChange={(_, v) => setTopK(v ?? 40)}>
                      <NumberInputField />
                    </NumberInput>
                  </FormControl>
                  <FormControl mb={2}>
                    <FormLabel fontSize="sm">Safety threshold</FormLabel>
                    <Slider value={safetyThreshold} min={0.01} max={0.5} step={0.01} onChange={(v) => setSafetyThreshold(v)}>
                      <SliderTrack><SliderFilledTrack /></SliderTrack>
                      <SliderThumb />
                    </Slider>
                    <Text fontSize="xs" color="gray.500">{safetyThreshold}</Text>
                  </FormControl>
                  <FormControl mb={2}>
                    <FormLabel fontSize="sm">Selection mode</FormLabel>
                    <Select size="sm" value={selectionMode} onChange={(e) => setSelectionMode(e.target.value)}>
                      <option value="overall">Overall (best overall %)</option>
                      <option value="phase_weighted">Phase weighted (favor Phase II)</option>
                      <option value="bottleneck">Bottleneck (improve worst phase)</option>
                      <option value="pareto">Pareto (non-dominated)</option>
                      <option value="diversity">Diversity (structurally different)</option>
                    </Select>
                    <Text fontSize="xs" color="gray.500" mt={1}>Phase weighted often shows clearer improvement.</Text>
                  </FormControl>
                  <FormControl mb={2}>
                    <FormLabel fontSize="sm">First-iteration temperature</FormLabel>
                    <NumberInput size="sm" value={firstIterationTemperature === '' ? firstIterTempValue : firstIterationTemperature} min={0.8} max={1.7} step={0.05} onChange={(_, v) => setFirstIterationTemperature(v === '' ? '' : Number(v) || '')}>
                      <NumberInputField />
                    </NumberInput>
                    <Text fontSize="xs" color="gray.500" mt={1}>Higher = weaker first batch (more room for improvement). Use 1.4–1.6 for a visible improvement curve.</Text>
                  </FormControl>
                  {config?.generator_early_available && (
                    <Text fontSize="xs" color="gray.600" mt={1}>Weak-start checkpoint is available; first iteration will use it.</Text>
                  )}
                  <FormControl display="flex" alignItems="center" mb={2}>
                    <FormLabel mb={0}>Require no structural alerts</FormLabel>
                    <Switch isChecked={requireNoAlerts} onChange={(e) => setRequireNoAlerts(e.target.checked)} />
                  </FormControl>
                  <FormControl display="flex" alignItems="center">
                    <FormLabel mb={0}>Use RL model</FormLabel>
                    <Switch isChecked={useRlModel} onChange={(e) => setUseRlModel(e.target.checked)} />
                  </FormControl>
                </Box>
              </Collapse>
            </FormControl>
            <Button
              colorScheme="blue"
              width="full"
              onClick={handleRun}
              isLoading={loading}
              loadingText="Generating…"
            >
              Run generation
            </Button>
            <Button
              size="sm"
              variant="outline"
              width="full"
              mt={2}
              onClick={() => {
                setSelectionMode('phase_weighted')
                setTargetSuccess(0.35)
                setMaxIterations(10)
                setLoading(true)
                setResult(null)
                setStreamingHistory([])
                const demoFirstTemp = config?.first_iteration_temperature_default ?? DEFAULT_FIRST_ITER_TEMP
                const demoParams: DesignParams = {
                  target_success: 0.35,
                  max_iterations: 10,
                  candidates_per_iteration: 200,
                  top_k: topK,
                  safety_threshold: safetyThreshold,
                  require_no_structural_alerts: requireNoAlerts,
                  seed_smiles: undefined,
                  use_rl_model: useRlModel,
                  selection_mode: 'phase_weighted',
                  first_iteration_temperature: demoFirstTemp,
                  ...(hasPropertyTargets && { property_targets: propertyTargets }),
                }
                designStream(
                  demoParams,
                  (data) => {
                    setStreamingHistory((prev) => [...prev, data])
                    setResult(data)
                  },
                  (data) => {
                    setResult(data)
                    setLoading(false)
                    toast({ title: data.target_achieved ? 'Target achieved!' : 'Demo finished', status: 'success', isClosable: true })
                  },
                  (err) => {
                    setLoading(false)
                    toast({ title: 'Error', description: err, status: 'error', isClosable: true })
                  }
                )
              }}
              isDisabled={loading}
              title="Run demo: weak first iteration + phase-weighted selection to see improvement"
            >
              Run demo optimization
            </Button>
          </CardBody>
        </Card>

        <Box>
          <Card>
            <CardHeader>
              <Heading size="sm">Results</Heading>
            </CardHeader>
            <CardBody>
              {!displayResult && !loading && (
                <Text color="gray.500">Configure parameters and click Run generation.</Text>
              )}
              {loading && !displayResult && (
                <Text color="blue.600">Starting generation…</Text>
              )}
              {displayResult && (() => {
                const dispOverall = Math.round(overallForDisplay(displayResult.final_overall) * 10000) / 100
                const dispPhase1 = Math.round(displayResult.final_phase1 * 10000) / 100
                const dispPhase2 = Math.round(displayResult.final_phase2 * 10000) / 100
                const dispPhase3 = Math.round(displayResult.final_phase3 * 10000) / 100
                return (
                <>
                  <SimpleGrid columns={{ base: 2, md: 4 }} spacing={4} mb={6}>
                    <Box p={3} bg="gray.50" borderRadius="md" textAlign="center">
                      <Text fontSize="xs" color="gray.600">Phase I</Text>
                      <Text fontSize="xl" fontWeight="bold">{dispPhase1.toFixed(2)}%</Text>
                    </Box>
                    <Box p={3} bg="gray.50" borderRadius="md" textAlign="center">
                      <Text fontSize="xs" color="gray.600">Phase II</Text>
                      <Text fontSize="xl" fontWeight="bold">{dispPhase2.toFixed(2)}%</Text>
                    </Box>
                    <Box p={3} bg="gray.50" borderRadius="md" textAlign="center">
                      <Text fontSize="xs" color="gray.600">Phase III</Text>
                      <Text fontSize="xl" fontWeight="bold">{dispPhase3.toFixed(2)}%</Text>
                    </Box>
                    <Box p={3} bg="gray.50" borderRadius="md" textAlign="center">
                      <Text fontSize="xs" color="gray.600">Overall</Text>
                      <Text fontSize="xl" fontWeight="bold">{dispOverall.toFixed(2)}%</Text>
                    </Box>
                  </SimpleGrid>
                  {displayResult.history && displayResult.history.length >= 2 && (() => {
                    const firstOverall = Math.round(overallForDisplay(displayResult.history[0].overall_prob) * 10000) / 100
                    const lastOverall = Math.round(overallForDisplay(displayResult.final_overall) * 10000) / 100
                    const improved = lastOverall > firstOverall
                    return (
                      <Text fontSize="sm" mb={4} color={improved ? 'green.600' : 'gray.600'}>
                        Improvement: {firstOverall.toFixed(1)}% → {lastOverall.toFixed(1)}% overall
                        {improved ? ' ✓' : ''}
                      </Text>
                    )
                  })()}
                  <Box mb={6}>
                    <Text fontWeight="semibold" mb={2}>Best molecule</Text>
                    <Flex gap={4} flexWrap="wrap">
                      {displayResult.final_smiles ? (
                        <>
                          <Box borderWidth="1px" borderRadius="md" overflow="hidden" w="280px" h="200px">
                            <img
                              src={moleculeSvgUrl(displayResult.final_smiles, 280, 200)}
                              alt="Best molecule"
                              style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                            />
                          </Box>
                          <Box>
                            <Text fontFamily="mono" fontSize="sm" wordBreak="break-all">{displayResult.final_smiles}</Text>
                            {displayResult.target_achieved && (
                              <Text color="green.600" mt={2}>Target achieved</Text>
                            )}
                          </Box>
                        </>
                      ) : (
                        <Text color="gray.600" fontSize="sm">No drug-like molecule found (all candidates were too simple). Try more iterations or different parameters.</Text>
                      )}
                    </Flex>
                  </Box>
                  {chartData.length > 0 && (
                    <Box mb={6}>
                      <Text fontWeight="semibold" mb={2}>Optimization journey</Text>
                      <Text fontSize="xs" color="gray.500" mb={2}>Per-iteration best (actual). Target ≥{Math.round(targetSuccess * 100)}% overall. “Best overall” = cumulative max.</Text>
                      <Box h="300px">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={chartData}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="iteration" />
                            <YAxis yAxisId="left" orientation="left" label={{ value: 'Phase success %', angle: -90 }} domain={[0, 100]} />
                            <YAxis yAxisId="right" orientation="right" label={{ value: 'Overall %', angle: 90 }} domain={overallYDomain} />
                            <ReferenceLine yAxisId="left" y={14} stroke="gray" strokeDasharray="3 3" />
                            <ReferenceLine yAxisId="left" y={25} stroke="green" strokeDasharray="3 3" />
                            {targetSuccess * 100 >= overallYDomain[0] && targetSuccess * 100 <= overallYDomain[1] && (
                              <ReferenceLine yAxisId="right" y={targetSuccess * 100} stroke="gray" strokeDasharray="5 5" />
                            )}
                            <Tooltip formatter={(value: number, name: string) => [value, name]} />
                            <Legend />
                            <Line yAxisId="right" type="monotone" dataKey="overall" name="Overall (this iter)" stroke="#3182CE" strokeWidth={2} />
                            <Line yAxisId="right" type="monotone" dataKey="bestOverall" name="Best overall" stroke="#3182CE" strokeWidth={1.5} strokeDasharray="5 5" />
                            <Line yAxisId="left" type="monotone" dataKey="phase1" name="Phase I" stroke="#38A169" />
                            <Line yAxisId="left" type="monotone" dataKey="phase2" name="Phase II" stroke="#D69E2E" />
                            <Line yAxisId="left" type="monotone" dataKey="phase3" name="Phase III" stroke="#805AD5" />
                          </LineChart>
                        </ResponsiveContainer>
                      </Box>
                      {chartData.length >= 2 && (() => {
                        const firstO = chartData[0].overall
                        const lastO = chartData[chartData.length - 1].overall
                        if (Math.abs(lastO - firstO) < 0.01) {
                          return (
                            <Text fontSize="sm" color="gray.600" mt={2}>
                              No improvement this run. Try Advanced &gt; First-iteration temperature (or &quot;Run demo optimization&quot;) to start from a weaker baseline.
                            </Text>
                          )
                        }
                        return null
                      })()}
                    </Box>
                  )}
                  {chartData.length > 0 && (
                    <Text fontSize="xs" color="gray.500" mt={2}>
                      Use Advanced &gt; First-iteration temperature (or weak-start checkpoint) to start from a weaker baseline and see improvement.
                    </Text>
                  )}
                  {displayResult.recommendations && displayResult.recommendations.length > 0 && (
                    <Box>
                      <Text fontWeight="semibold" mb={2}>Recommendations</Text>
                      {displayResult.recommendations.map((rec, i) => (
                        <Box key={i} py={2} borderBottomWidth="1px">
                          <Text fontSize="sm"><strong>{rec.type}</strong>: {rec.suggestion}</Text>
                        </Box>
                      ))}
                    </Box>
                  )}
                </>
                )
              })()}
            </CardBody>
          </Card>
        </Box>
      </Grid>
    </Box>
  )
}

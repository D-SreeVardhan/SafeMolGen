import { useState } from 'react'
import {
  Box,
  Button,
  Card,
  CardBody,
  CardHeader,
  Flex,
  FormControl,
  FormLabel,
  Grid,
  Heading,
  Input,
  SimpleGrid,
  Text,
  useToast,
} from '@chakra-ui/react'
import { analyze, moleculeSvgUrl, type AnalyzeResponse, type OraclePrediction } from '../api/client'

function OracleDashboard({ prediction }: { prediction: OraclePrediction }) {
  const metrics = [
    { label: 'Phase I', value: prediction.phase1_prob },
    { label: 'Phase II', value: prediction.phase2_prob },
    { label: 'Phase III', value: prediction.phase3_prob },
    { label: 'Overall', value: prediction.overall_prob },
  ]
  return (
    <SimpleGrid columns={{ base: 2, md: 4 }} spacing={4}>
      {metrics.map(({ label, value }) => (
        <Box key={label} p={4} bg="white" borderRadius="md" borderWidth="1px" textAlign="center">
          <Text fontSize="sm" color="gray.600">{label}</Text>
          <Text fontSize="2xl" fontWeight="bold">{(value * 100).toFixed(1)}%</Text>
        </Box>
      ))}
    </SimpleGrid>
  )
}

export default function Analyze() {
  const [smiles, setSmiles] = useState('CC(=O)Oc1ccccc1C(=O)O')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<AnalyzeResponse | null>(null)
  const toast = useToast()

  async function handleAnalyze() {
    if (!smiles.trim()) return
    setLoading(true)
    setResult(null)
    try {
      const data = await analyze(smiles.trim())
      setResult(data)
    } catch (e) {
      toast({ title: 'Error', description: String(e), status: 'error', isClosable: true })
    } finally {
      setLoading(false)
    }
  }

  return (
    <Box>
      <Heading size="lg" mb={2}>Analyze Molecule</Heading>
      <Text color="gray.600" mb={6}>Enter a SMILES string to get Oracle predictions and recommendations.</Text>

      <Grid templateColumns={{ base: '1fr', lg: '1fr 2fr' }} gap={6}>
        <Card>
          <CardHeader>
            <Heading size="sm">Input</Heading>
          </CardHeader>
          <CardBody>
            <FormControl mb={4}>
              <FormLabel>SMILES</FormLabel>
              <Input
                value={smiles}
                onChange={(e) => setSmiles(e.target.value)}
                placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O"
              />
            </FormControl>
            <Button colorScheme="blue" onClick={handleAnalyze} isLoading={loading} width="full">
              Analyze
            </Button>
          </CardBody>
        </Card>

        <Box>
          {result ? (
            <Card>
              <CardHeader>
                <Heading size="sm">Results</Heading>
              </CardHeader>
              <CardBody>
                <Flex gap={6} flexWrap="wrap">
                  <Box flex="0 0 auto">
                    <Text fontSize="sm" color="gray.600" mb={2}>Structure</Text>
                    <Box borderWidth="1px" borderRadius="md" overflow="hidden" w="280px" h="200px">
                      <img
                        src={moleculeSvgUrl(result.smiles, 280, 200)}
                        alt="Molecule"
                        style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                      />
                    </Box>
                    <Text fontFamily="mono" fontSize="xs" mt={2} noOfLines={2}>{result.smiles}</Text>
                  </Box>
                  <Box flex="1" minW="200px">
                    <Text fontSize="sm" color="gray.600" mb={2}>Oracle prediction</Text>
                    <OracleDashboard prediction={result.prediction} />
                  </Box>
                </Flex>
                {result.prediction.risk_factors?.length ? (
                  <Box mt={4}>
                    <Text fontWeight="semibold" mb={2}>Risk factors</Text>
                    {result.prediction.risk_factors.map((r, i) => (
                      <Box key={i} py={2} borderBottomWidth="1px">
                        <Text fontSize="sm">{r.name} — {r.description}</Text>
                      </Box>
                    ))}
                  </Box>
                ) : null}
                {result.prediction.recommendations?.length ? (
                  <Box mt={4}>
                    <Text fontWeight="semibold" mb={2}>Recommendations</Text>
                    {result.prediction.recommendations.map((rec, i) => (
                      <Box key={i} py={2} borderBottomWidth="1px">
                        <Text fontSize="sm"><strong>{rec.type}</strong>: {rec.suggestion}</Text>
                      </Box>
                    ))}
                  </Box>
                ) : null}
              </CardBody>
            </Card>
          ) : (
            <Card>
              <CardBody>
                <Text color="gray.500">Configure SMILES and click Analyze to see results.</Text>
              </CardBody>
            </Card>
          )}
        </Box>
      </Grid>
    </Box>
  )
}

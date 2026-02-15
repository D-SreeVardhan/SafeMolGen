import { useState } from 'react'
import {
  Box,
  Button,
  Card,
  CardBody,
  CardHeader,
  Heading,
  Table,
  TableContainer,
  Tbody,
  Td,
  Text,
  Textarea,
  Th,
  Thead,
  Tr,
  useToast,
} from '@chakra-ui/react'
import { compare, type CompareResultItem } from '../api/client'

export default function Compare() {
  const [smilesText, setSmilesText] = useState('CC(=O)Oc1ccccc1C(=O)O\nCCO\nc1ccccc1')
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<CompareResultItem[] | null>(null)
  const toast = useToast()

  async function handleCompare() {
    const list = smilesText.split('\n').map(s => s.trim()).filter(Boolean)
    if (list.length < 2) {
      toast({ title: 'Enter at least 2 SMILES', status: 'warning', isClosable: true })
      return
    }
    setLoading(true)
    setResults(null)
    try {
      const { results: data } = await compare(list)
      setResults(data)
    } catch (e) {
      toast({ title: 'Error', description: String(e), status: 'error', isClosable: true })
    } finally {
      setLoading(false)
    }
  }

  return (
    <Box>
      <Heading size="lg" mb={2}>Compare Molecules</Heading>
      <Text color="gray.600" mb={6}>Enter one SMILES per line to compare Oracle predictions.</Text>

      <Card mb={6}>
        <CardHeader>
          <Heading size="sm">Input</Heading>
        </CardHeader>
        <CardBody>
          <Textarea
            value={smilesText}
            onChange={(e) => setSmilesText(e.target.value)}
            placeholder="One SMILES per line"
            rows={5}
            fontFamily="mono"
            mb={4}
          />
          <Button colorScheme="blue" onClick={handleCompare} isLoading={loading}>
            Compare
          </Button>
        </CardBody>
      </Card>

      {results && results.length > 0 && (
        <Card>
          <CardHeader>
            <Heading size="sm">Comparison</Heading>
          </CardHeader>
          <CardBody>
          <TableContainer>
            <Table size="sm">
              <Thead>
                <Tr>
                  <Th>SMILES</Th>
                  <Th isNumeric>Phase I</Th>
                  <Th isNumeric>Phase II</Th>
                  <Th isNumeric>Phase III</Th>
                  <Th isNumeric>Overall</Th>
                  <Th isNumeric>Alerts</Th>
                </Tr>
              </Thead>
              <Tbody>
                {results.map((r, i) => (
                  <Tr key={i}>
                    <Td fontFamily="mono" maxW="200px" isTruncated>{r.smiles.length > 35 ? r.smiles.slice(0, 32) + '...' : r.smiles}</Td>
                    <Td isNumeric>{(r.prediction.phase1_prob * 100).toFixed(1)}%</Td>
                    <Td isNumeric>{(r.prediction.phase2_prob * 100).toFixed(1)}%</Td>
                    <Td isNumeric>{(r.prediction.phase3_prob * 100).toFixed(1)}%</Td>
                    <Td isNumeric>{(r.prediction.overall_prob * 100).toFixed(1)}%</Td>
                    <Td isNumeric>{r.prediction.structural_alerts?.length ?? 0}</Td>
                  </Tr>
                ))}
              </Tbody>
            </Table>
          </TableContainer>
          </CardBody>
        </Card>
      )}

      {results && results.length === 0 && (
        <Text color="gray.500">No valid molecules to compare.</Text>
      )}
    </Box>
  )
}

import { Box, Heading, Text, VStack } from '@chakra-ui/react'

export default function About() {
  return (
    <Box>
      <Heading size="lg" mb={2}>
        About SafeMolGen-DrugOracle
      </Heading>
      <Text color="gray.600" mb={6}>
        Integrated AI system for intelligent drug design
      </Text>
      <VStack align="stretch" spacing={4}>
        <Text>
          SafeMolGen-DrugOracle combines:
        </Text>
        <Box as="ol" pl={6}>
          <Text as="li">A Transformer-based generator for drug-like molecules</Text>
          <Text as="li">DrugOracle for clinical trial success probability</Text>
          <Text as="li">Iterative feedback loops to guide generation</Text>
          <Text as="li">Actionable recommendations and explainability</Text>
        </Box>
      </VStack>
    </Box>
  )
}

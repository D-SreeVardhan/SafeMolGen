import { Outlet, NavLink } from 'react-router-dom'
import { Box, Flex, Heading, Link, VStack } from '@chakra-ui/react'

const navItems = [
  { to: '/generate', label: 'Generate' },
  { to: '/analyze', label: 'Analyze' },
  { to: '/compare', label: 'Compare' },
  { to: '/about', label: 'About' },
]

export default function Layout() {
  return (
    <Flex minH="100vh" bg="gray.50">
      <Box
        as="aside"
        w="220px"
        py={6}
        px={4}
        bg="white"
        borderRight="1px"
        borderColor="gray.200"
        shadow="sm"
      >
        <Heading size="md" mb={6} px={2}>
          SafeMolGen-DrugOracle
        </Heading>
        <VStack align="stretch" spacing={1}>
          {navItems.map(({ to, label }) => (
            <Link
              key={to}
              as={NavLink}
              to={to}
              px={3}
              py={2}
              borderRadius="md"
              _activeLink={{ bg: 'blue.50', color: 'blue.700', fontWeight: 'semibold' }}
              _hover={{ bg: 'gray.100' }}
            >
              {label}
            </Link>
          ))}
        </VStack>
      </Box>
      <Box as="main" flex={1} p={8} overflow="auto">
        <Outlet />
      </Box>
    </Flex>
  )
}

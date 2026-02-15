import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import About from './pages/About'
import Analyze from './pages/Analyze'
import Compare from './pages/Compare'
import Generate from './pages/Generate'

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Navigate to="/generate" replace />} />
        <Route path="generate" element={<Generate />} />
        <Route path="analyze" element={<Analyze />} />
        <Route path="compare" element={<Compare />} />
        <Route path="about" element={<About />} />
      </Route>
    </Routes>
  )
}

export default App

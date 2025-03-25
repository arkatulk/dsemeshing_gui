import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import Tester from './tester.jsx'
import { Trail } from '@react-three/drei'
import Trial1 from './test1.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <Trial1 />
  </StrictMode>,
)

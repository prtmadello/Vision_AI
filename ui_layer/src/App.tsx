import { Route, Routes, useLocation } from 'react-router-dom'
import './App.css'
import Dashboard from './screens/Dashboard'
import Header from './components/sections/Header'
import Footer from './components/sections/Footer'
import DemoDashboard from './screens/DemoDashbaord'
import ProtoDashboard from './screens/ProtoDashboard'

function App() {

  const location = useLocation()

  return (
   <div className=' background  space-y-2'>
      {location.pathname != '/' && <Header/>}
      <div className="main">
        <Routes>
          <Route path='/dashboard' element={<ProtoDashboard/>}/>
        </Routes>
      </div>
      {location.pathname != '/' && <Footer/>}
   </div>
  )
}

export default App

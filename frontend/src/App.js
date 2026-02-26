import React from 'react';
import './App.css';
import Home from './components/Home';
import SymptomSelection from './components/SymptomSelection';
import ResultsPage from './components/ResultsPage';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/symptoms" element={<SymptomSelection />} />
          <Route path="/results" element={<ResultsPage />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
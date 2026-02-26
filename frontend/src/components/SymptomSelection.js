import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const SymptomSelection = () => {
  const [symptoms, setSymptoms] = useState([]);
  const [selectedSymptoms, setSelectedSymptoms] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [severity, setSeverity] = useState(3);
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('male');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchSymptoms = async () => {
      try {
        const response = await axios.get('http://localhost:8000/symptoms');
        setSymptoms(response.data.symptoms);
      } catch {
        setSymptoms([
          'fever','cough','headache','fatigue','nausea','vomiting','diarrhea',
          'abdominal_pain','chest_pain','shortness_of_breath','dizziness',
          'joint_pain','muscle_pain','sore_throat','runny_nose','skin_rash',
          'itching','swelling','weight_loss','weight_gain','loss_of_appetite',
          'excessive_thirst','frequent_urination','blurred_vision','tingling',
          'numbness','confusion','memory_problems','mood_changes','sleep_disturbance'
        ]);
      }
    };
    fetchSymptoms();
  }, []);

  const toggleSymptom = (symptom) => {
    setSelectedSymptoms(prev =>
      prev.includes(symptom)
        ? prev.filter(s => s !== symptom)
        : [...prev, symptom]
    );
  };

  const filteredSymptoms = symptoms.filter(symptom =>
    symptom.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedSymptoms.length) return;

    setLoading(true);
    try {
      const response = await axios.post('http://localhost:8000/predict', {
        symptoms: selectedSymptoms,
        age: parseInt(age) || 25,
        gender,
        severity
      });

      localStorage.setItem('predictionResults', JSON.stringify(response.data));
      navigate('/results');
    } catch (error) {
      console.error(error);
      alert('Error getting prediction. Make sure backend is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-wrapper">
      <div className="card">
        <form onSubmit={handleSubmit}>
          <h1 style={{ textAlign: "center" }}>Symptom Selection</h1>
          <p style={{ textAlign: "center", marginBottom: "25px" }}>
            Select your symptoms to get AI-powered disease predictions
          </p>

          {/* Search */}
          <div style={{ maxWidth: "400px", margin: "0 auto 25px" }}>
            <input
              type="text"
              placeholder="Search symptoms..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>

          {/* Selected Symptoms */}
          <h3 style={{ textAlign: "center" }}>Selected Symptoms ({selectedSymptoms.length})</h3>
          <div style={{ textAlign: "center", marginBottom: "20px" }}>
            {selectedSymptoms.map((symptom, i) => (
              <span
                key={i}
                className="tag"
                onClick={() => toggleSymptom(symptom)}
                style={{ margin: "5px" }}
              >
                {symptom} ×
              </span>
            ))}
            {selectedSymptoms.length === 0 && <p>No symptoms selected</p>}
          </div>

          {/* Available Symptoms */}
          <h3 style={{ textAlign: "center" }}>Available Symptoms</h3>
          <div className="symptom-grid">
            {filteredSymptoms.map((symptom, i) => (
              <div
                key={i}
                onClick={() => toggleSymptom(symptom)}
                className={`symptom-btn ${selectedSymptoms.includes(symptom) ? "selected" : ""}`}
              >
                {symptom}
              </div>
            ))}
          </div>

          {/* Age & Gender */}
          <div style={{ display: "flex", gap: "15px", marginTop: "30px", flexWrap: "wrap" }}>
            <input
              type="number"
              placeholder="Age"
              value={age}
              onChange={(e) => setAge(e.target.value)}
            />

            <select value={gender} onChange={(e) => setGender(e.target.value)}>
              <option value="male">Male</option>
              <option value="female">Female</option>
              <option value="other">Other</option>
            </select>
          </div>

          {/* Severity */}
          <div style={{ marginTop: "20px" }}>
            <label>Severity: {severity}/5</label>
            <input
              type="range"
              min="1"
              max="5"
              value={severity}
              onChange={(e) => setSeverity(parseInt(e.target.value))}
            />
          </div>

          {/* Submit Button */}
          <div style={{ textAlign: "center", marginTop: "30px" }}>
            <button type="submit" disabled={!selectedSymptoms.length || loading}>
              {loading ? "Analyzing..." : "Get Predictions"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default SymptomSelection;

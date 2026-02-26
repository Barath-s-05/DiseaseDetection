import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';

const ResultsPage = () => {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const storedResults = localStorage.getItem('predictionResults');
    if (storedResults) {
      setResults(JSON.parse(storedResults));
    }
    setLoading(false);
  }, []);

  if (loading) {
    return (
      <div className="page-wrapper">
        <div className="card" style={{ textAlign: "center" }}>
          <p>Loading results...</p>
        </div>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="page-wrapper">
        <div className="card" style={{ textAlign: "center" }}>
          <h2>No Results Found</h2>
          <p>Please go back and complete the symptom selection.</p>
          <Link to="/symptoms">
            <button>Go Back</button>
          </Link>
        </div>
      </div>
    );
  }

  const { predictions, disclaimer } = results;

  return (
    <div className="page-wrapper">
      <div className="card">

        <h1 style={{ textAlign: "center" }}>Disease Prediction Results</h1>
        <p style={{ textAlign: "center", marginBottom: "30px", color: "#555" }}>
          AI-powered analysis based on your symptoms
        </p>

        {predictions.map((prediction, index) => {
          const probabilityValue = parseFloat(prediction.probability.replace('%', ''));

          return (
            <div key={index} style={{ marginBottom: "35px" }}>
              <h2 style={{ textAlign: "center" }}>{prediction.disease}</h2>
              <p style={{ textAlign: "center", fontWeight: "bold", color: "#2563eb" }}>
                {prediction.probability}
              </p>

              <div className="probability-bar" style={{ margin: "10px 0" }}>
                <div
                  className="probability-fill"
                  style={{ width: `${probabilityValue}%` }}
                ></div>
              </div>

              <p style={{ marginTop: "10px" }}>{prediction.description}</p>

              <ul style={{ marginTop: "10px" }}>
                {prediction.precautions.map((p, i) => (
                  <li key={i}>{p}</li>
                ))}
              </ul>
            </div>
          );
        })}

        <div className="disclaimer-box" style={{ marginTop: "30px" }}>
          <p>{disclaimer}</p>
        </div>

        <div style={{ textAlign: "center", marginTop: "30px" }}>
          <Link to="/symptoms">
            <button style={{ marginRight: "10px" }}>New Diagnosis</button>
          </Link>
          <Link to="/">
            <button>Home</button>
          </Link>
        </div>

      </div>
    </div>
  );
};

export default ResultsPage;

import React from 'react';
import { Link } from 'react-router-dom';
const circle = {
  width: "50px",
  height: "50px",
  borderRadius: "50%",
  background: "#2563eb",
  color: "white",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  margin: "0 auto 8px",
  fontWeight: "bold",
  fontSize: "18px"
};

const Home = () => {
  return (
  <div className="page-wrapper">
    <div className="card">
      <h1 style={{ textAlign: "center" }}>AI Disease Prediction System</h1>

      <p style={{ textAlign: "center", maxWidth: "700px", margin: "0 auto 30px" }}>
        Predict possible diseases based on your symptoms using advanced machine learning algorithms.
        This tool is for educational purposes only and not a substitute for professional medical advice.
      </p>

      <h2 style={{ textAlign: "center", marginBottom: "20px" }}>How it works:</h2>

      <div style={{
        display: "flex",
        justifyContent: "center",
        gap: "60px",
        textAlign: "center",
        marginBottom: "30px"
      }}>
        <div>
          <div style={circle}>1</div>
          <p>Select Symptoms</p>
        </div>
        <div>
          <div style={circle}>2</div>
          <p>AI Analysis</p>
        </div>
        <div>
          <div style={circle}>3</div>
          <p>Get Results</p>
        </div>
      </div>

      <div style={{ textAlign: "center" }}>
        <Link to="/symptoms">
          <button>Start Diagnosis</button>
        </Link>
      </div>

      <div className="disclaimer-box" style={{ marginTop: "30px" }}>
        <strong>Disclaimer:</strong> This application is an AI-based educational tool and does not provide medical diagnosis. Always consult a healthcare professional.
      </div>
    </div>
  </div>
);

};

export default Home;

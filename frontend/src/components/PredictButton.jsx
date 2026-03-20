import React from 'react'

export default function PredictButton({ disabled, loading, onClick }) {
  return (
    <div className="predict-section">
      <button
        className={`predict-btn${loading ? ' loading' : ''}`}
        disabled={disabled || loading}
        onClick={onClick}
      >
        {loading ? 'Predicting…' : 'Predict Matchup'}
      </button>
    </div>
  )
}

import React from 'react'

export default function PredictButton({ disabled, loading, onClick }) {
  return (
    <div className="predict-section">
      <button
        className={`predict-btn${loading ? ' loading' : ''}`}
        disabled={disabled || loading}
        onClick={onClick}
        data-umami-event="predict-click"
      >
        {loading ? 'Predicting…' : 'Predict Matchup'}
      </button>
    </div>
  )
}

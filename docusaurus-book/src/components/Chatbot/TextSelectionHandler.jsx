import React, { useState, useEffect, useCallback } from 'react';
import ReactDOM from 'react-dom';

const TextSelectionHandler = ({ onAskAssistant }) => {
  const [selection, setSelection] = useState(null);
  const [tooltipPosition, setTooltipPosition] = useState({ top: 0, left: 0 });

  const handleSelection = useCallback(() => {
    const activeSelection = window.getSelection();
    
    // Only handle selection inside the book content or main area to avoid UI noise
    // Check if selection exists and is valid
    if (!activeSelection) return;

    // Safe string conversion
    const rawText = activeSelection.toString();
    const text = rawText ? rawText.trim() : '';

    if (text.length > 0) {
      const range = activeSelection.getRangeAt(0);
      const rect = range.getBoundingClientRect();
      
      // Calculate position (centered above selection)
      setTooltipPosition({
        top: rect.top + window.scrollY - 40, // 40px above
        left: rect.left + window.scrollX + (rect.width / 2) - 50, // Center approx
      });
      setSelection(text);
    } else {
      setSelection(null);
    }
  }, []);

  // Listen for mouseup to detect selection
  useEffect(() => {
    document.addEventListener('mouseup', handleSelection);
    return () => {
      document.removeEventListener('mouseup', handleSelection);
    };
  }, [handleSelection]);

  if (!selection) return null;

  // Render tooltip using Portal to ensure it sits on top of everything
  return ReactDOM.createPortal(
    <div
      style={{
        position: 'absolute',
        top: tooltipPosition.top,
        left: tooltipPosition.left,
        backgroundColor: '#2c5f9e',
        color: 'white',
        padding: '8px 12px',
        borderRadius: '4px',
        cursor: 'pointer',
        zIndex: 10000,
        boxShadow: '0 2px 5px rgba(0,0,0,0.2)',
        fontSize: '14px',
        fontWeight: 'bold',
        pointerEvents: 'auto',
        userSelect: 'none',
      }}
      onMouseDown={(e) => {
        e.preventDefault(); // Prevent clearing selection immediately
        e.stopPropagation();
        onAskAssistant(selection);
        setSelection(null); 
        // window.getSelection().removeAllRanges(); // Keep selection for context if user wants
      }}
    >
      Ask Assistant 🤖
    </div>,
    document.body
  );
};

export default TextSelectionHandler;

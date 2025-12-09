import React, { useState, useCallback } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';
import Chatbot from './Chatbot';
import TextSelectionHandler from './TextSelectionHandler';

function ChatButtonContent() {
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [selectedText, setSelectedText] = useState(null);

  const toggleChat = useCallback(() => {
    setIsChatOpen((prev) => !prev);
  }, []);

  const handleAskAssistant = useCallback((text) => {
    setSelectedText(text);
    setIsChatOpen(true);
  }, []);

  const handleClearSelectedText = useCallback(() => {
    setSelectedText(null);
  }, []);

  return (
    <>
      <TextSelectionHandler onAskAssistant={handleAskAssistant} />

      <button
        onClick={toggleChat}
        style={{
          position: 'fixed',
          bottom: '20px',
          right: '20px',
          backgroundColor: '#007bff',
          color: 'white',
          border: 'none',
          borderRadius: '50%',
          width: '60px',
          height: '60px',
          fontSize: '24px',
          cursor: 'pointer',
          boxShadow: '0 2px 10px rgba(0,0,0,0.2)',
          zIndex: 1000,
        }}
      >
        {isChatOpen ? '✕' : '💬'}
      </button>

      {isChatOpen && (
        <div
          style={{
            position: 'fixed',
            bottom: '90px',
            right: '20px',
            zIndex: 999,
          }}
        >
          <Chatbot 
            selectedText={selectedText}
            onClearSelectedText={handleClearSelectedText}
          />
        </div>
      )}
    </>
  );
}

export default function ChatButton() {
  return (
    <BrowserOnly fallback={null}>
      {() => <ChatButtonContent />}
    </BrowserOnly>
  );
}
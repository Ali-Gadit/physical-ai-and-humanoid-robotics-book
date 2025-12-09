import React, { useEffect, useCallback, useState } from 'react';
import Head from '@docusaurus/Head';
import { ChatKit, useChatKit } from '@openai/chatkit-react';

const LOCAL_STORAGE_THREAD_ID_KEY = 'chatkit_thread_id';

// Define styles based on Docusaurus theme variables
const PRIMARY_COLOR = '#2c5f9e';
const FONT_COLOR_BASE = '#333333';
const FONT_FAMILY_UI = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif';

// Responsive container - changes based on screen size
const containerStyle = {
  height: '600px',
  width: '400px',
  maxWidth: '100%', // Added for responsiveness
  maxHeight: '90vh', // Added for responsiveness
  border: `1px solid ${PRIMARY_COLOR}`,
  borderRadius: '8px',
  overflow: 'hidden',
  backgroundColor: '#ffffff',
  display: 'flex',
  flexDirection: 'column',
  boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
  fontFamily: FONT_FAMILY_UI,
  color: FONT_COLOR_BASE,
  // Responsive properties
  '@media (max-width: 768px)': {
    height: 'calc(100vh - 200px)', // Full height minus some padding on mobile
    width: 'calc(100vw - 32px)', // Full width minus padding
    margin: '0 auto',
  },
  '@media (min-width: 769px) and (max-width: 1024px)': {
    height: '70vh',
    width: '500px',
  }
};

// Inline style object for responsive container
const getContainerStyle = (isMobile, isTablet) => ({
  height: isMobile ? 'calc(100vh - 200px)' : isTablet ? '70vh' : '600px',
  width: isMobile ? 'calc(100vw - 32px)' : isTablet ? '500px' : '400px',
  maxWidth: '100%',
  maxHeight: '90vh',
  border: `1px solid ${PRIMARY_COLOR}`,
  borderRadius: '8px',
  overflow: 'hidden',
  backgroundColor: '#ffffff',
  display: 'flex',
  flexDirection: 'column',
  boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
  fontFamily: FONT_FAMILY_UI,
  color: FONT_COLOR_BASE,
  margin: '0 auto',
});

const headerStyle = {
  padding: '12px 16px',
  borderBottom: `1px solid ${PRIMARY_COLOR}`,
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  backgroundColor: PRIMARY_COLOR,
  color: '#ffffff',
  fontSize: '16px',
  flexShrink: 0, // Prevent header from shrinking
};

const clearButtonStyle = {
  background: 'none',
  border: 'none',
  color: '#ffffff',
  cursor: 'pointer',
  fontSize: '14px',
  marginLeft: '10px',
  padding: '4px 8px',
  borderRadius: '4px',
  transition: 'background-color 0.2s ease',
  whiteSpace: 'nowrap', // Prevent button text from wrapping
};

const chatContainerStyle = {
  flexGrow: 1,
  position: 'relative',
  height: '100%',
  width: '100%',
  backgroundColor: '#f8f8f8',
  overflow: 'hidden', // Important for responsive layout
};

const errorBannerStyle = {
  padding: '8px',
  backgroundColor: '#ffdddd',
  color: '#cc0000',
  borderBottom: '1px solid #ff0000',
  textAlign: 'center',
  fontSize: '0.9em',
  flexShrink: 0, // Prevent error banner from shrinking
};

function Chatbot({ selectedText, onClearSelectedText }) {
  const [initialThread, setInitialThread] = useState(null);
  const [isReady, setIsReady] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  const [windowSize, setWindowSize] = useState({
    width: typeof window !== 'undefined' ? window.innerWidth : 1200,
    height: typeof window !== 'undefined' ? window.innerHeight : 800,
  });

  // Detect screen size for responsiveness
  const isMobile = windowSize.width < 768;
  const isTablet = windowSize.width >= 768 && windowSize.width < 1024;

  // Handle window resize
  useEffect(() => {
    if (typeof window === 'undefined') return;

    const handleResize = () => {
      setWindowSize({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    };

    window.addEventListener('resize', handleResize);
    handleResize(); // Set initial size

    return () => window.removeEventListener('resize', handleResize);
  }, []);

  useEffect(() => {
    const savedThread = typeof window !== 'undefined' 
      ? localStorage.getItem(LOCAL_STORAGE_THREAD_ID_KEY) 
      : null;
    setInitialThread(savedThread);
    setIsReady(true);
  }, []);

  const { control, threadId, setComposerValue, focusComposer } = useChatKit({
    api: {
      url: 'https://rag-chatbot-backend-q1x0.onrender.com/chatkit',
      domainKey: 'domain_pk_6935b0215f8c81908eb11899c2fc88f70b4b0cc7d1f97a9d',
    },
    initialThreadId: initialThread || undefined,
    theme: {
      colorScheme: 'light',
      color: {
        accent: { primary: PRIMARY_COLOR, level: 1 },
      },
      typography: { fontFamily: FONT_FAMILY_UI },
    },
    startScreen: {
      greeting: "Hello! I'm your AI tutor for the Physical AI and Humanoid Robotics textbook. What can I help you with today?",
      prompts: [
        { label: "What is Physical AI?", prompt: "What is Physical AI?" },
        { label: "Explain Humanoid Robotics", prompt: "Explain Humanoid Robotics" },
        { label: "Tell me about ROS2", prompt: "Tell me about ROS2" },
      ],
    },
    onError: ({ error }) => {
      console.error('ChatKit error:', error);
      setErrorMessage("Oops! Something went wrong. Please try again or check your network connection.");
    },
  });

  // Effect to save the threadId to localStorage whenever it changes
  useEffect(() => {
    if (threadId && typeof window !== 'undefined') {
      localStorage.setItem(LOCAL_STORAGE_THREAD_ID_KEY, threadId);
    }
  }, [threadId]);

  // Effect to handle pre-filling the composer with selected text
  useEffect(() => {
    if (selectedText) {
      if (typeof setComposerValue === 'function') {
        console.log('Chatbot: Attempting to set composer value with:', selectedText);
        
        // Small delay to ensure the composer UI is mounted and ready to receive input
        const timer = setTimeout(() => {
          try {
            if (focusComposer) focusComposer();
            // Pass an object with 'text' property, as expected by ChatKit internals
            setComposerValue({ text: String(selectedText) });
            console.log('Chatbot: Successfully set composer value');
          } catch (err) {
            console.error('Chatbot: Error setting composer value:', err);
          }
          
          // Clear the selectedText prop after use
          if (onClearSelectedText) {
            onClearSelectedText();
          }
        }, 100);

        return () => clearTimeout(timer);
      } else {
        console.warn('Chatbot: selectedText present but setComposerValue is not available', { setComposerValue });
      }
    }
  }, [selectedText, setComposerValue, focusComposer, onClearSelectedText]);

  const clearConversation = useCallback(() => {
    if (typeof window !== 'undefined') {
      localStorage.removeItem(LOCAL_STORAGE_THREAD_ID_KEY);
    }
    control.setThreadId(null); 
    setErrorMessage(null);
  }, [control]);

  if (!isReady) {
    return (
      <div style={getContainerStyle(isMobile, isTablet)}>
        <div style={headerStyle}>
          <span style={{ fontWeight: '600', fontSize: '16px' }}>Textbook Tutor</span>
          <button style={{ ...clearButtonStyle, opacity: 0.5 }} disabled>...</button>
        </div>
        <div style={{ flexGrow: 1, display: 'flex', justifyContent: 'center', alignItems: 'center', color: FONT_COLOR_BASE }}>
          Loading Chatbot UI...
        </div>
      </div>
    );
  }

  return (
    <div style={getContainerStyle(isMobile, isTablet)}>
      <div style={headerStyle}>
        <span style={{ fontWeight: '600', fontSize: isMobile ? '14px' : '16px' }}>
          Textbook Tutor
        </span>
        <button 
          onClick={clearConversation} 
          style={{
            ...clearButtonStyle,
            fontSize: isMobile ? '12px' : '14px',
            padding: isMobile ? '3px 6px' : '4px 8px'
          }}
          onMouseOver={(e) => e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.2)'}
          onMouseOut={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
        >
          Clear Chat
        </button>
      </div>
      {errorMessage && (
        <div style={errorBannerStyle}>
          {errorMessage}
        </div>
      )}
      <div style={chatContainerStyle}>
        <ChatKit control={control} />
      </div>
    </div>
  );
}

export default Chatbot;
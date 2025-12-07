import React, { useEffect, useCallback, useState } from 'react';
import Head from '@docusaurus/Head';
import { ChatKit, useChatKit } from '@openai/chatkit-react';

const LOCAL_STORAGE_THREAD_ID_KEY = 'chatkit_thread_id';

// Define styles based on Docusaurus theme variables (hardcoded for inline usage)
const PRIMARY_COLOR = '#2c5f9e'; // --ifm-color-primary
const FONT_COLOR_BASE = '#333333'; // --ifm-font-color-base
const FONT_FAMILY_UI = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif';
const FONT_FAMILY_CONTENT = 'Georgia, "Times New Roman", serif';


const containerStyle = {
  height: '600px',
  width: '400px',
  border: `1px solid ${PRIMARY_COLOR}`,
  borderRadius: '8px',
  overflow: 'hidden',
  backgroundColor: '#ffffff', // White background for chatbot
  display: 'flex',
  flexDirection: 'column',
  boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
  fontFamily: FONT_FAMILY_UI,
  color: FONT_COLOR_BASE,
};

const headerStyle = {
  padding: '12px 16px',
  borderBottom: `1px solid ${PRIMARY_COLOR}`,
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  backgroundColor: PRIMARY_COLOR,
  color: '#ffffff', // White text on primary background
  fontSize: '16px',
};

const clearButtonStyle = {
  background: 'none',
  border: 'none',
  color: '#ffffff', // White text for clear button on primary background
  cursor: 'pointer',
  fontSize: '14px',
  marginLeft: '10px',
  padding: '4px 8px',
  borderRadius: '4px',
  transition: 'background-color 0.2s ease',
};

const chatContainerStyle = {
  flexGrow: 1,
  position: 'relative',
  height: '100%',
  width: '100%',
  backgroundColor: '#f8f8f8', // Light background for chat area
};

const errorBannerStyle = {
  padding: '8px',
  backgroundColor: '#ffdddd',
  color: '#cc0000',
  borderBottom: '1px solid #ff0000',
  textAlign: 'center',
  fontSize: '0.9em',
};

function Chatbot({ selectedText, onClearSelectedText }) {
  const [initialThread, setInitialThread] = useState(null);
  const [isReady, setIsReady] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null); // New state for error messages


  useEffect(() => {
    const savedThread = typeof window !== 'undefined' 
      ? localStorage.getItem(LOCAL_STORAGE_THREAD_ID_KEY) 
      : null;
    setInitialThread(savedThread);
    setIsReady(true);
  }, []);

  const { control, threadId } = useChatKit({
    api: {
      url: 'https://rag-chatbot-backend-q1x0.onrender.com/chatkit',
      domainKey: 'ali-gadit.github.io', // Updated to your GitHub Pages domain
    },
    initialThreadId: initialThread || undefined,
    theme: {
      colorScheme: 'light', // Force light theme for consistency with textbook, or detect Docusaurus theme
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
    // Implement error handling
    onError: ({ error }) => {
      console.error('ChatKit error:', error);
      // Display a user-friendly message
      setErrorMessage("Oops! Something went wrong. Please try again or check your network connection.");
      // You might want to clear the error after some time
      // setTimeout(() => setErrorMessage(null), 5000);
    },
  });

  // Effect to save the threadId to localStorage whenever it changes
  useEffect(() => {
    if (threadId && typeof window !== 'undefined') {
      localStorage.setItem(LOCAL_STORAGE_THREAD_ID_KEY, threadId);
    }
  }, [threadId]);

  const clearConversation = useCallback(() => {
    if (typeof window !== 'undefined') {
      localStorage.removeItem(LOCAL_STORAGE_THREAD_ID_KEY);
    }
    control.setThreadId(undefined); 
    setErrorMessage(null); // Clear any error message on new conversation
  }, [control]);

  if (!isReady) {
    return (
      <div style={containerStyle}>
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
    <div style={containerStyle}>
      {/* Inject CDN script via Head for robustness */}
      <Head>
        <script src="https://cdn.platform.openai.com/deployments/chatkit/chatkit.js" type="module" async></script>
      </Head>
      
      <div style={headerStyle}>
        <span style={{ fontWeight: '600', fontSize: '16px' }}>Textbook Tutor</span>
        <button 
          onClick={clearConversation} 
          style={clearButtonStyle}
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
import React, { useEffect, useCallback, useState } from 'react';
import Head from '@docusaurus/Head';
import { ChatKit, useChatKit } from '@openai/chatkit-react';
import useIsBrowser from '@docusaurus/useIsBrowser';
import { useColorMode } from '@docusaurus/theme-common';

const LOCAL_STORAGE_THREAD_ID_KEY = 'chatkit_thread_id';

// Responsive container styles
const getContainerStyle = (isMobile, isTablet) => ({
  height: isMobile ? 'calc(100vh - 100px)' : isTablet ? '80vh' : '600px',
  width: isMobile ? 'calc(100vw - 32px)' : isTablet ? 'calc(100vw - 64px)' : '400px',
  maxWidth: isMobile ? '100%' : isTablet ? '600px' : '400px',
  maxHeight: isMobile ? 'calc(100vh - 100px)' : isTablet ? '80vh' : '600px',
  minHeight: isMobile ? '400px' : '450px',
  minWidth: isMobile ? '280px' : '320px',
  border: `1px solid var(--ifm-color-primary)`,
  borderRadius: '12px',
  overflow: 'hidden',
  backgroundColor: 'var(--ifm-background-surface-color)',
  display: 'flex',
  flexDirection: 'column',
  boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
  fontFamily: 'var(--ifm-font-family-base)',
  color: 'var(--ifm-font-color-base)',
  margin: isMobile ? '16px auto' : '0 auto',
  position: 'relative',
  transition: 'all 0.3s ease',
});

const headerStyle = {
  padding: '16px 20px',
  borderBottom: `1px solid var(--ifm-color-emphasis-300)`,
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  backgroundColor: 'var(--ifm-color-primary)',
  color: '#ffffff',
  fontSize: 'clamp(16px, 2vw, 18px)',
  flexShrink: 0,
  minHeight: '56px',
};

const clearButtonStyle = {
  background: 'rgba(255, 255, 255, 0.2)',
  border: '1px solid rgba(255, 255, 255, 0.3)',
  color: '#ffffff',
  cursor: 'pointer',
  fontSize: 'clamp(12px, 1.5vw, 14px)',
  padding: '6px 12px',
  borderRadius: '6px',
  transition: 'all 0.2s ease',
  whiteSpace: 'nowrap',
  marginLeft: '12px',
};

const chatContainerStyle = {
  flexGrow: 1,
  position: 'relative',
  height: '100%',
  width: '100%',
  backgroundColor: 'var(--ifm-background-color)',
  overflow: 'hidden',
};

const errorBannerStyle = {
  padding: '10px 16px',
  backgroundColor: 'var(--ifm-color-danger-lightest)',
  color: 'var(--ifm-color-danger-dark)',
  borderBottom: '1px solid var(--ifm-color-danger-light)',
  textAlign: 'center',
  fontSize: 'clamp(0.8em, 2vw, 0.9em)',
  flexShrink: 0,
};

const loadingStyle = {
  flexGrow: 1,
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
  color: 'var(--ifm-font-color-base)',
  fontSize: 'clamp(14px, 2vw, 16px)',
  padding: '20px',
  textAlign: 'center',
};

function Chatbot({ selectedText, onClearSelectedText }) {
  const isBrowser = useIsBrowser();
  const { colorMode } = useColorMode();
  const [initialThread, setInitialThread] = useState(null);
  const [isReady, setIsReady] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  const [windowSize, setWindowSize] = useState({
    width: isBrowser ? window.innerWidth : 1200,
    height: isBrowser ? window.innerHeight : 800,
  });

  // Responsive breakpoints
  const isMobile = windowSize.width < 768;
  const isTablet = windowSize.width >= 768 && windowSize.width < 1024;

  // Handle window resize
  useEffect(() => {
    if (!isBrowser) return;

    const handleResize = () => {
      setWindowSize({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    };

    window.addEventListener('resize', handleResize);
    handleResize(); // Set initial size

    return () => window.removeEventListener('resize', handleResize);
  }, [isBrowser]);

  useEffect(() => {
    if (!isBrowser) return;

    const savedThread = localStorage.getItem(LOCAL_STORAGE_THREAD_ID_KEY);
    setInitialThread(savedThread);
    setIsReady(true);
  }, [isBrowser]);

  const { control, threadId } = useChatKit({
    api: {
      url: 'https://rag-chatbot-backend-q1x0.onrender.com/chatkit',
      domainKey: 'domain_pk_6935b0215f8c81908eb11899c2fc88f70b4b0cc7d1f97a9d',
    },
    initialThreadId: initialThread || undefined,
    theme: {
      colorScheme: colorMode,
      color: {
        accent: { 
          primary: 'var(--ifm-color-primary)', 
          level: 1 
        },
      },
      typography: { 
        fontFamily: 'var(--ifm-font-family-base)',
        fontSize: {
          base: isMobile ? '14px' : '16px',
        }
      },
      spacing: {
        unit: isMobile ? 3 : 4,
      }
    },
    startScreen: {
      greeting: "Hello! I'm your AI tutor for the Physical AI and Humanoid Robotics textbook. What can I help you with today?",
      prompts: [
        { 
          label: "What is Physical AI?", 
          prompt: "What is Physical AI?" 
        },
        { 
          label: "Explain Humanoid Robotics", 
          prompt: "Explain Humanoid Robotics" 
        },
        { 
          label: "Tell me about ROS2", 
          prompt: "Tell me about ROS2" 
        },
      ],
    },
    onError: ({ error }) => {
      console.error('ChatKit error:', error);
      setErrorMessage("Oops! Something went wrong. Please try again or check your network connection.");
    },
  });

  // Effect to save the threadId to localStorage whenever it changes
  useEffect(() => {
    if (threadId && isBrowser) {
      localStorage.setItem(LOCAL_STORAGE_THREAD_ID_KEY, threadId);
    }
  }, [threadId, isBrowser]);

  const clearConversation = useCallback(() => {
    if (isBrowser) {
      localStorage.removeItem(LOCAL_STORAGE_THREAD_ID_KEY);
    }
    control.setThreadId(undefined);
    setErrorMessage(null);
  }, [control, isBrowser]);

  if (!isReady) {
    return (
      <div style={getContainerStyle(isMobile, isTablet)}>
        <div style={headerStyle}>
          <span style={{ fontWeight: '600', fontSize: 'clamp(16px, 2vw, 18px)' }}>
            Textbook Tutor
          </span>
          <button 
            style={{ ...clearButtonStyle, opacity: 0.5 }} 
            disabled
          >
            ...
          </button>
        </div>
        <div style={loadingStyle}>
          Loading Chatbot UI...
        </div>
      </div>
    );
  }

  return (
    <div style={getContainerStyle(isMobile, isTablet)}>
      {/* Inject CDN script via Head for robustness */}
      <Head>
        <script 
          src="https://cdn.platform.openai.com/deployments/chatkit/chatkit.js" 
          type="module" 
          async 
        />
        {/* Add viewport meta for mobile responsiveness */}
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      </Head>
      
      <div style={headerStyle}>
        <span style={{ 
          fontWeight: '600', 
          fontSize: 'clamp(16px, 2vw, 18px)',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap'
        }}>
          Textbook Tutor
        </span>
        <button 
          onClick={clearConversation} 
          style={clearButtonStyle}
          onMouseOver={(e) => e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.3)'}
          onMouseOut={(e) => e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.2)'}
          onFocus={(e) => e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.3)'}
          onBlur={(e) => e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.2)'}
          aria-label="Clear chat conversation"
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
        <ChatKit 
          control={control} 
          style={{
            '--chatkit-font-size-base': isMobile ? '14px' : '16px',
            '--chatkit-spacing-unit': isMobile ? '3px' : '4px',
          }}
        />
      </div>
    </div>
  );
}

export default Chatbot;
import React, { useEffect, useCallback, useState } from 'react';
import Head from '@docusaurus/Head';
import { ChatKit, useChatKit } from '@openai/chatkit-react';

const LOCAL_STORAGE_THREAD_ID_KEY = 'chatkit_thread_id';

// Default theme colors for fallback
const DEFAULT_PRIMARY_COLOR = '#2c5f9e';
const DEFAULT_BACKGROUND_COLOR = '#ffffff';
const DEFAULT_FONT_COLOR = '#333333';
const DEFAULT_DANGER_COLOR = '#ff4444';
const DEFAULT_DANGER_LIGHT = '#ffdddd';

// Responsive container styles
const getContainerStyle = (isMobile, isTablet) => ({
  height: isMobile ? 'calc(100vh - 100px)' : isTablet ? '80vh' : '600px',
  width: isMobile ? 'calc(100vw - 32px)' : isTablet ? 'calc(100vw - 64px)' : '400px',
  maxWidth: isMobile ? '100%' : isTablet ? '600px' : '400px',
  maxHeight: isMobile ? 'calc(100vh - 100px)' : isTablet ? '80vh' : '600px',
  minHeight: isMobile ? '400px' : '450px',
  minWidth: isMobile ? '280px' : '320px',
  border: `1px solid ${DEFAULT_PRIMARY_COLOR}`,
  borderRadius: '12px',
  overflow: 'hidden',
  backgroundColor: DEFAULT_BACKGROUND_COLOR,
  display: 'flex',
  flexDirection: 'column',
  boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
  fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
  color: DEFAULT_FONT_COLOR,
  margin: isMobile ? '16px auto' : '0 auto',
  position: 'relative',
  transition: 'all 0.3s ease',
});

const getHeaderStyle = (isDarkMode = false) => ({
  padding: '16px 20px',
  borderBottom: `1px solid ${isDarkMode ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'}`,
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  backgroundColor: DEFAULT_PRIMARY_COLOR,
  color: '#ffffff',
  fontSize: 'clamp(16px, 2vw, 18px)',
  flexShrink: 0,
  minHeight: '56px',
});

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

const chatContainerStyle = (isDarkMode = false) => ({
  flexGrow: 1,
  position: 'relative',
  height: '100%',
  width: '100%',
  backgroundColor: isDarkMode ? '#1a1a1a' : '#f8f8f8',
  overflow: 'hidden',
});

const errorBannerStyle = (isDarkMode = false) => ({
  padding: '10px 16px',
  backgroundColor: isDarkMode ? 'rgba(255, 68, 68, 0.2)' : DEFAULT_DANGER_LIGHT,
  color: isDarkMode ? '#ff8888' : DEFAULT_DANGER_COLOR,
  borderBottom: `1px solid ${isDarkMode ? 'rgba(255, 68, 68, 0.3)' : '#ffaaaa'}`,
  textAlign: 'center',
  fontSize: 'clamp(0.8em, 2vw, 0.9em)',
  flexShrink: 0,
});

const loadingStyle = (isDarkMode = false) => ({
  flexGrow: 1,
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
  color: isDarkMode ? '#ffffff' : DEFAULT_FONT_COLOR,
  fontSize: 'clamp(14px, 2vw, 16px)',
  padding: '20px',
  textAlign: 'center',
});

function Chatbot({ selectedText, onClearSelectedText }) {
  const [initialThread, setInitialThread] = useState(null);
  const [isReady, setIsReady] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  const [windowSize, setWindowSize] = useState({
    width: typeof window !== 'undefined' ? window.innerWidth : 1200,
    height: typeof window !== 'undefined' ? window.innerHeight : 800,
  });
  
  // Detect dark mode from localStorage or system preference
  const [isDarkMode, setIsDarkMode] = useState(false);

  // Responsive breakpoints
  const isMobile = windowSize.width < 768;
  const isTablet = windowSize.width >= 768 && windowSize.width < 1024;

  // Handle window resize and dark mode detection
  useEffect(() => {
    if (typeof window === 'undefined') return;

    const handleResize = () => {
      setWindowSize({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    };

    // Check for dark mode
    const checkDarkMode = () => {
      // First check localStorage
      const storedMode = localStorage.getItem('theme');
      if (storedMode === 'dark' || storedMode === 'light') {
        setIsDarkMode(storedMode === 'dark');
      } else {
        // Check system preference
        const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        setIsDarkMode(systemPrefersDark);
      }
    };

    // Listen for theme changes
    const themeChangeHandler = (e) => {
      setIsDarkMode(e.matches);
    };
    
    const darkModeMediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    darkModeMediaQuery.addListener(themeChangeHandler);

    window.addEventListener('resize', handleResize);
    
    // Initial calls
    handleResize();
    checkDarkMode();

    return () => {
      window.removeEventListener('resize', handleResize);
      darkModeMediaQuery.removeListener(themeChangeHandler);
    };
  }, []);

  useEffect(() => {
    if (typeof window === 'undefined') return;

    const savedThread = localStorage.getItem(LOCAL_STORAGE_THREAD_ID_KEY);
    setInitialThread(savedThread);
    setIsReady(true);
  }, []);

  const { control, threadId } = useChatKit({
    api: {
      url: 'https://rag-chatbot-backend-q1x0.onrender.com/chatkit',
      domainKey: 'domain_pk_6935b0215f8c81908eb11899c2fc88f70b4b0cc7d1f97a9d',
    },
    initialThreadId: initialThread || undefined,
    theme: {
      colorScheme: isDarkMode ? 'dark' : 'light',
      color: {
        accent: { 
          primary: DEFAULT_PRIMARY_COLOR, 
          level: 1 
        },
      },
      typography: { 
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
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
    if (threadId && typeof window !== 'undefined') {
      localStorage.setItem(LOCAL_STORAGE_THREAD_ID_KEY, threadId);
    }
  }, [threadId]);

  const clearConversation = useCallback(() => {
    if (typeof window !== 'undefined') {
      localStorage.removeItem(LOCAL_STORAGE_THREAD_ID_KEY);
    }
    control.setThreadId(undefined);
    setErrorMessage(null);
  }, [control]);

  if (!isReady) {
    return (
      <div style={getContainerStyle(isMobile, isTablet)}>
        <div style={getHeaderStyle(isDarkMode)}>
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
        <div style={loadingStyle(isDarkMode)}>
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
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
      </Head>
      
      <div style={getHeaderStyle(isDarkMode)}>
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
        <div style={errorBannerStyle(isDarkMode)}>
          {errorMessage}
        </div>
      )}
      
      <div style={chatContainerStyle(isDarkMode)}>
        <ChatKit 
          control={control}
        />
      </div>
    </div>
  );
}

export default Chatbot;
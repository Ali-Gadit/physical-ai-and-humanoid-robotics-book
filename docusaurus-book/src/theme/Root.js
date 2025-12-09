import React, { useEffect } from 'react';

export default function Root({ children }) {
  useEffect(() => {
    // Load ChatKit script globally before any component renders
    const script = document.createElement("script");
    script.src = "https://cdn.platform.openai.com/deployments/chatkit/chatkit.js";
    script.type = "module";
    script.async = false; // IMPORTANT: must load before React renders ChatKit
    document.head.appendChild(script);
  }, []);

  return <>{children}</>;
}

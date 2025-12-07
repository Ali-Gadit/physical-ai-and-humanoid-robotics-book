import React from 'react';
import Layout from '@theme-original/Layout';
import ChatButton from '@site/src/components/Chatbot/ChatButton';

export default function LayoutWrapper(props) {
  return (
    <>
      <Layout {...props} />
      <ChatButton />
    </>
  );
}
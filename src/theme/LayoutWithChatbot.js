import React from 'react';
import Layout from '@theme/Layout';
import Chatbot from '@site/src/components/Chatbot';
import styles from './LayoutWithChatbot.module.css';

function LayoutWithChatbot(props) {
  const { children, ...layoutProps } = props;

  return (
    <Layout {...layoutProps}>
      <div className={styles.mainContainer}>
        <div className={styles.content}>
          {children}
        </div>
        <div className={styles.chatbotPanel}>
          <Chatbot />
        </div>
      </div>
    </Layout>
  );
}

export default LayoutWithChatbot;
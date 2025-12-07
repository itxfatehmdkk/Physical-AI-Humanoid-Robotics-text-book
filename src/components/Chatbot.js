import React, { useState, useEffect, useRef } from 'react';
import './Chatbot.css';
import apiConfig from '../config/apiConfig';

const Chatbot = () => {
  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedText, setSelectedText] = useState('');
  const messagesEndRef = useRef(null);

  // Create a new chat session when component mounts
  useEffect(() => {
    createNewSession();
  }, []);

  // Scroll to bottom of messages when new messages are added
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Function to create a new chat session
  const createNewSession = async () => {
    try {
      const API_URL = apiConfig.getApiEndpoint('/api/v1/chat/session');

      const response = await fetch(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        const data = await response.json();
        setSessionId(data.session_id);
      } else {
        console.error('Failed to create session');
      }
    } catch (error) {
      console.error('Error creating session:', error);
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Function to handle sending a message
  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || !sessionId || isLoading) return;

    // Add user message to the chat
    const userMessage = {
      id: Date.now(),
      sender: 'user',
      content: inputValue,
      timestamp: new Date().toISOString(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const API_URL = apiConfig.getApiEndpoint('/api/v1/chat/message/stream');

      // Send message to backend
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          message: inputValue,
          selected_text: selectedText || null,
        }),
      });

      if (response.ok && response.body) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let done = false;
        let botMessageId = Date.now();
        let botMessageContent = '';
        let citations = [];

        // Add a placeholder for the bot's response
        setMessages(prev => [...prev, {
          id: botMessageId,
          sender: 'chatbot',
          content: '',
          timestamp: new Date().toISOString(),
          citations: []
        }]);

        while (!done) {
          const { value, done: readerDone } = await reader.read();
          done = readerDone;

          if (value) {
            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                try {
                  const data = JSON.parse(line.slice(6)); // Remove 'data: ' prefix

                  if (data.type === 'chunk') {
                    // Update the bot's message with new content
                    botMessageContent += data.content;
                    setMessages(prev => prev.map(msg => 
                      msg.id === botMessageId 
                        ? { ...msg, content: botMessageContent }
                        : msg
                    ));
                  } else if (data.type === 'final') {
                    // Final chunk with citations
                    citations = data.citations || [];
                    setMessages(prev => prev.map(msg => 
                      msg.id === botMessageId 
                        ? { ...msg, citations }
                        : msg
                    ));
                  } else if (data.type === 'error') {
                    console.error('Error from backend:', data.message);
                  }
                } catch (e) {
                  // Skip lines that aren't valid JSON
                }
              }
            }
          }
        }
      } else {
        // Handle error response
        setMessages(prev => [...prev, {
          id: Date.now(),
          sender: 'chatbot',
          content: 'Sorry, I encountered an error processing your request.',
          timestamp: new Date().toISOString(),
        }]);
      }
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prev => [...prev, {
        id: Date.now(),
        sender: 'chatbot',
        content: 'Sorry, I encountered an error connecting to the server.',
        timestamp: new Date().toISOString(),
      }]);
    } finally {
      setIsLoading(false);
      setSelectedText(''); // Clear selected text after sending
    }
  };

  // Function to handle text selection
  useEffect(() => {
    const handleSelection = () => {
      const selectedText = window.getSelection().toString().trim();
      if (selectedText) {
        setSelectedText(selectedText);
      }
    };

    document.addEventListener('mouseup', handleSelection);
    return () => {
      document.removeEventListener('mouseup', handleSelection);
    };
  }, []);

  return (
    <div className="chatbot-container">
      <div className="chatbot-header">
        <h3>Physical AI Textbook Assistant</h3>
        {selectedText && (
          <div className="selected-text-preview">
            <strong>Selected text:</strong> "{selectedText.substring(0, 60)}{selectedText.length > 60 ? '...' : ''}"
          </div>
        )}
      </div>
      
      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="welcome-message">
            <p>Hello! I'm your Physical AI & Humanoid Robotics textbook assistant.</p>
            <p>Ask me anything about the content, or select text from the page to ask specific questions about it.</p>
          </div>
        ) : (
          messages.map((message) => (
            <div 
              key={message.id} 
              className={`message ${message.sender === 'user' ? 'user-message' : 'bot-message'}`}
            >
              <div className="message-content">
                {message.content}
              </div>
              
              {message.citations && message.citations.length > 0 && (
                <div className="citations">
                  <strong>Sources:</strong>
                  <ul>
                    {message.citations.map((citation, index) => (
                      <li key={index}>
                        {citation.chapter_title} 
                        {citation.heading && ` - ${citation.heading}`}
                        {citation.source_url && (
                          <a href={citation.source_url} target="_blank" rel="noopener noreferrer"> (View)</a>
                        )}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ))
        )}
        {isLoading && (
          <div className="message bot-message">
            <div className="message-content">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      
      <form className="chat-input-form" onSubmit={handleSendMessage}>
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder={selectedText 
            ? "Ask about the selected text..." 
            : "Ask about Physical AI & Humanoid Robotics..."}
          disabled={isLoading || !sessionId}
        />
        <button 
          type="submit" 
          disabled={!inputValue.trim() || isLoading || !sessionId}
        >
          Send
        </button>
      </form>
    </div>
  );
};

export default Chatbot;
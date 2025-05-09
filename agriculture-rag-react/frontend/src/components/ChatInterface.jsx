import React, { useState, useRef, useEffect } from 'react';
import Message from './Message';
import SourceDocuments from './SourceDocuments';
import DocumentUpload from './DocumentUpload';
import { sendMessage, uploadDocument } from '../api';

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showSources, setShowSources] = useState(false);
  const [currentSources, setCurrentSources] = useState([]);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await sendMessage(input);
      const botMessage = { 
        role: 'assistant', 
        content: response.answer,
        sources: response.sources
      };
      setMessages(prev => [...prev, botMessage]);
      setCurrentSources(response.sources);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = { 
        role: 'assistant', 
        content: 'Sorry, I encountered an error processing your request.'
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (file) => {
    try {
      const response = await uploadDocument(file);
      if (response.success) {
        const systemMessage = {
          role: 'system',
          content: `Document "${file.name}" processed successfully!`
        };
        setMessages(prev => [...prev, systemMessage]);
        return response.preview;
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      const errorMessage = {
        role: 'system',
        content: `Error processing document: ${error.message}`
      };
      setMessages(prev => [...prev, errorMessage]);
    }
    return null;
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h1>Agriculture Document Analysis</h1>
        <DocumentUpload onUpload={handleFileUpload} />
      </div>
      
      <div className="chat-messages">
        {messages.map((message, index) => (
          <Message 
            key={index} 
            role={message.role} 
            content={message.content} 
          />
        ))}
        {isLoading && (
          <Message 
            role="assistant" 
            content={<div className="loading-dots">Analyzing documents<span>.</span><span>.</span><span>.</span></div>} 
          />
        )}
        <div ref={messagesEndRef} />
      </div>
      
      <form onSubmit={handleSendMessage} className="chat-input">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about agriculture documents..."
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading || !input.trim()}>
          Send
        </button>
      </form>
      
      {currentSources.length > 0 && (
        <SourceDocuments 
          sources={currentSources} 
          isVisible={showSources} 
          toggleVisibility={() => setShowSources(!showSources)} 
        />
      )}
    </div>
  );
};

export default ChatInterface;
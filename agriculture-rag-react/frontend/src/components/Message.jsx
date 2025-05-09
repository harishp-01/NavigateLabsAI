import React from 'react';

const Message = ({ role, content }) => {
  const isUser = role === 'user';
  const isSystem = role === 'system';
  
  return (
    <div className={`message ${isUser ? 'user' : isSystem ? 'system' : 'assistant'}`}>
      <div className="message-content">
        {typeof content === 'string' ? content : content}
      </div>
    </div>
  );
};

export default Message;
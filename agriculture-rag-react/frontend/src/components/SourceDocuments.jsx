import React from 'react';

const SourceDocuments = ({ sources, isVisible, toggleVisibility }) => {
  return (
    <div className={`source-documents ${isVisible ? 'visible' : ''}`}>
      <button onClick={toggleVisibility} className="toggle-sources">
        {isVisible ? 'Hide Sources' : 'Show Sources'}
      </button>
      
      {isVisible && (
        <div className="sources-list">
          <h3>Source Documents</h3>
          {sources.map((source, index) => (
            <div key={index} className="source-item">
              <div className="source-header">
                <span className="source-type">{source.type}</span>
                <span className="source-page">Page {source.page_num}</span>
              </div>
              <div className="source-content">{source.content}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default SourceDocuments;
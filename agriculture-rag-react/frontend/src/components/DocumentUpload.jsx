import React, { useState } from 'react';

const DocumentUpload = ({ onUpload }) => {
  const [previewImage, setPreviewImage] = useState(null);

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const preview = await onUpload(file);
    if (preview) {
      setPreviewImage(`${process.env.REACT_APP_API_URL}/api/preview/${preview}`);
    }
  };

  return (
    <div className="document-upload">
      <label className="upload-button">
        Upload Document
        <input 
          type="file" 
          accept=".pdf,.png,.jpg,.jpeg" 
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />
      </label>
      {previewImage && (
        <div className="document-preview">
          <img src={previewImage} alt="Document preview" />
        </div>
      )}
    </div>
  );
};

export default DocumentUpload;
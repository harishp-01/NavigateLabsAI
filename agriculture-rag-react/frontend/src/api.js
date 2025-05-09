const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

export const sendMessage = async (message) => {
  const response = await fetch(`${API_URL}/api/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ message })
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return await response.json();
};

export const uploadDocument = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch(`${API_URL}/api/upload`, {
    method: 'POST',
    body: formData
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return await response.json();
};

export const getPreview = async (filename) => {
  const response = await fetch(`${API_URL}/api/preview/${filename}`, {
    credentials: 'include'
  });
  if (!response.ok) {
    throw new Error('Preview not available');
  }
  return response.url;
};
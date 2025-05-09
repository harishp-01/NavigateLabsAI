# Agriculture Document Analysis Chatbot 🌾🤖  
*A Multi-Modal RAG System for Agricultural Knowledge Retrieval*  

[![Deployed on Vercel](https://img.shields.io/badge/Deployed%20on-Vercel-black?style=for-the-badge&logo=vercel)](https://vercel.com)  
[![Frontend: React](https://img.shields.io/badge/Frontend-React-61DAFB?style=for-the-badge&logo=react)](https://reactjs.org)  
[![Backend: Flask](https://img.shields.io/badge/Backend-Flask-000000?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com)  

---

## Table of Contents 📚
- [Introduction](#introduction)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [System Architecture](#system-architecture)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Deployment](#deployment)
- [Setup & Installation](#setup--installation)
- [Future Enhancements](#future-enhancements)
- [License](#license)

---

## Introduction ✨
The Agriculture Document Analysis Chatbot is an AI-powered RAG system that helps users analyze agricultural documents through:
- **Multi-modal processing** of PDFs and images
- **Natural language Q&A** powered by OpenAI
- **Vercel deployment** with React frontend and Flask backend
- **Source-attributed answers** from documents

---

## Features 🚀
| Feature | Description |
|---------|-------------|
| 📄 Document Upload | PDF & image processing with metadata extraction |
| 🤖 AI Chat Interface | Context-aware question answering |
| 🔍 Multi-Modal Search | Combined text + image vector search |
| 📑 Source References | Page numbers and document sources |
| ⚡ Scalable API | Flask backend with REST endpoints |

---

## Tech Stack 🛠️
**Frontend**  
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react)
![Vite](https://img.shields.io/badge/Vite-B73BFE?style=for-the-badge&logo=vite)
![Vercel](https://img.shields.io/badge/Vercel-000000?style=for-the-badge&logo=vercel)

**Backend**  
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker)

**AI/ML**  
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21F?style=for-the-badge)
![FAISS](https://img.shields.io/badge/FAISS-00C4CC?style=for-the-badge)

---

## System Architecture 🏗️
```mermaid
graph TD
    A[User] -->|Upload| B(React Frontend)
    B -->|REST API| C[Flask Backend]
    C --> D{Document Processor}
    D -->|Text| E[Text Embeddings]
    D -->|Images| F[Image Embeddings]
    E --> G[FAISS Vector DB]
    F --> G
    G --> H[RAG Pipeline]
    H --> I[OpenAI LLM]
    I -->|Answer| B --> A